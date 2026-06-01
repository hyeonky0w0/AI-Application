import os
import json
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, List
import logging

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from dataset import (
    get_dataloaders, denormalize_clothes, denormalize_clothes_batch,
    CLOTHES_KEYS, BODY_DIM,
)
from model import (
    ClothingMeasurementNet, ClothingMeasurementNetCA,
    ClothingMeasurementNetQFormer,
    BaselineMLP, NoFiLMModel, CombinedLoss,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Ablation 실험 설정
#   body_dim 은 BODY_DIM(=13) 으로 통일 — 카테고리 임베딩 포함
# ──────────────────────────────────────────────────────────────
EXPERIMENTS = {
    1: {
        "name":      "exp1_mlp_baseline",
        "desc":      "Exp1: 신체 치수만 입력 (MLP Baseline, 이미지 없음)",
        "model_cls": BaselineMLP,
        "mediapipe": False,
        "lambda1":   0.0,
        "lambda2":   1.0,
    },
    2: {
        "name":      "exp2_image_only",
        "desc":      "Exp2: 이미지만 입력 (FiLM 없음)",
        "model_cls": NoFiLMModel,
        "mediapipe": False,
        "lambda1":   1.0,
        "lambda2":   0.1,
    },
    3: {
        "name":      "exp3_image_body_concat",
        "desc":      "Exp3: 이미지 + body concat (FiLM 없음)",
        "model_cls": NoFiLMModel,
        "mediapipe": False,
        "lambda1":   1.0,
        "lambda2":   0.1,
    },
    4: {
        "name":      "exp4_backbone",
        "desc":      "Exp4: Full model (이미지 + body + FiLM)",
        "model_cls": ClothingMeasurementNet,
        "mediapipe": False,
        "lambda1":   1.0,
        "lambda2":   0.1,
    },
    5: {
        "name":      "exp5_cross_attn",
        "desc":      "Exp5: Cross-Attention 조건화",
        "model_cls": ClothingMeasurementNetCA,
        "mediapipe": False,
        "lambda1":   1.0,
        "lambda2":   0.1,
    },
    6: {
        "name":      "exp6_qformer",
        "desc":      "Exp6: Q-Former 헤드",
        "model_cls": ClothingMeasurementNetQFormer,
        "mediapipe": False,
        "lambda1":   1.0,
        "lambda2":   0.1,
    },
}


# ──────────────────────────────────────────────────────────────
# 평가 함수
# ──────────────────────────────────────────────────────────────
def evaluate(
    model: torch.nn.Module,
    loader,
    criterion: CombinedLoss,
    device: torch.device,
) -> Dict:
    model.eval()
    total_loss = seg_loss_sum = meas_loss_sum = 0.0
    all_pred, all_gt = [], []
    all_cats: List[str] = []
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            image    = batch["image"].to(device)
            body_vec = batch["body_vec"].to(device)   # [B, 13]
            mask_gt  = batch["mask"].to(device)
            clothes  = batch["clothes"].to(device)
            cats     = batch["category"]              # List[str], len=B

            seg_pred, meas_pred = model(image, body_vec)
            loss, sl, ml = criterion(seg_pred, mask_gt, meas_pred, clothes)

            total_loss    += loss.item()
            seg_loss_sum  += sl.item()
            meas_loss_sum += ml.item()
            n_batches     += 1

            all_pred.append(meas_pred.cpu())
            all_gt.append(clothes.cpu())
            all_cats.extend(cats)

    all_pred = torch.cat(all_pred, dim=0)
    all_gt   = torch.cat(all_gt,   dim=0)

    # ② 카테고리별 역변환 후 MAE / RMSE 계산
    pred_cm = denormalize_clothes_batch(all_pred, all_cats)
    gt_cm   = denormalize_clothes_batch(all_gt,   all_cats)

    mae_per_item  = (pred_cm - gt_cm).abs().mean(dim=0)
    rmse_per_item = ((pred_cm - gt_cm)**2).mean(dim=0).sqrt()
    item_names    = [k.split(".")[-1] for k in CLOTHES_KEYS]

    metrics = {
        "total_loss":   total_loss   / max(n_batches, 1),
        "seg_loss":     seg_loss_sum / max(n_batches, 1),
        "meas_loss":    meas_loss_sum / max(n_batches, 1),
        "mae_overall":  mae_per_item.mean().item(),
        "rmse_overall": rmse_per_item.mean().item(),
        "mae_per_item": {
            name: mae_per_item[i].item()
            for i, name in enumerate(item_names)
        },
        "rmse_per_item": {
            name: rmse_per_item[i].item()
            for i, name in enumerate(item_names)
        },
    }

    # 카테고리별 MAE 집계
    unique_cats = sorted(set(all_cats))
    if len(unique_cats) > 1:
        cat_mae          = {}   # cat → 전체 평균 MAE
        cat_mae_per_item = {}   # cat → {item_name → MAE}

        for cat in unique_cats:
            mask = torch.tensor([c == cat for c in all_cats])
            if mask.sum() == 0:
                continue
            p = pred_cm[mask]
            g = gt_cm[mask]
            diff = (p - g).abs()

            cat_mae[cat] = diff.mean().item()
            cat_mae_per_item[cat] = {
                name: diff[:, i].mean().item()
                for i, name in enumerate(item_names)
            }

        metrics["mae_per_category"]          = cat_mae
        metrics["mae_per_category_per_item"] = cat_mae_per_item

    return metrics


# ──────────────────────────────────────────────────────────────
# 학습 루프
# ──────────────────────────────────────────────────────────────
def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: CombinedLoss,
    device: torch.device,
    epoch: int,
    log_interval: int = 50,
) -> Dict:
    model.train()
    total_loss = seg_loss_sum = meas_loss_sum = 0.0
    n_batches = 0

    for i, batch in enumerate(loader):
        image    = batch["image"].to(device)
        body_vec = batch["body_vec"].to(device)   # [B, 13]
        mask_gt  = batch["mask"].to(device)
        clothes  = batch["clothes"].to(device)

        optimizer.zero_grad()
        seg_pred, meas_pred = model(image, body_vec)
        loss, sl, ml = criterion(seg_pred, mask_gt, meas_pred, clothes)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss    += loss.item()
        seg_loss_sum  += sl.item()
        meas_loss_sum += ml.item()
        n_batches     += 1

        if (i + 1) % log_interval == 0:
            logger.info(
                f"  Epoch {epoch} [{i+1}/{len(loader)}] "
                f"Loss={loss.item():.4f} "
                f"(seg={sl.item():.4f}, meas={ml.item():.4f})"
            )

    return {
        "train_loss":      total_loss    / max(n_batches, 1),
        "train_seg_loss":  seg_loss_sum  / max(n_batches, 1),
        "train_meas_loss": meas_loss_sum / max(n_batches, 1),
    }


# ──────────────────────────────────────────────────────────────
# 메인 학습 함수
# ──────────────────────────────────────────────────────────────
def train(args):
    exp_cfg       = EXPERIMENTS[args.exp]
    use_mediapipe = exp_cfg["mediapipe"] and not args.no_mediapipe
    exp_name      = exp_cfg["name"] + ("_no_norm" if args.no_mediapipe else "")
    cats_tag      = "_".join(sorted(args.categories))
    exp_name      = f"{exp_name}_{cats_tag}"   # ex) exp6_qformer_blouse_shirt

    logger.info(f"\n{'='*55}")
    logger.info(f"실험: {exp_name}")
    logger.info(f"설명: {exp_cfg['desc']}")
    logger.info(f"카테고리: {args.categories}  |  body_dim: {BODY_DIM}")
    logger.info(f"{'='*55}\n")

    ckpt_dir   = os.path.join("checkpoints", exp_name)
    result_dir = os.path.join("results",     exp_name)
    os.makedirs(ckpt_dir,   exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    use_wandb = WANDB_AVAILABLE and args.wandb_project is not None
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run or exp_name,
            config={**vars(args), "exp_desc": exp_cfg["desc"],
                    "body_dim": BODY_DIM, "categories": args.categories},
        )
    elif args.wandb_project:
        logger.warning("wandb 미설치")

    json_dir, image_dir = build_dir_maps(args)
    loaders = get_dataloaders(
        json_dir=json_dir,
        image_dir=image_dir,
        categories=args.categories,
        view_type=args.view_type,
        use_mediapipe=use_mediapipe,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    logger.info(f"Train: {len(loaders['train'].dataset)}  "
                f"Val: {len(loaders['val'].dataset)}  "
                f"Test: {len(loaders['test'].dataset)}")

    # ① body_dim=BODY_DIM(13) 으로 모델 생성
    model = exp_cfg["model_cls"](
        body_dim=BODY_DIM,
        num_measurements=5,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"파라미터 수: {total_params:,}")

    criterion = CombinedLoss(
        lambda1=exp_cfg["lambda1"],
        lambda2=exp_cfg["lambda2"],
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_mae = float("inf")
    history  = []

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n[Epoch {epoch}/{args.epochs}]")

        train_metrics = train_one_epoch(
            model, loaders["train"], optimizer, criterion, device, epoch
        )
        val_metrics = evaluate(model, loaders["val"], criterion, device)
        scheduler.step()

        logger.info(
            f"  Train Loss: {train_metrics['train_loss']:.4f}  "
            f"Val Loss: {val_metrics['total_loss']:.4f}  "
            f"Val MAE: {val_metrics['mae_overall']:.2f}cm"
        )
        logger.info("  Val MAE per item:")
        for name, mae in val_metrics["mae_per_item"].items():
            logger.info(f"    {name}: {mae:.2f}cm")

        # 카테고리별 MAE 로그
        if "mae_per_category_per_item" in val_metrics:
            logger.info("  Val MAE per category:")
            for cat, item_dict in val_metrics["mae_per_category_per_item"].items():
                row = "  ".join(f"{k[:7]}={v:.2f}" for k, v in item_dict.items())
                overall = val_metrics["mae_per_category"][cat]
                logger.info(f"    [{cat}] overall={overall:.2f}cm  {row}")

        row = {
            **train_metrics,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "epoch": epoch,
        }
        history.append(row)

        ckpt_payload = {
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "best_mae":    best_mae,
            "args":        vars(args),
            "body_dim":    BODY_DIM,      # 저장해두면 로드 시 확인 가능
            "categories":  args.categories,
        }

        if val_metrics["mae_overall"] < best_mae:
            best_mae = val_metrics["mae_overall"]
            ckpt_payload["best_mae"] = best_mae
            torch.save(ckpt_payload, os.path.join(ckpt_dir, "best.pth"))
            logger.info(f"Best 모델 저장 (MAE={best_mae:.2f}cm)")

        if epoch % args.ckpt_interval == 0:
            torch.save(ckpt_payload,
                       os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth"))
            logger.info(f"  체크포인트 저장: epoch_{epoch:03d}.pth")

        if use_wandb:
            log_dict = {
                "epoch":           epoch,
                "train/loss":      train_metrics["train_loss"],
                "train/seg_loss":  train_metrics["train_seg_loss"],
                "train/meas_loss": train_metrics["train_meas_loss"],
                "val/loss":        val_metrics["total_loss"],
                "val/mae":         val_metrics["mae_overall"],
                "val/rmse":        val_metrics["rmse_overall"],
                "lr":              scheduler.get_last_lr()[0],
            }
            for name, mae in val_metrics["mae_per_item"].items():
                log_dict[f"val/mae_{name}"] = mae
            if "mae_per_category" in val_metrics:
                for cat, mae in val_metrics["mae_per_category"].items():
                    log_dict[f"val/mae/{cat}"] = mae
            if "mae_per_category_per_item" in val_metrics:
                for cat, item_dict in val_metrics["mae_per_category_per_item"].items():
                    for item_name, mae in item_dict.items():
                        log_dict[f"val/mae/{cat}/{item_name}"] = mae
            wandb.log(log_dict)

    # ── 테스트 평가 ───────────────────────────────────────
    logger.info("\n[Test 평가]")
    ckpt = torch.load(os.path.join(ckpt_dir, "best.pth"), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, loaders["test"], criterion, device)

    logger.info(f"Test MAE:  {test_metrics['mae_overall']:.2f}cm")
    logger.info(f"Test RMSE: {test_metrics['rmse_overall']:.2f}cm")
    logger.info("Test MAE per item:")
    for name, mae in test_metrics["mae_per_item"].items():
        logger.info(f"  {name}: {mae:.2f}cm")
    if "mae_per_category" in test_metrics:
        logger.info("Test MAE per category:")
        for cat, mae in test_metrics["mae_per_category"].items():
            logger.info(f"  {cat}: {mae:.2f}cm")

    results = {
        "exp_name":     exp_name,
        "exp_desc":     exp_cfg["desc"],
        "categories":   args.categories,
        "body_dim":     BODY_DIM,
        "best_val_mae": best_mae,
        "test_metrics": test_metrics,
        "history":      history,
        "args":         vars(args),
    }
    with open(os.path.join(result_dir, "metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"\n결과 저장: {result_dir}/metrics.json")

    if use_wandb:
        test_log = {
            "test/mae":  test_metrics["mae_overall"],
            "test/rmse": test_metrics["rmse_overall"],
        }
        for name, mae in test_metrics["mae_per_item"].items():
            test_log[f"test/mae_{name}"] = mae
        if "mae_per_category" in test_metrics:
            for cat, mae in test_metrics["mae_per_category"].items():
                test_log[f"test/mae/{cat}"] = mae
        if "mae_per_category_per_item" in test_metrics:
            for cat, item_dict in test_metrics["mae_per_category_per_item"].items():
                for item_name, mae in item_dict.items():
                    test_log[f"test/mae/{cat}/{item_name}"] = mae
        wandb.log(test_log)
        wandb.finish()

    return results


# ── Ablation 전체 실행 ────────────────────────────────────────
def run_all_ablations(args):
    all_results = {}
    for exp_id in [1, 2, 3, 4]:
        logger.info(f"\n{'#'*60}")
        logger.info(f"# Ablation Exp {exp_id}: {EXPERIMENTS[exp_id]['desc']}")
        logger.info(f"{'#'*60}")
        args.exp = exp_id
        result   = train(args)
        all_results[exp_id] = result

    print("\n" + "="*65)
    print("Ablation 결과 비교 (Test MAE, cm)")
    print("="*65)
    item_keys = ["shoulder_width", "front_length", "chest_size",
                 "waist_size", "sleeve_length"]
    header = f"{'실험':<35} {'전체MAE':>8}" + "".join(f" {k[:7]:>8}" for k in item_keys)
    print(header)
    print("-"*65)
    for exp_id, result in all_results.items():
        tm  = result["test_metrics"]
        mae = tm["mae_per_item"]
        row = (f"{EXPERIMENTS[exp_id]['desc']:<35} "
               f"{tm['mae_overall']:>8.2f}"
               + "".join(f" {mae.get(k, 0):>8.2f}" for k in item_keys))
        print(row)
    print("="*65)


# ── 인자 파싱 ─────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="의류 치수 추정 모델 학습")

    # ── 경로 인자 (단일 or 복수) ────────────────────────────────
    # 방법 A: 카테고리마다 같은 루트 폴더 (기존 방식)
    #   --json_dir /content/label_blouse  --categories blouse
    #
    # 방법 B: 카테고리마다 다른 폴더 (신규 멀티카테고리)
    #   --json_dirs /content/label_blouse /content/label_shirt
    #   --image_dirs /content/image_blouse /content/image_shirt
    #   --categories blouse shirt
    #   (--json_dirs/--image_dirs의 순서가 --categories 순서와 일치해야 함)
    p.add_argument("--json_dir",      type=str, default=None,
                   help="단일 JSON 디렉토리 (카테고리 1개일 때)")
    p.add_argument("--image_dir",     type=str, default=None,
                   help="단일 이미지 디렉토리 (카테고리 1개일 때)")
    p.add_argument("--json_dirs",     nargs="+", default=None,
                   help="카테고리별 JSON 디렉토리 (--categories 순서와 동일하게)")
    p.add_argument("--image_dirs",    nargs="+", default=None,
                   help="카테고리별 이미지 디렉토리 (--categories 순서와 동일하게)")
    # ────────────────────────────────────────────────────────────

    p.add_argument("--exp",           type=int, default=4,
                   choices=[1, 2, 3, 4, 5, 6])
    p.add_argument("--all",           action="store_true")
    p.add_argument("--no_mediapipe",  action="store_true")
    p.add_argument("--categories",    nargs="+",
                   default=["blouse"],
                   help="학습할 카테고리 (예: blouse shirt coat)")
    p.add_argument("--view_type",     type=str, default="wear",
                   choices=["front", "wear", "all"])
    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--batch_size",    type=int,   default=16)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--val_ratio",     type=float, default=0.1)
    p.add_argument("--test_ratio",    type=float, default=0.1)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--eval_only",     action="store_true")
    p.add_argument("--body_ablation", action="store_true")
    p.add_argument("--ckpt",          type=str,   default=None)
    p.add_argument("--wandb_project", type=str,   default=None)
    p.add_argument("--wandb_run",     type=str,   default=None)
    p.add_argument("--ckpt_interval", type=int,   default=10)
    return p.parse_args()


def build_dir_maps(args):
    """
    args에서 json/image 경로를 Dict[category, path] 형태로 만든다.

    방법 A (--json_dir / --image_dir):
        모든 카테고리가 같은 루트 폴더를 가리키는 str 반환
        → dataset.py의 _resolve_dir()가 str이면 그대로 씀

    방법 B (--json_dirs / --image_dirs):
        카테고리 순서에 맞춰 Dict를 만들어 반환
    """
    cats = args.categories

    if args.json_dirs is not None:
        if len(args.json_dirs) != len(cats):
            raise ValueError(
                f"--json_dirs 개수({len(args.json_dirs)})와 "
                f"--categories 개수({len(cats)})가 다릅니다."
            )
        json_dir = dict(zip(cats, args.json_dirs))
    elif args.json_dir is not None:
        json_dir = args.json_dir   # str — 모든 카테고리 공유
    else:
        raise ValueError("--json_dir 또는 --json_dirs 중 하나는 반드시 지정해야 합니다.")

    if args.image_dirs is not None:
        if len(args.image_dirs) != len(cats):
            raise ValueError(
                f"--image_dirs 개수({len(args.image_dirs)})와 "
                f"--categories 개수({len(cats)})가 다릅니다."
            )
        image_dir = dict(zip(cats, args.image_dirs))
    elif args.image_dir is not None:
        image_dir = args.image_dir
    else:
        raise ValueError("--image_dir 또는 --image_dirs 중 하나는 반드시 지정해야 합니다.")

    return json_dir, image_dir


# ── body feature ablation ──────────────────────────────────────
BODY_FEATURE_NAMES = [
    "body_height", "breast_size_female", "waist_size", "hip_seize",
    "shoulders_width", "arm_length", "waist_height", "back_length",
    "weight", "gender",
    # 카테고리 one-hot 3차원도 ablation 가능
    "cat_blouse", "cat_shirt", "cat_coat",
]


def evaluate_masked(model, loader, criterion, device, mask_idx: int):
    model.eval()
    all_pred, all_gt, all_cats = [], [], []
    with torch.no_grad():
        for batch in loader:
            image    = batch["image"].to(device)
            body_vec = batch["body_vec"].to(device).clone()
            body_vec[:, mask_idx] = 0.0
            mask_gt  = batch["mask"].to(device)
            clothes  = batch["clothes"].to(device)
            cats     = batch["category"]

            _, meas_pred = model(image, body_vec)
            all_pred.append(meas_pred.cpu())
            all_gt.append(clothes.cpu())
            all_cats.extend(cats)

    all_pred = torch.cat(all_pred, dim=0)
    all_gt   = torch.cat(all_gt,   dim=0)
    pred_cm  = denormalize_clothes_batch(all_pred, all_cats)
    gt_cm    = denormalize_clothes_batch(all_gt,   all_cats)
    mae_per  = (pred_cm - gt_cm).abs().mean(dim=0)
    return mae_per.mean().item(), mae_per.tolist()


def body_ablation(args):
    exp_cfg   = EXPERIMENTS[args.exp]
    cats_tag  = "_".join(sorted(args.categories))
    exp_name  = exp_cfg["name"] + ("_no_norm" if args.no_mediapipe else "") + f"_{cats_tag}"
    ckpt_path = args.ckpt or os.path.join("checkpoints", exp_name, "best.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    json_dir, image_dir = build_dir_maps(args)
    loaders = get_dataloaders(
        json_dir=json_dir,
        image_dir=image_dir,
        categories=args.categories,
        view_type=args.view_type,
        use_mediapipe=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    model = exp_cfg["model_cls"](body_dim=BODY_DIM, num_measurements=5).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    criterion = CombinedLoss(lambda1=exp_cfg["lambda1"],
                             lambda2=exp_cfg["lambda2"])

    base = evaluate(model, loaders["test"], criterion, device)
    base_mae  = base["mae_overall"]
    base_per  = list(base["mae_per_item"].values())
    item_names = list(base["mae_per_item"].keys())

    print(f"\n{'='*65}")
    print(f"신체치수 피처 Ablation  (exp={exp_name}, body_dim={BODY_DIM})")
    print(f"{'='*65}")
    print(f"  {'[baseline]':<22} {base_mae:>8.2f}  " +
          "  ".join(f"{v:>7.2f}" for v in base_per))
    print("-"*65)

    results = []
    for i, feat_name in enumerate(BODY_FEATURE_NAMES):
        mae, per = evaluate_masked(model, loaders["test"], criterion, device, i)
        delta = mae - base_mae
        results.append((delta, feat_name, mae, per))
        print(f"  {feat_name:<22} {mae:>8.2f} {delta:>+7.2f}  " +
              "  ".join(f"{v:>7.2f}" for v in per))

    print(f"{'='*65}")
    results.sort(reverse=True)
    print("\n중요도 순위:")
    for rank, (delta, name, mae, _) in enumerate(results, 1):
        bar = "█" * max(0, int(delta * 5))
        print(f"  {rank}. {name:<24} Δ={delta:+.2f}cm  {bar}")


def eval_only(args):
    exp_cfg   = EXPERIMENTS[args.exp]
    cats_tag  = "_".join(sorted(args.categories))
    exp_name  = exp_cfg["name"] + ("_no_norm" if args.no_mediapipe else "") + f"_{cats_tag}"
    ckpt_path = args.ckpt or os.path.join("checkpoints", exp_name, "best.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"체크포인트: {ckpt_path}  /  Device: {device}")

    json_dir, image_dir = build_dir_maps(args)
    loaders = get_dataloaders(
        json_dir=json_dir,
        image_dir=image_dir,
        categories=args.categories,
        view_type=args.view_type,
        use_mediapipe=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    model = exp_cfg["model_cls"](body_dim=BODY_DIM, num_measurements=5).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    logger.info(f"epoch {ckpt['epoch']} 체크포인트 로드  "
                f"(저장 시 categories={ckpt.get('categories', '?')})")

    criterion = CombinedLoss(lambda1=exp_cfg["lambda1"],
                             lambda2=exp_cfg["lambda2"])
    for split in ["val", "test"]:
        m = evaluate(model, loaders[split], criterion, device)
        logger.info(f"\n[{split.upper()}]  MAE={m['mae_overall']:.2f}cm  "
                    f"RMSE={m['rmse_overall']:.2f}cm")
        for name, mae in m["mae_per_item"].items():
            logger.info(f"  {name}: {mae:.2f}cm")
        if "mae_per_category" in m:
            for cat, mae in m["mae_per_category"].items():
                logger.info(f"  [{cat}]: {mae:.2f}cm")


if __name__ == "__main__":
    args = parse_args()
    if args.body_ablation:
        body_ablation(args)
    elif args.eval_only:
        eval_only(args)
    elif args.all:
        run_all_ablations(args)
    else:
        train(args)