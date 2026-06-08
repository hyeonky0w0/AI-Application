import os
import json
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Tuple
import logging

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from dataset import (
    get_dataloaders, denormalize_clothes, CLOTHES_KEYS
)
from model import (
    ClothingMeasurementNet, ClothingMeasurementNetCA,
    ClothingMeasurementNetQFormer,
    ImageOnlyQFormerModel, ImageBodyNoFiLMQFormerModel,
    BaselineMLP, NoFiLMModel, CombinedLoss
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# Ablation 실험 설정
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
        "name":      "exp2_image_only_qformer",
        "desc":      "Exp2: 이미지만 (body 없음, FiLM 없음, Q-Former)",
        "model_cls": ImageOnlyQFormerModel,
        "mediapipe": False,
        "lambda1":   1.0,
        "lambda2":   0.1,
    },
    3: {
        "name":      "exp3_image_body_no_film_qformer",
        "desc":      "Exp3: 이미지 + body (FiLM 없음, Q-Former)",
        "model_cls": ImageBodyNoFiLMQFormerModel,
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
        "desc":      "Exp5: Cross-Attention 조건화 (FiLM → BodyCrossAttention)",
        "model_cls": ClothingMeasurementNetCA,
        "mediapipe": False,
        "lambda1":   1.0,
        "lambda2":   0.1,
    },
    6: {
        "name":      "exp6_qformer",
        "desc":      "Exp6: Q-Former 헤드 (치수별 전담 쿼리 토큰)",
        "model_cls": ClothingMeasurementNetQFormer,
        "mediapipe": False,
        "lambda1":   1.0,
        "lambda2":   0.5,
    },
}


def _build_model(args, exp_cfg) -> torch.nn.Module:
    """args와 exp_cfg를 보고 올바른 model_kwargs로 모델을 생성해 반환."""
    model_kwargs = dict(body_dim=10, num_measurements=5)
    if args.exp == 3 and getattr(args, "no_backbone", False):
        model_kwargs["use_backbone"] = False
        model_kwargs["pretrained"]   = False
    return exp_cfg["model_cls"](**model_kwargs)


# 평가 함수
def evaluate(
    model: torch.nn.Module,
    loader,
    criterion: CombinedLoss,
    device: torch.device,
) -> Dict:
    model.eval()
    total_loss = seg_loss_sum = meas_loss_sum = 0.0
    all_pred = []
    all_gt   = []
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            image    = batch["image"].to(device)
            body_vec = batch["body_vec"].to(device)
            mask_gt  = batch["mask"].to(device)
            clothes  = batch["clothes"].to(device)

            seg_pred, meas_pred = model(image, body_vec)
            loss, sl, ml = criterion(seg_pred, mask_gt, meas_pred, clothes)

            total_loss    += loss.item()
            seg_loss_sum  += sl.item()
            meas_loss_sum += ml.item()
            n_batches     += 1

            all_pred.append(meas_pred.cpu())
            all_gt.append(clothes.cpu())

    all_pred = torch.cat(all_pred, dim=0)
    all_gt   = torch.cat(all_gt,   dim=0)

    pred_cm = denormalize_clothes(all_pred)
    gt_cm   = denormalize_clothes(all_gt)

    mae_per_item  = (pred_cm - gt_cm).abs().mean(dim=0)
    rmse_per_item = ((pred_cm - gt_cm)**2).mean(dim=0).sqrt()

    item_names = [k.split(".")[-1] for k in CLOTHES_KEYS]

    metrics = {
        "total_loss":   total_loss  / max(n_batches, 1),
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
    return metrics


# 학습 루프
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
        body_vec = batch["body_vec"].to(device)
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


# 메인 학습 함수
def train(args):
    exp_cfg = EXPERIMENTS[args.exp]
    use_mediapipe = exp_cfg["mediapipe"] and not args.no_mediapipe

    # no_backbone 옵션이 켜져 있으면 체크포인트 디렉토리 이름에 반영
    suffix = ""
    if args.no_mediapipe:
        suffix += "_no_norm"
    if args.exp == 3 and getattr(args, "no_backbone", False):
        suffix += "_no_backbone"
    exp_name = exp_cfg["name"] + suffix

    logger.info(f"\n{'='*50}")
    logger.info(f"실험: {exp_name}")
    logger.info(f"설명: {exp_cfg['desc']}")
    logger.info(f"MediaPipe: {use_mediapipe}")
    if args.exp == 3:
        logger.info(f"Backbone: {not getattr(args, 'no_backbone', False)}")
    logger.info(f"{'='*50}\n")

    ckpt_dir   = os.path.join("checkpoints", exp_name)
    result_dir = os.path.join("results", exp_name)
    os.makedirs(ckpt_dir,   exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # wandb 초기화
    use_wandb = WANDB_AVAILABLE and args.wandb_project is not None
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run or exp_name,
            config={**vars(args), "exp_desc": exp_cfg["desc"]},
        )
    elif args.wandb_project:
        logger.warning("wandb 미설치")

    # DataLoader
    loaders = get_dataloaders(
        json_dir=args.json_dir,
        image_dir=args.image_dir,
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

    model = _build_model(args, exp_cfg).to(device)
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

        row = {**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}, "epoch": epoch}
        history.append(row)

        ckpt_payload = {
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "best_mae":    best_mae,
            "args":        vars(args),
        }

        if val_metrics["mae_overall"] < best_mae:
            best_mae = val_metrics["mae_overall"]
            ckpt_payload["best_mae"] = best_mae
            torch.save(ckpt_payload, os.path.join(ckpt_dir, "best.pth"))
            logger.info(f"Best 모델 저장 (MAE={best_mae:.2f}cm)")

        if epoch % args.ckpt_interval == 0:
            torch.save(ckpt_payload, os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth"))
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
            wandb.log(log_dict)

    # 테스트 평가
    logger.info("\n[Test 평가]")
    ckpt = torch.load(os.path.join(ckpt_dir, "best.pth"), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, loaders["test"], criterion, device)

    logger.info(f"Test MAE: {test_metrics['mae_overall']:.2f}cm")
    logger.info(f"Test RMSE: {test_metrics['rmse_overall']:.2f}cm")
    logger.info("Test MAE per item:")
    for name, mae in test_metrics["mae_per_item"].items():
        logger.info(f"  {name}: {mae:.2f}cm")

    results = {
        "exp_name":     exp_name,
        "exp_desc":     exp_cfg["desc"],
        "best_val_mae": best_mae,
        "test_metrics": test_metrics,
        "history":      history,
        "args":         vars(args),
    }
    with open(os.path.join(result_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"\n결과 저장: {result_dir}/metrics.json")

    if use_wandb:
        test_log = {
            "test/mae":  test_metrics["mae_overall"],
            "test/rmse": test_metrics["rmse_overall"],
        }
        for name, mae in test_metrics["mae_per_item"].items():
            test_log[f"test/mae_{name}"] = mae
        wandb.log(test_log)
        wandb.finish()

    return results


# Ablation 전체 실행 함수
def run_all_ablations(args):
    ablation_ids = [1, 2, 3, 4, 6]
    all_results = {}
    for exp_id in ablation_ids:
        logger.info(f"\n{'#'*60}")
        logger.info(f"# Ablation Exp {exp_id}: {EXPERIMENTS[exp_id]['desc']}")
        logger.info(f"{'#'*60}")
        args.exp = exp_id
        result = train(args)
        all_results[exp_id] = result

    item_keys   = ["shoulder_width", "front_length", "chest_size", "waist_size", "sleeve_length"]
    item_labels = ["어깨", "총장", "가슴", "허리", "소매"]
    col_w = 8

    print("\n" + "="*80)
    print("Ablation 결과 비교 (Test MAE, cm)")
    print("="*80)
    header = f"{'실험':<40} {'전체':>{col_w}}" + "".join(f"{lb:>{col_w}}" for lb in item_labels)
    print(header)
    print("-"*80)

    for exp_id, result in all_results.items():
        tm = result["test_metrics"]
        mae_items = tm["mae_per_item"]
        desc = EXPERIMENTS[exp_id]["desc"][:38]
        row = (
            f"{desc:<40} {tm['mae_overall']:>{col_w}.2f}"
            + "".join(f"{mae_items.get(k, 0):>{col_w}.2f}" for k in item_keys)
        )
        print(row)
    print("="*80)


def parse_args():
    p = argparse.ArgumentParser(description="의류 치수 추정 모델 학습")
    p.add_argument("--json_dir",    type=str, required=True)
    p.add_argument("--image_dir",   type=str, nargs="+", required=True)
    p.add_argument("--exp",         type=int,   default=4, choices=[1,2,3,4,5,6])
    p.add_argument("--all",         action="store_true")
    p.add_argument("--no_mediapipe",action="store_true")
    p.add_argument("--no_backbone", action="store_true",
                   help="백본(ResNet50) 대신 LightEncoder 사용 (exp3 전용)")
    p.add_argument("--categories",  nargs="+",  default=["blouse", "coat", "shirt"])
    p.add_argument("--view_type",   type=str,   default="front",
                   choices=["front", "wear", "all"])
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--val_ratio",   type=float, default=0.1)
    p.add_argument("--test_ratio",  type=float, default=0.1)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--eval_only",   action="store_true")
    p.add_argument("--body_ablation", action="store_true")
    p.add_argument("--ckpt",        type=str,   default=None)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_run",   type=str,   default=None)
    p.add_argument("--ckpt_interval", type=int, default=10)
    return p.parse_args()


BODY_FEATURE_NAMES = [
    "body_height", "breast_size_female", "waist_size", "hip_seize",
    "shoulders_width", "arm_length", "waist_height", "back_length",
    "weight", "gender",
]


def evaluate_masked(model, loader, criterion, device, mask_idx: int):
    model.eval()
    all_pred, all_gt = [], []
    with torch.no_grad():
        for batch in loader:
            image    = batch["image"].to(device)
            body_vec = batch["body_vec"].to(device).clone()
            body_vec[:, mask_idx] = 0.0
            mask_gt  = batch["mask"].to(device)
            clothes  = batch["clothes"].to(device)

            _, meas_pred = model(image, body_vec)
            all_pred.append(meas_pred.cpu())
            all_gt.append(clothes.cpu())

    all_pred = torch.cat(all_pred, dim=0)
    all_gt   = torch.cat(all_gt,   dim=0)
    pred_cm  = denormalize_clothes(all_pred)
    gt_cm    = denormalize_clothes(all_gt)
    mae_per  = (pred_cm - gt_cm).abs().mean(dim=0)
    return mae_per.mean().item(), mae_per.tolist()


def body_ablation(args):
    exp_cfg  = EXPERIMENTS[args.exp]
    suffix   = ("_no_norm" if args.no_mediapipe else "") + \
               ("_no_backbone" if args.exp == 3 and getattr(args, "no_backbone", False) else "")
    exp_name = exp_cfg["name"] + suffix
    ckpt_path = args.ckpt or os.path.join("checkpoints", exp_name, "best.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = get_dataloaders(
        json_dir=args.json_dir,
        image_dir=args.image_dir,
        categories=args.categories,
        view_type=args.view_type,
        use_mediapipe=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    model = _build_model(args, exp_cfg).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    criterion = CombinedLoss(lambda1=exp_cfg["lambda1"], lambda2=exp_cfg["lambda2"])

    base = evaluate(model, loaders["test"], criterion, device)
    base_mae      = base["mae_overall"]
    base_per      = list(base["mae_per_item"].values())
    clothes_names = list(base["mae_per_item"].keys())

    print(f"\n{'='*65}")
    print(f"신체치수 피처 Ablation  (exp={exp_name}, test set)")
    print(f"{'='*65}")
    header = f"{'피처':<22} {'전체MAE':>8} {'Δ':>7}  " + "  ".join(f"{n[:6]:>7}" for n in clothes_names)
    print(header)
    print(f"  {'[baseline]':<20} {base_mae:>8.2f}{'':>8}  " +
          "  ".join(f"{v:>7.2f}" for v in base_per))
    print("-"*65)

    results = []
    for i, feat_name in enumerate(BODY_FEATURE_NAMES):
        mae, per = evaluate_masked(model, loaders["test"], criterion, device, mask_idx=i)
        delta = mae - base_mae
        results.append((delta, feat_name, mae, per))
        print(f"  {feat_name:<20} {mae:>8.2f} {delta:>+7.2f}  " +
              "  ".join(f"{v:>7.2f}" for v in per))

    print(f"{'='*65}")
    results.sort(reverse=True)
    print("\n중요도 순위 (델타 클수록 해당 피처 의존도 높음):")
    for rank, (delta, name, mae, _) in enumerate(results, 1):
        bar = "█" * max(0, int(delta * 5))
        print(f"  {rank}. {name:<22} Δ={delta:+.2f}cm  {bar}")


def eval_only(args):
    exp_cfg  = EXPERIMENTS[args.exp]
    suffix   = ("_no_norm" if args.no_mediapipe else "") + \
               ("_no_backbone" if args.exp == 3 and getattr(args, "no_backbone", False) else "")
    exp_name = exp_cfg["name"] + suffix
    ckpt_path = args.ckpt or os.path.join("checkpoints", exp_name, "best.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"체크포인트: {ckpt_path}  /  Device: {device}")

    loaders = get_dataloaders(
        json_dir=args.json_dir,
        image_dir=args.image_dir,
        categories=args.categories,
        view_type=args.view_type,
        use_mediapipe=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    model = _build_model(args, exp_cfg).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    logger.info(f"epoch {ckpt['epoch']} 체크포인트 로드")

    criterion = CombinedLoss(lambda1=exp_cfg["lambda1"], lambda2=exp_cfg["lambda2"])

    for split in ["val", "test"]:
        m = evaluate(model, loaders[split], criterion, device)
        logger.info(f"\n[{split.upper()}]  MAE={m['mae_overall']:.2f}cm  RMSE={m['rmse_overall']:.2f}cm")
        for name, mae in m["mae_per_item"].items():
            logger.info(f"  {name}: {mae:.2f}cm")


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