import os
import json
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict
import logging

from dataset import (
    get_dataloaders, denormalize_clothes, CLOTHES_KEYS
)
from model import (
    ClothingMeasurementNet, BaselineMLP, NoFiLMModel,
    SegOnlyModel, CombinedLoss,
    get_optimizer_phase1, get_optimizer_phase2,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Ablation 실험 설정
# ────────────────────────────────────────────────────────────────────
EXPERIMENTS = {
    1: {
        "name":      "exp1_mlp_baseline",
        "desc":      "수치 단독 MLP (이미지 없음)",
        "model_cls": BaselineMLP,
        "mediapipe": False,
        "lambda1":   0.0,
        "lambda2":   1.0,
    },
    2: {
        "name":      "exp2_seg_no_norm",
        "desc":      "세그멘테이션 + 회귀 (MediaPipe 정규화 없음)",
        "model_cls": SegOnlyModel,
        "mediapipe": False,
        "lambda1":   1.0,
        "lambda2":   0.1,
    },
    3: {
        "name":      "exp3_no_film",
        "desc":      "세그멘테이션 + 회귀 (FiLM 없음)",
        "model_cls": NoFiLMModel,
        "mediapipe": True,
        "lambda1":   1.0,
        "lambda2":   0.1,
    },
    4: {
        "name":      "exp4_full",
        "desc":      "Full model (MediaPipe + FiLM + 세그멘테이션 + 회귀)",
        "model_cls": ClothingMeasurementNet,
        "mediapipe": True,
        "lambda1":   1.0,
        "lambda2":   0.5,
    },
}

PHASE1_EPOCHS = 10


# ────────────────────────────────────────────────────────────────────
# 평가 함수
# ────────────────────────────────────────────────────────────────────
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
    rmse_per_item = ((pred_cm - gt_cm) ** 2).mean(dim=0).sqrt()

    item_names = [k.split(".")[-1] for k in CLOTHES_KEYS]

    return {
        "total_loss":   total_loss    / max(n_batches, 1),
        "seg_loss":     seg_loss_sum  / max(n_batches, 1),
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


# ────────────────────────────────────────────────────────────────────
# 학습 루프 (1 에폭)
# ────────────────────────────────────────────────────────────────────
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


# ────────────────────────────────────────────────────────────────────
# 헬퍼
# ────────────────────────────────────────────────────────────────────
def _log_epoch(epoch: int, train_metrics: Dict, val_metrics: Dict):
    logger.info(
        f"  Train Loss: {train_metrics['train_loss']:.4f}  "
        f"Val Loss: {val_metrics['total_loss']:.4f}  "
        f"Val MAE: {val_metrics['mae_overall']:.2f}cm"
    )
    logger.info("  Val MAE per item:")
    for name, mae in val_metrics["mae_per_item"].items():
        logger.info(f"    {name}: {mae:.2f}cm")


def _maybe_save_best(model, optimizer, epoch, val_metrics, best_mae, ckpt_dir, args):
    if val_metrics["mae_overall"] < best_mae:
        best_mae = val_metrics["mae_overall"]
        torch.save({
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "best_mae":    best_mae,
            "args":        vars(args),
        }, os.path.join(ckpt_dir, "best.pth"))
        logger.info(f"  ★ Best 모델 저장 (MAE={best_mae:.2f}cm)")
    return best_mae


# ────────────────────────────────────────────────────────────────────
# 메인 학습 함수
# ────────────────────────────────────────────────────────────────────
def train(args):
    exp_cfg       = EXPERIMENTS[args.exp]
    use_mediapipe = exp_cfg["mediapipe"] and not args.no_mediapipe
    exp_name      = exp_cfg["name"] + ("_no_norm" if args.no_mediapipe else "")
    is_full_model = (exp_cfg["model_cls"] is ClothingMeasurementNet)

    logger.info(f"\n{'='*50}")
    logger.info(f"실험: {exp_name}")
    logger.info(f"설명: {exp_cfg['desc']}")
    logger.info(f"MediaPipe: {use_mediapipe}")
    logger.info(f"cache_dir: {args.cache_dir}")
    logger.info(f"Phase 학습: {is_full_model}")
    logger.info(f"{'='*50}\n")

    ckpt_dir   = os.path.join("checkpoints", exp_name)
    result_dir = os.path.join("results",     exp_name)
    os.makedirs(ckpt_dir,   exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    loaders = get_dataloaders(
        json_dir=args.json_dir,
        image_dir=args.image_dir,
        cache_dir=args.cache_dir,        
        categories=args.categories,
        view_type=args.view_type,
        use_mediapipe=use_mediapipe,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    logger.info(
        f"Train: {len(loaders['train'].dataset)}  "
        f"Val: {len(loaders['val'].dataset)}  "
        f"Test: {len(loaders['test'].dataset)}"
    )

    if is_full_model:
        model = exp_cfg["model_cls"](
            body_dim=10,
            num_measurements=5,
            pretrained=True,
        ).to(device)
    else:
        model = exp_cfg["model_cls"](
            body_dim=10,
            num_measurements=5,
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"파라미터 수: {total_params:,}")

    criterion = CombinedLoss(
        lambda1=exp_cfg["lambda1"],
        lambda2=exp_cfg["lambda2"],
    )

    best_mae = float("inf")
    history  = []

    # ── Phase 학습 (ClothingMeasurementNet 전용) ──────────────────────
    if is_full_model:
        phase2_epochs = args.epochs

        logger.info(f"\n=== Phase 1: Encoder freeze ({PHASE1_EPOCHS} epochs) ===")
        optimizer = get_optimizer_phase1(model, lr_decoder=1e-3)
        scheduler = CosineAnnealingLR(optimizer, T_max=PHASE1_EPOCHS, eta_min=1e-5)

        for epoch in range(1, PHASE1_EPOCHS + 1):
            logger.info(f"\n[Phase1 Epoch {epoch}/{PHASE1_EPOCHS}]")
            train_metrics = train_one_epoch(
                model, loaders["train"], optimizer, criterion, device, epoch
            )
            val_metrics = evaluate(model, loaders["val"], criterion, device)
            scheduler.step()
            _log_epoch(epoch, train_metrics, val_metrics)
            history.append({
                **train_metrics,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "epoch": epoch, "phase": 1,
            })
            best_mae = _maybe_save_best(
                model, optimizer, epoch, val_metrics, best_mae, ckpt_dir, args
            )

        logger.info(f"\n=== Phase 2: 전체 unfreeze ({phase2_epochs} epochs) ===")
        optimizer = get_optimizer_phase2(model, lr_encoder=1e-4, lr_others=1e-3)
        scheduler = CosineAnnealingLR(optimizer, T_max=phase2_epochs, eta_min=1e-6)

        for epoch in range(PHASE1_EPOCHS + 1, PHASE1_EPOCHS + phase2_epochs + 1):
            logger.info(f"\n[Phase2 Epoch {epoch}/{PHASE1_EPOCHS + phase2_epochs}]")
            train_metrics = train_one_epoch(
                model, loaders["train"], optimizer, criterion, device, epoch
            )
            val_metrics = evaluate(model, loaders["val"], criterion, device)
            scheduler.step()
            _log_epoch(epoch, train_metrics, val_metrics)
            history.append({
                **train_metrics,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "epoch": epoch, "phase": 2,
            })
            best_mae = _maybe_save_best(
                model, optimizer, epoch, val_metrics, best_mae, ckpt_dir, args
            )

    # ── Ablation 단일 루프 ────────────────────────────────────────────
    else:
        optimizer = optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

        for epoch in range(1, args.epochs + 1):
            logger.info(f"\n[Epoch {epoch}/{args.epochs}]")
            train_metrics = train_one_epoch(
                model, loaders["train"], optimizer, criterion, device, epoch
            )
            val_metrics = evaluate(model, loaders["val"], criterion, device)
            scheduler.step()
            _log_epoch(epoch, train_metrics, val_metrics)
            history.append({
                **train_metrics,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "epoch": epoch,
            })
            best_mae = _maybe_save_best(
                model, optimizer, epoch, val_metrics, best_mae, ckpt_dir, args
            )

    # ── 테스트 평가 ───────────────────────────────────────────────────
    logger.info("\n[Test 평가]")
    ckpt = torch.load(os.path.join(ckpt_dir, "best.pth"), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, loaders["test"], criterion, device)

    logger.info(f"Test MAE:  {test_metrics['mae_overall']:.2f}cm")
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

    return results


# ────────────────────────────────────────────────────────────────────
# Ablation 전체 실행
# ────────────────────────────────────────────────────────────────────
def run_all_ablations(args):
    all_results = {}
    for exp_id in [1, 2, 3, 4]:
        logger.info(f"\n{'#'*60}")
        logger.info(f"# Ablation Exp {exp_id}: {EXPERIMENTS[exp_id]['desc']}")
        logger.info(f"{'#'*60}")
        args.exp = exp_id
        result = train(args)
        all_results[exp_id] = result

    print("\n" + "="*60)
    print("Ablation 결과 비교 (Test MAE, cm)")
    print("="*60)
    header = (
        f"{'실험':<35} {'전체MAE':>8} "
        f"{'어깨':>8} {'총장':>8} {'가슴':>8} {'소매':>8}"
    )
    print(header)
    print("-"*60)

    item_keys = ["shoulder_width", "front_length", "chest_size", "sleeve_length"]
    for exp_id, result in all_results.items():
        tm = result["test_metrics"]
        mae_items = tm["mae_per_item"]
        row = (
            f"{EXPERIMENTS[exp_id]['desc']:<35} "
            f"{tm['mae_overall']:>8.2f} "
            + " ".join(f"{mae_items.get(k, 0):>8.2f}" for k in item_keys)
        )
        print(row)
    print("="*60)


# ────────────────────────────────────────────────────────────────────
# argparse
# ────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="의류 치수 추정 모델 학습")
    p.add_argument("--json_dir",     type=str, required=True)
    p.add_argument("--image_dir",    type=str, required=True)
    p.add_argument("--cache_dir",    type=str, default=None,      # ← 추가
                   help="preprocess_cache.py로 생성한 .npy 캐시 디렉토리 "
                        "(지정 시 MediaPipe를 학습 중 실행하지 않음)")
    p.add_argument("--exp",          type=int,   default=4, choices=[1, 2, 3, 4])
    p.add_argument("--all",          action="store_true")
    p.add_argument("--no_mediapipe", action="store_true")
    p.add_argument("--categories",   nargs="+", default=["blouse", "coat", "shirt"])
    p.add_argument("--view_type",    type=str,  default="front",
                   choices=["front", "wear", "all"])
    p.add_argument("--epochs",       type=int,  default=50)
    p.add_argument("--batch_size",   type=int,  default=16)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers",  type=int,  default=4)
    p.add_argument("--val_ratio",    type=float, default=0.1)
    p.add_argument("--test_ratio",   type=float, default=0.1)
    p.add_argument("--seed",         type=int,  default=42)
    p.add_argument("--eval_only",    action="store_true")
    p.add_argument("--body_ablation", action="store_true")
    p.add_argument("--ckpt",         type=str,  default=None)
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────
# 신체치수 피처 중요도 ablation
# ────────────────────────────────────────────────────────────────────
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
    exp_cfg   = EXPERIMENTS[args.exp]
    exp_name  = exp_cfg["name"] + ("_no_norm" if args.no_mediapipe else "")
    ckpt_path = args.ckpt or os.path.join("checkpoints", exp_name, "best.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = get_dataloaders(
        json_dir=args.json_dir,
        image_dir=args.image_dir,
        cache_dir=args.cache_dir,
        categories=args.categories,
        view_type=args.view_type,
        use_mediapipe=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    is_full = (exp_cfg["model_cls"] is ClothingMeasurementNet)
    if is_full:
        model = exp_cfg["model_cls"](body_dim=10, num_measurements=5,
                                     pretrained=False).to(device)
    else:
        model = exp_cfg["model_cls"](body_dim=10, num_measurements=5).to(device)

    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    criterion = CombinedLoss(lambda1=exp_cfg["lambda1"], lambda2=exp_cfg["lambda2"])

    base          = evaluate(model, loaders["test"], criterion, device)
    base_mae      = base["mae_overall"]
    base_per      = list(base["mae_per_item"].values())
    clothes_names = list(base["mae_per_item"].keys())

    print(f"\n{'='*65}")
    print(f"신체치수 피처 Ablation  (exp={exp_name}, test set)")
    print(f"{'='*65}")
    header = (
        f"{'피처':<22} {'전체MAE':>8} {'Δ':>7}  "
        + "  ".join(f"{n[:6]:>7}" for n in clothes_names)
    )
    print(header)
    print(f"  {'[baseline]':<20} {base_mae:>8.2f}{'':>8}  "
          + "  ".join(f"{v:>7.2f}" for v in base_per))
    print("-"*65)

    results = []
    for i, feat_name in enumerate(BODY_FEATURE_NAMES):
        mae, per = evaluate_masked(model, loaders["test"], criterion, device, mask_idx=i)
        delta = mae - base_mae
        results.append((delta, feat_name, mae, per))
        print(f"  {feat_name:<20} {mae:>8.2f} {delta:>+7.2f}  "
              + "  ".join(f"{v:>7.2f}" for v in per))

    print(f"{'='*65}")
    results.sort(reverse=True)
    print("\n중요도 순위 (델타 클수록 해당 피처 의존도 높음):")
    for rank, (delta, name, mae, _) in enumerate(results, 1):
        bar = "█" * max(0, int(delta * 5))
        print(f"  {rank}. {name:<22} Δ={delta:+.2f}cm  {bar}")


def eval_only(args):
    exp_cfg   = EXPERIMENTS[args.exp]
    exp_name  = exp_cfg["name"] + ("_no_norm" if args.no_mediapipe else "")
    ckpt_path = args.ckpt or os.path.join("checkpoints", exp_name, "best.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"체크포인트: {ckpt_path}  /  Device: {device}")

    loaders = get_dataloaders(
        json_dir=args.json_dir,
        image_dir=args.image_dir,
        cache_dir=args.cache_dir,
        categories=args.categories,
        view_type=args.view_type,
        use_mediapipe=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    is_full = (exp_cfg["model_cls"] is ClothingMeasurementNet)
    if is_full:
        model = exp_cfg["model_cls"](body_dim=10, num_measurements=5,
                                     pretrained=False).to(device)
    else:
        model = exp_cfg["model_cls"](body_dim=10, num_measurements=5).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    logger.info(f"epoch {ckpt['epoch']} 체크포인트 로드")

    criterion = CombinedLoss(lambda1=exp_cfg["lambda1"], lambda2=exp_cfg["lambda2"])

    for split in ["val", "test"]:
        m = evaluate(model, loaders[split], criterion, device)
        logger.info(
            f"\n[{split.upper()}]  "
            f"MAE={m['mae_overall']:.2f}cm  "
            f"RMSE={m['rmse_overall']:.2f}cm"
        )
        for name, mae in m["mae_per_item"].items():
            logger.info(f"  {name}: {mae:.2f}cm")


# ────────────────────────────────────────────────────────────────────
# 진입점
# ────────────────────────────────────────────────────────────────────
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