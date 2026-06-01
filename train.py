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

from dataset_orig_multi import (
    get_dataloaders, denormalize_clothes, CLOTHES_KEYS
)
from model import (
    ClothingMeasurementNet, ClothingMeasurementNetCA,
    ClothingMeasurementNetQFormer,
    BaselineMLP, NoFiLMModel, CombinedLoss
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

EXPERIMENTS = {
    1: {"name": "exp1_mlp_baseline",      "desc": "Exp1: MLP Baseline",
        "model_cls": BaselineMLP,               "lambda1": 0.0, "lambda2": 1.0},
    2: {"name": "exp2_image_only",        "desc": "Exp2: 이미지만 (FiLM 없음)",
        "model_cls": NoFiLMModel,               "lambda1": 1.0, "lambda2": 0.1},
    3: {"name": "exp3_image_body_concat", "desc": "Exp3: 이미지+body concat",
        "model_cls": NoFiLMModel,               "lambda1": 1.0, "lambda2": 0.1},
    4: {"name": "exp4_backbone",          "desc": "Exp4: Full (FiLM)",
        "model_cls": ClothingMeasurementNet,    "lambda1": 1.0, "lambda2": 0.1},
    5: {"name": "exp5_cross_attn",        "desc": "Exp5: CrossAttention",
        "model_cls": ClothingMeasurementNetCA,  "lambda1": 1.0, "lambda2": 0.1},
    6: {"name": "exp6_qformer",           "desc": "Exp6: Q-Former",
        "model_cls": ClothingMeasurementNetQFormer, "lambda1": 1.0, "lambda2": 0.1},
}


def evaluate(model, loader, criterion, device) -> Dict:
    model.eval()
    total_loss = seg_loss_sum = meas_loss_sum = 0.0
    all_pred, all_gt = [], []
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
    pred_cm  = denormalize_clothes(all_pred)
    gt_cm    = denormalize_clothes(all_gt)

    mae_per_item  = (pred_cm - gt_cm).abs().mean(dim=0)
    rmse_per_item = ((pred_cm - gt_cm)**2).mean(dim=0).sqrt()
    item_names    = [k.split(".")[-1] for k in CLOTHES_KEYS]

    return {
        "total_loss":    total_loss    / max(n_batches, 1),
        "seg_loss":      seg_loss_sum  / max(n_batches, 1),
        "meas_loss":     meas_loss_sum / max(n_batches, 1),
        "mae_overall":   mae_per_item.mean().item(),
        "rmse_overall":  rmse_per_item.mean().item(),
        "mae_per_item":  {n: mae_per_item[i].item()  for i, n in enumerate(item_names)},
        "rmse_per_item": {n: rmse_per_item[i].item() for i, n in enumerate(item_names)},
    }


def train_one_epoch(model, loader, optimizer, criterion, device, epoch,
                    log_interval=50) -> Dict:
    model.train()
    total_loss = seg_loss_sum = meas_loss_sum = 0.0
    n_batches  = 0

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
            logger.info(f"  Epoch {epoch} [{i+1}/{len(loader)}] "
                        f"Loss={loss.item():.4f} "
                        f"(seg={sl.item():.4f}, meas={ml.item():.4f})")

    return {
        "train_loss":      total_loss    / max(n_batches, 1),
        "train_seg_loss":  seg_loss_sum  / max(n_batches, 1),
        "train_meas_loss": meas_loss_sum / max(n_batches, 1),
    }


def train(args):
    exp_cfg  = EXPERIMENTS[args.exp]
    exp_name = exp_cfg["name"] + ("_no_norm" if args.no_mediapipe else "")

    logger.info(f"\n{'='*50}")
    logger.info(f"실험: {exp_name}  |  {exp_cfg['desc']}")
    logger.info(f"json_dirs:  {args.json_dirs}")
    logger.info(f"image_dirs: {args.image_dirs}")
    logger.info(f"{'='*50}\n")

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
            config={**vars(args), "exp_desc": exp_cfg["desc"]},
        )
    elif args.wandb_project:
        logger.warning("wandb 미설치")

    # ── 핵심 변경: json_dirs / image_dirs 리스트로 전달 ──────────
    loaders = get_dataloaders(
        json_dirs=args.json_dirs,
        image_dirs=args.image_dirs,
        view_type=args.view_type,
        use_mediapipe=not args.no_mediapipe,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    logger.info(f"Train: {len(loaders['train'].dataset)}  "
                f"Val: {len(loaders['val'].dataset)}  "
                f"Test: {len(loaders['test'].dataset)}")

    model = exp_cfg["model_cls"](
        body_dim=10,          # 원본과 동일
        num_measurements=5,
    ).to(device)
    logger.info(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    criterion = CombinedLoss(lambda1=exp_cfg["lambda1"], lambda2=exp_cfg["lambda2"])
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_mae = float("inf")
    history  = []

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n[Epoch {epoch}/{args.epochs}]")

        train_metrics = train_one_epoch(
            model, loaders["train"], optimizer, criterion, device, epoch)
        val_metrics = evaluate(model, loaders["val"], criterion, device)
        scheduler.step()

        logger.info(f"  Train Loss: {train_metrics['train_loss']:.4f}  "
                    f"Val Loss: {val_metrics['total_loss']:.4f}  "
                    f"Val MAE: {val_metrics['mae_overall']:.2f}cm")
        logger.info("  Val MAE per item:")
        for name, mae in val_metrics["mae_per_item"].items():
            logger.info(f"    {name}: {mae:.2f}cm")

        row = {**train_metrics,
               **{f"val_{k}": v for k, v in val_metrics.items()},
               "epoch": epoch}
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
            logger.info(f"  Best 저장 (MAE={best_mae:.2f}cm)")

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
            wandb.log(log_dict)

    # ── 테스트 ─────────────────────────────────────────────────
    logger.info("\n[Test 평가]")
    ckpt = torch.load(os.path.join(ckpt_dir, "best.pth"), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, loaders["test"], criterion, device)

    logger.info(f"Test MAE:  {test_metrics['mae_overall']:.2f}cm")
    logger.info(f"Test RMSE: {test_metrics['rmse_overall']:.2f}cm")
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
        test_log = {"test/mae": test_metrics["mae_overall"],
                    "test/rmse": test_metrics["rmse_overall"]}
        for name, mae in test_metrics["mae_per_item"].items():
            test_log[f"test/mae_{name}"] = mae
        wandb.log(test_log)
        wandb.finish()

    return results


def parse_args():
    p = argparse.ArgumentParser(description="의류 치수 추정 (멀티 카테고리)")

    # ── 변경된 인자 ──────────────────────────────────────────────
    p.add_argument("--json_dirs",   nargs="+", required=True,
                   help="JSON 디렉토리들 (순서대로, 예: /content/label_blouse /content/label_shirt)")
    p.add_argument("--image_dirs",  nargs="+", required=True,
                   help="이미지 디렉토리들 (json_dirs와 순서 동일)")
    # ────────────────────────────────────────────────────────────

    p.add_argument("--exp",           type=int, default=4, choices=[1,2,3,4,5,6])
    p.add_argument("--no_mediapipe",  action="store_true")
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
    p.add_argument("--wandb_project", type=str,   default=None)
    p.add_argument("--wandb_run",     type=str,   default=None)
    p.add_argument("--ckpt_interval", type=int,   default=10)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if len(args.json_dirs) != len(args.image_dirs):
        raise ValueError("--json_dirs와 --image_dirs 개수가 같아야 합니다.")

    train(args)