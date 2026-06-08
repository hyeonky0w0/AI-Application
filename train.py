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


#Ablation 실험 설정
EXPERIMENTS = {
    1: {
        "name":      "exp1_mlp_baseline",
        "desc":      "Exp1: 신체 치수만 입력 (MLP Baseline, 이미지 없음)",
        "model_cls": BaselineMLP,
        "mediapipe": False,
        "lambda1":   0.0,   # 세그멘테이션 loss 없음
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
        "lambda2":   0.5,
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


#평가 함수
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

    all_pred = torch.cat(all_pred, dim=0)  # [N, 5]
    all_gt   = torch.cat(all_gt,   dim=0)  # [N, 5]

    # cm 단위로 변환 후 MAE/RMSE 계산
    pred_cm = denormalize_clothes(all_pred)
    gt_cm   = denormalize_clothes(all_gt)

    mae_per_item  = (pred_cm - gt_cm).abs().mean(dim=0)   # [5]
    rmse_per_item = ((pred_cm - gt_cm)**2).mean(dim=0).sqrt()

    item_names = [k.split(".")[-1] for k in CLOTHES_KEYS]

    metrics = {
        "total_loss":  total_loss  / max(n_batches, 1),
        "seg_loss":    seg_loss_sum / max(n_batches, 1),
        "meas_loss":   meas_loss_sum / max(n_batches, 1),
        "mae_overall": mae_per_item.mean().item(),
        "rmse_overall":rmse_per_item.mean().item(),
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


#학습 루프
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

        # Gradient clipping — 학습 안정성
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
    exp_name = exp_cfg["name"] + ("_no_norm" if args.no_mediapipe else "")

    logger.info(f"\n{'='*50}")
    logger.info(f"실험: {exp_name}")
    logger.info(f"설명: {exp_cfg['desc']}")
    logger.info(f"MediaPipe: {use_mediapipe}")
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

    model = exp_cfg["model_cls"](
        body_dim=3,
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

    #학습 루프
    best_mae  = float("inf")
    history   = []

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n[Epoch {epoch}/{args.epochs}]")

        train_metrics = train_one_epoch(
            model, loaders["train"], optimizer, criterion, device, epoch
        )
        val_metrics = evaluate(model, loaders["val"], criterion, device)
        scheduler.step()

        # if epoch == 1 and hasattr(loaders["train"].dataset, "scale_normalizer"):
        #     norm = loaders["train"].dataset.scale_normalizer
        #     if norm is not None:
        #         rate = norm.success_rate()
        #         logger.info(f"  MediaPipe 성공률: {rate*100:.1f}% "
        #                     f"({norm.n_success}/{norm.n_total})")
        #         if rate < 0.5:
        #             logger.warning("성공률 50% 미만: letterbox fallback 비중 높음")

        # 로그
        logger.info(
            f"  Train Loss: {train_metrics['train_loss']:.4f}  "
            f"Val Loss: {val_metrics['total_loss']:.4f}  "
            f"Val MAE: {val_metrics['mae_overall']:.2f}cm"
        )
        logger.info("  Val MAE per item:")
        for name, mae in val_metrics["mae_per_item"].items():
            logger.info(f"    {name}: {mae:.2f}cm")

        # 체크포인트 저장
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

        #wandb 로깅
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

    #테스트 평가
    logger.info("\n[Test 평가]")
    ckpt = torch.load(os.path.join(ckpt_dir, "best.pth"), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, loaders["test"], criterion, device)

    logger.info(f"Test MAE: {test_metrics['mae_overall']:.2f}cm")
    logger.info(f"Test RMSE: {test_metrics['rmse_overall']:.2f}cm")
    logger.info("Test MAE per item:")
    for name, mae in test_metrics["mae_per_item"].items():
        logger.info(f"  {name}: {mae:.2f}cm")

    # 결과 저장
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


#Ablation 전체 실행 함수
def run_all_ablations(args):
    # Exp1~4: 구성요소 ablation, Exp6: 최종 모델 (Q-Former)
    ablation_ids = [1, 2, 3, 4, 6]
    all_results = {}
    for exp_id in ablation_ids:
        logger.info(f"\n{'#'*60}")
        logger.info(f"# Ablation Exp {exp_id}: {EXPERIMENTS[exp_id]['desc']}")
        logger.info(f"{'#'*60}")
        args.exp = exp_id
        result = train(args)
        all_results[exp_id] = result

    # 비교 테이블
    item_keys = ["shoulder_width", "front_length", "chest_size", "waist_size", "sleeve_length"]
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
    p.add_argument("--json_dir",    type=str, required=True,
                   help="JSON 라벨 디렉토리 (예: /data/.../label_blouse)")
    p.add_argument("--image_dir",   type=str, nargs="+", required=True,
                   help="이미지 디렉토리 (여러 개 가능, 예: /data/.../image_blouse /data/.../image_shirt)")
    p.add_argument("--exp",         type=int,   default=4, choices=[1,2,3,4,5,6],
                   help="실험 번호 (1=MLP, 2=세그no정규화, 3=FiLM없음, 4=Full)")
    p.add_argument("--all",         action="store_true",
                   help="Ablation 전체 순차 실행 (Exp1~4 + Exp6 최종 모델)")
    p.add_argument("--no_mediapipe",action="store_true",
                   help="MediaPipe 스케일 정규화 비활성화 (Ablation용)")
    p.add_argument("--categories",  nargs="+",  default=["blouse", "coat", "shirt"],
                   help="학습할 의류 카테고리")
    p.add_argument("--view_type",   type=str,   default="front",
                   choices=["front", "wear", "all"],
                   help="사용할 이미지 각도 타입")
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--val_ratio",   type=float, default=0.1)
    p.add_argument("--test_ratio",  type=float, default=0.1)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--eval_only",   action="store_true",
                   help="체크포인트 로드 후 val/test 평가만 실행")
    p.add_argument("--body_ablation", action="store_true",
                   help="신체치수 피처 하나씩 마스킹해 중요도 측정")
    p.add_argument("--ckpt",        type=str,   default=None,
                   help="--eval_only / --body_ablation 시 사용할 .pth 경로 (미지정 시 checkpoints/{exp}/best.pth)")
    p.add_argument("--wandb_project", type=str, default=None,
                   help="W&B 프로젝트 이름 (미지정 시 W&B 비활성화)")
    p.add_argument("--wandb_run",   type=str,   default=None,
                   help="W&B run 이름 (미지정 시 exp_name 사용)")
    p.add_argument("--ckpt_interval", type=int, default=10,
                   help="N 에폭마다 체크포인트 저장 (기본 10)")
    return p.parse_args()


BODY_FEATURE_NAMES = [
    "body_height", "breast_size_female", "waist_size", "hip_seize",
    "shoulders_width", "arm_length", "waist_height", "back_length",
    "weight", "gender",
]


def evaluate_masked(model, loader, criterion, device, mask_idx: int) -> float:
    #body_vec의 mask_idx 번째 피처를 0으로 마스킹한 뒤 MAE 반환
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
    #각 신체치수 피처를 하나씩 0으로 마스킹해 MAE 변화 측정
    exp_cfg   = EXPERIMENTS[args.exp]
    exp_name  = exp_cfg["name"] + ("_no_norm" if args.no_mediapipe else "")
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

    model = exp_cfg["model_cls"](body_dim=10, num_measurements=5).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    criterion = CombinedLoss(lambda1=exp_cfg["lambda1"], lambda2=exp_cfg["lambda2"])

    # 베이스라인 (마스킹 없음)
    base = evaluate(model, loaders["test"], criterion, device)
    base_mae     = base["mae_overall"]
    base_per     = list(base["mae_per_item"].values())
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
    exp_name = exp_cfg["name"] + ("_no_norm" if args.no_mediapipe else "")
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

    model = exp_cfg["model_cls"](body_dim=10, num_measurements=5).to(device)
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