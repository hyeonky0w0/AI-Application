import os, sys, json, glob, random, argparse
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
matplotlib.rcParams["axes.unicode_minus"] = False
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from dataset import (
    letterbox, polygon_to_mask, normalize_body,
    extract_clothes_measurements, denormalize_clothes,
    TARGET_SIZE, CLOTHES_KEYS, IMAGENET_MEAN, IMAGENET_STD,
)
from model import (
    ClothingMeasurementNet, ClothingMeasurementNetQFormer,
    ClothingMeasurementNetQFormerMP,
    ImageOnlyQFormerModel, ImageBodyNoFiLMQFormerModel,
    BaselineMLP, CombinedLoss,
)

ITEM_LABELS = ["Shoulder", "Length", "Chest", "Waist", "Sleeve"]
ITEM_KEYS   = [k.split(".")[-1] for k in CLOTHES_KEYS]

MODEL_MAP = {
    1: BaselineMLP,
    2: ImageOnlyQFormerModel,
    3: ImageBodyNoFiLMQFormerModel,
    4: ClothingMeasurementNet,
    6: ClothingMeasurementNetQFormer,
    7: ClothingMeasurementNetQFormerMP,
}


def load_model(ckpt_path: str, device: torch.device):
    ckpt      = torch.load(ckpt_path, map_location=device)
    exp_id    = ckpt["args"].get("exp", 6)
    model_cls = MODEL_MAP.get(exp_id, ClothingMeasurementNetQFormer)
    model     = model_cls().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def preprocess_image(img_bgr: np.ndarray):
    lb  = letterbox(img_bgr, TARGET_SIZE)
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    tf  = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return tf(pil).unsqueeze(0), lb  # [1,3,H,W], letterbox numpy


def load_sample(jpath: str, image_dirs: list):
    with open(jpath, encoding="utf-8") as f:
        data = json.load(f)

    rel      = data.get("dataset", {}).get("dataset.image_path", "")
    fname    = os.path.basename(rel) if rel else os.path.basename(jpath).replace(".json", ".jpg")
    img_path = None
    for d in image_dirs:
        c = os.path.join(d, fname)
        if os.path.exists(c):
            img_path = c
            break
    if img_path is None:
        return None

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None

    meta         = data.get("metadata.model", {})
    body_vec     = normalize_body(meta)
    clothes_norm = extract_clothes_measurements(data)
    if clothes_norm is None:
        return None

    clothes_cm = denormalize_clothes(clothes_norm)

    ds    = data.get("dataset", {})
    img_w = int(ds.get("dataset.width",  img_bgr.shape[1]))
    img_h = int(ds.get("dataset.height", img_bgr.shape[0]))

    annots = data.get("annotation", [])
    if annots and annots[0].get("annotation_point"):
        mask_np = polygon_to_mask(annots[0]["annotation_point"], img_w, img_h, TARGET_SIZE)
    else:
        mask_np = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)

    return {
        "fname":      fname,
        "img_bgr":    img_bgr,
        "mask_gt":    mask_np,
        "body_vec":   body_vec,
        "clothes_cm": clothes_cm,
    }


@torch.no_grad()
def run_inference(model, sample: dict, device: torch.device):
    img_tensor, lb_bgr = preprocess_image(sample["img_bgr"])
    body_tensor = sample["body_vec"].unsqueeze(0).to(device)
    img_tensor  = img_tensor.to(device)

    seg_pred, meas_pred = model(img_tensor, body_tensor)

    seg_np  = seg_pred[0, 0].cpu().numpy()
    meas_cm = denormalize_clothes(meas_pred[0].cpu())

    lb_rgb  = cv2.cvtColor(lb_bgr, cv2.COLOR_BGR2RGB)
    overlay = lb_rgb.copy()
    seg_bin = (seg_np > 0.5).astype(np.uint8)
    overlay[seg_bin == 1] = (
        overlay[seg_bin == 1] * 0.45 + np.array([255, 80, 80]) * 0.55
    ).astype(np.uint8)

    return {
        "lb_rgb":  lb_rgb,
        "seg_np":  seg_np,
        "overlay": overlay,
        "meas_cm": meas_cm,
    }


def visualize(samples_with_results, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    n = len(samples_with_results)

    for i, (sample, result) in enumerate(samples_with_results):
        fig = plt.figure(figsize=(18, 6))
        fig.suptitle(sample["fname"], fontsize=10, y=1.01)

        ax1 = fig.add_subplot(1, 4, 1)
        ax1.imshow(cv2.cvtColor(sample["img_bgr"], cv2.COLOR_BGR2RGB))
        ax1.set_title("Original", fontsize=10)
        ax1.axis("off")

        ax2 = fig.add_subplot(1, 4, 2)
        ax2.imshow(result["lb_rgb"])
        ax2.set_title("Letterbox (224x224)", fontsize=10)
        ax2.axis("off")

        ax3 = fig.add_subplot(1, 4, 3)
        ax3.imshow(result["seg_np"], cmap="RdYlGn", vmin=0, vmax=1)
        ax3.set_title("Predicted Segmentation", fontsize=10)
        ax3.axis("off")

        ax4 = fig.add_subplot(1, 4, 4)
        ax4.imshow(result["overlay"])
        patch = mpatches.Patch(color=(1.0, 0.31, 0.31), alpha=0.6, label="Predicted mask")
        ax4.legend(handles=[patch], loc="lower right", fontsize=8)
        ax4.set_title("Mask Overlay", fontsize=10)
        ax4.axis("off")

        plt.tight_layout()

        gt   = sample["clothes_cm"].numpy()
        pred = result["meas_cm"].numpy()
        lines = ["           GT    Pred   Error"]
        for label, g, p in zip(ITEM_LABELS, gt, pred):
            err  = p - g
            sign = "+" if err >= 0 else ""
            lines.append(f"{label:10s} {g:5.1f}  {p:5.1f}  {sign}{err:.1f}cm")
        mae = float(np.abs(pred - gt).mean())
        lines.append(f"{'MAE':10s}                {mae:.2f}cm")

        fig.text(0.5, -0.06, "\n".join(lines), ha="center", va="top",
                 fontsize=10, family="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8))

        path = os.path.join(out_dir, f"sample_{i+1:02d}_{sample['fname'].replace('.jpg','')}.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  saved: {path}")

    if n > 1:
        fig, axes = plt.subplots(1, len(ITEM_LABELS), figsize=(16, 4), sharey=False)
        fig.suptitle("Prediction Error per Item (|pred - GT|, cm)", fontsize=12)

        errors_per_item = [[] for _ in ITEM_LABELS]
        for sample, result in samples_with_results:
            gt   = sample["clothes_cm"].numpy()
            pred = result["meas_cm"].numpy()
            for j in range(len(ITEM_LABELS)):
                errors_per_item[j].append(abs(pred[j] - gt[j]))

        for j, (ax, label) in enumerate(zip(axes, ITEM_LABELS)):
            vals = errors_per_item[j]
            ax.bar(range(n), vals, color="#4C72B0", alpha=0.8)
            ax.axhline(np.mean(vals), color="red", linestyle="--", linewidth=1.5,
                       label=f"mean {np.mean(vals):.1f}cm")
            ax.set_title(label, fontsize=10)
            ax.set_xlabel("Sample")
            ax.set_ylabel("Error (cm)")
            ax.legend(fontsize=8)
            ax.set_xticks(range(n))

        plt.tight_layout()
        summary_path = os.path.join(out_dir, "error_summary.png")
        plt.savefig(summary_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  saved: {summary_path}")


def main():
    ap = argparse.ArgumentParser(description="Clothing measurement inference demo")
    ap.add_argument("--ckpt",      required=True, help="checkpoint path (e.g. checkpoints/exp6_qformer/best.pth)")
    ap.add_argument("--n",         type=int, default=5, help="number of samples to visualize")
    ap.add_argument("--out_dir",   default="./infer_output")
    ap.add_argument("--seed",      type=int, default=0, help="random seed for sample selection")
    ap.add_argument("--split",     default="test", choices=["test", "val", "train", "all"],
                    help="which split to sample from (default: test)")
    ap.add_argument("--image_dir", nargs="+", default=None,
                    help="image directories (defaults to checkpoint args)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading model...")
    model, ckpt = load_model(args.ckpt, device)
    ckpt_args   = ckpt["args"]
    print(f"  exp{ckpt_args.get('exp')} ({type(model).__name__}), epoch={ckpt['epoch']}, best_val_mae={ckpt['best_mae']:.2f}cm")

    image_dirs = args.image_dir or ckpt_args["image_dir"]
    if isinstance(image_dirs, str):
        image_dirs = [image_dirs]

    print(f"Collecting samples (split={args.split})...")
    if args.split == "all":
        json_dir        = ckpt_args["json_dir"]
        all_jsons       = glob.glob(os.path.join(json_dir, "**", "*.json"), recursive=True)
        candidate_paths = all_jsons
        filter_view     = ckpt_args.get("view_type", "wear")
    else:
        from dataset import ClothingDataset
        ds = ClothingDataset(
            json_dir      = ckpt_args["json_dir"],
            image_dir     = image_dirs,
            categories    = ckpt_args.get("categories", ["blouse"]),
            view_type     = ckpt_args.get("view_type", "wear"),
            use_mediapipe = False,
            augment       = False,
            split         = args.split,
            val_ratio     = ckpt_args.get("val_ratio", 0.1),
            test_ratio    = ckpt_args.get("test_ratio", 0.1),
            seed          = ckpt_args.get("seed", 42),
        )
        candidate_paths = ds.samples
        filter_view     = None
        print(f"  {args.split} set size: {len(candidate_paths)}")

    random.seed(args.seed)
    random.shuffle(candidate_paths)

    samples = []
    for jpath in candidate_paths:
        if len(samples) >= args.n:
            break
        try:
            if filter_view:
                with open(jpath, encoding="utf-8") as f:
                    data = json.load(f)
                vt = data.get("metadata.model", {}).get("metadata.model.type", "").lower()
                if vt != filter_view:
                    continue
            s = load_sample(jpath, image_dirs)
            if s is not None:
                samples.append(s)
                print(f"  loaded: {s['fname']}")
        except Exception:
            continue

    if not samples:
        print("No valid samples found.")
        return

    print(f"\nRunning inference ({len(samples)} samples)...")
    results = []
    for s in samples:
        r    = run_inference(model, s, device)
        gt   = s["clothes_cm"].numpy()
        pred = r["meas_cm"].numpy()
        mae  = float(np.abs(pred - gt).mean())
        print(f"  {s['fname']:50s}  MAE={mae:.2f}cm")
        results.append((s, r))

    print("\nSaving visualizations...")
    visualize(results, args.out_dir)
    print(f"\nDone: {args.out_dir}")


if __name__ == "__main__":
    main()
