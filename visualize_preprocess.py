#전처리 시각화 용도
import os, json, glob, argparse, random
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from dataset import letterbox, polygon_to_mask, TARGET_SIZE


def load_sample(jpath: str, image_dir: str):
    with open(jpath, encoding="utf-8") as f:
        data = json.load(f)

    # 이미지 경로
    rel = data.get("dataset", {}).get("dataset.image_path", "")
    fname = os.path.basename(rel) if rel else os.path.basename(jpath).replace(".json", ".jpg")
    img_path = os.path.join(image_dir, fname)
    if not os.path.exists(img_path):
        return None

    image_bgr = cv2.imread(img_path)
    if image_bgr is None:
        return None

    ds = data.get("dataset", {})
    img_w = int(ds.get("dataset.width",  image_bgr.shape[1]))
    img_h = int(ds.get("dataset.height", image_bgr.shape[0]))

    annotations = data.get("annotation", [])
    if annotations and annotations[0].get("annotation_point"):
        pts = annotations[0]["annotation_point"]
        mask = polygon_to_mask(pts, img_w, img_h, TARGET_SIZE)
    else:
        mask = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)

    lb = letterbox(image_bgr, TARGET_SIZE)
    lb_rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
    orig_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 마스크 오버레이
    overlay = lb_rgb.copy()
    overlay[mask == 1] = (
        overlay[mask == 1] * 0.5 + np.array([255, 0, 0]) * 0.5
    ).astype(np.uint8)

    meta = data.get("metadata.model", {})
    clothes = data.get("metadata.clothes", {})
    shoulder_w = clothes.get("metadata.top.shoulder_width", "?")
    chest     = clothes.get("metadata.top.chest_size",     "?")
    view_type = meta.get("metadata.model.type", "?")

    return {
        "orig":    orig_rgb,
        "lb":      lb_rgb,
        "mask":    mask,
        "overlay": overlay,
        "fname":   fname,
        "view":    view_type,
        "shoulder_w": shoulder_w,
        "chest":      chest,
        "mask_px":    int(mask.sum()),
    }


def visualize(samples, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    n = len(samples)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = [axes]

    col_titles = ["원본 이미지", "Letterbox (224×224)", "세그멘테이션 마스크", "마스크 오버레이"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)

    for row, s in enumerate(samples):
        # 1) 원본
        axes[row][0].imshow(s["orig"])
        axes[row][0].set_xlabel(
            f"{s['fname'][:30]}\nview={s['view']}  shoulder={s['shoulder_w']}cm  chest={s['chest']}cm",
            fontsize=8
        )

        # 2) letterbox
        axes[row][1].imshow(s["lb"])
        h, w = s["orig"].shape[:2]
        axes[row][1].set_xlabel(f"원본 {w}×{h} → 224×224 (회색 패딩)", fontsize=8)

        # 3) 마스크 (흑백)
        axes[row][2].imshow(s["mask"], cmap="gray", vmin=0, vmax=1)
        axes[row][2].set_xlabel(f"옷 영역: {s['mask_px']:,} px ({s['mask_px']/TARGET_SIZE**2*100:.1f}%)", fontsize=8)

        # 4) 오버레이
        axes[row][3].imshow(s["overlay"])
        patch = mpatches.Patch(color="red", alpha=0.5, label="옷 영역")
        axes[row][3].legend(handles=[patch], loc="lower right", fontsize=8)
        axes[row][3].set_xlabel("빨간 영역 = 세그멘테이션 마스크", fontsize=8)

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    out_path = os.path.join(out_dir, "preprocess_vis.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"저장 완료: {out_path}")

    # 샘플별 개별 저장도
    for i, s in enumerate(samples):
        fig2, axes2 = plt.subplots(1, 4, figsize=(16, 4))
        axes2[0].imshow(s["orig"]);    axes2[0].set_title("원본")
        axes2[1].imshow(s["lb"]);      axes2[1].set_title("Letterbox")
        axes2[2].imshow(s["mask"], cmap="gray"); axes2[2].set_title("마스크")
        axes2[3].imshow(s["overlay"]); axes2[3].set_title("오버레이")
        for ax in axes2:
            ax.axis("off")
        fig2.suptitle(s["fname"], fontsize=9)
        plt.tight_layout()
        path2 = os.path.join(out_dir, f"sample_{i+1:02d}.png")
        plt.savefig(path2, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  개별 저장: {path2}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir",  required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--n",         type=int, default=3, help="시각화할 샘플 수")
    ap.add_argument("--view_type", default="wear",
                    help="wear | front | back | all")
    ap.add_argument("--out_dir",   default="./vis_output")
    ap.add_argument("--seed",      type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    all_jsons = glob.glob(os.path.join(args.json_dir, "*.json"))
    random.shuffle(all_jsons)

    samples = []
    for jpath in all_jsons:
        if len(samples) >= args.n:
            break
        try:
            with open(jpath, encoding="utf-8") as f:
                data = json.load(f)
            vt = data.get("metadata.model", {}).get("metadata.model.type", "").lower()
            if args.view_type != "all" and vt != args.view_type:
                continue
            s = load_sample(jpath, args.image_dir)
            if s is not None:
                samples.append(s)
                print(f"  로드: {s['fname']}  (mask_px={s['mask_px']:,})")
        except Exception as e:
            continue

    visualize(samples, args.out_dir)


if __name__ == "__main__":
    main()
