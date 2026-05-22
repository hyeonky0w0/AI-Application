import os
import json
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Optional, Tuple, List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TARGET_SIZE = 224

BODY_STATS = {
    "body_height":          (161.6, 5.0),
    "breast_size_female":   (86.5,  7.2),
    "waist_size":           (73.5,  8.6),
    "hip_seize":            (93.0,  5.9),
    "shoulders_width":      (40.4,  2.4),
    "arm_length":           (52.7,  2.3),
    "waist_height":         (96.5,  5.3),
    "back_length":          (40.8,  2.7),
    "weight":               (54.9,  7.7),
    "gender":               (0.5,   0.5),
}

CLOTHES_KEYS = [
    "metadata.top.shoulder_width",
    "metadata.top.front_length",
    "metadata.top.chest_size",
    "metadata.top.waist_size",
    "metadata.top.sleeve_length",
]

CLOTHES_VALID_RANGE = {
    "metadata.top.shoulder_width": (30,  60),
    "metadata.top.front_length":   (40, 120),
    "metadata.top.chest_size":     (70, 160),
    "metadata.top.waist_size":     (60, 160),
    "metadata.top.sleeve_length":  (10,  80),
}

CLOTHES_STATS = {
    "metadata.top.shoulder_width": (40.5, 5.7),
    "metadata.top.front_length":   (61.4, 7.9),
    "metadata.top.chest_size":     (96.2, 11.5),
    "metadata.top.waist_size":     (98.9, 13.4),
    "metadata.top.sleeve_length":  (42.2, 15.1),
}

# ImageNet 정규화
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class ScaleNormalizer:
    def __init__(self):
        self.n_total   = 0
        self.n_success = 0
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            import urllib.request, os

            model_path = "/tmp/pose_landmarker.task"
            if not os.path.exists(model_path):
                urllib.request.urlretrieve(
                    "https://storage.googleapis.com/mediapipe-models/"
                    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
                    model_path
                )

            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=False,
            )
            self.detector    = vision.PoseLandmarker.create_from_options(options)
            self.use_new_api = True
            self.available   = True
            logger.info("MediaPipe PoseLandmarker 초기화 (new API)")

        except Exception as e:
            self.available   = False
            self.use_new_api = False
            logger.warning(f"mediapipe 없이 letterbox resize만 사용: {e}")

    def success_rate(self):
        return self.n_success / self.n_total if self.n_total > 0 else 0.0

    def __call__(
        self,
        image: np.ndarray,
        shoulder_width_cm: float,
        target_px_per_cm: float = 5.0,
    ):
        self.n_total += 1
        if not self.available or shoulder_width_cm <= 0:
            return letterbox(image, TARGET_SIZE), False

        import mediapipe as mp
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.use_new_api:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.detector.detect(mp_image)
            if not result.pose_landmarks:
                return letterbox(image, TARGET_SIZE), False

            h, w = image.shape[:2]
            lms = result.pose_landmarks[0]
            lx = int(lms[11].x * w);  ly = int(lms[11].y * h)
            rx = int(lms[12].x * w);  ry = int(lms[12].y * h)
        else:
            return letterbox(image, TARGET_SIZE), False

        shoulder_px = np.sqrt((rx - lx)**2 + (ry - ly)**2)
        if shoulder_px < 10:
            return letterbox(image, TARGET_SIZE), False

        scale = target_px_per_cm / (shoulder_px / shoulder_width_cm)
        new_w, new_h = int(image.shape[1] * scale), int(image.shape[0] * scale)
        if new_w < 10 or new_h < 10:
            return letterbox(image, TARGET_SIZE), False

        rescaled = cv2.resize(image, (new_w, new_h))
        self.n_success += 1
        return center_crop_or_pad(rescaled, TARGET_SIZE), True


def letterbox(image: np.ndarray, size: int = TARGET_SIZE) -> np.ndarray:
    h, w = image.shape[:2]
    scale = size / max(h, w)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top    = (size - nh) // 2
    bottom = size - nh - top
    left   = (size - nw) // 2
    right  = size - nw - left
    return cv2.copyMakeBorder(resized, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=(128, 128, 128))


def center_crop_or_pad(image: np.ndarray, size: int = TARGET_SIZE) -> np.ndarray:
    h, w = image.shape[:2]
    if h >= size and w >= size:
        sy = (h - size) // 2
        sx = (w - size) // 2
        return image[sy:sy+size, sx:sx+size]
    return letterbox(image, size)


def polygon_to_mask(
    points: List[float],
    img_w: int,
    img_h: int,
    size: int = TARGET_SIZE,
) -> np.ndarray:
    pts = np.array(points, dtype=np.float32).reshape(-1, 2).astype(np.int32)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], color=1)

    scale = size / max(img_h, img_w)
    nw, nh = int(img_w * scale), int(img_h * scale)
    resized = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

    top    = (size - nh) // 2
    bottom = size - nh - top
    left   = (size - nw) // 2
    right  = size - nw - left
    return cv2.copyMakeBorder(resized, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=0)


def normalize_body(meta: dict) -> torch.Tensor:
    gender = 0.0 if meta.get("metadata.model.gender", "FEMALE") == "FEMALE" else 1.0

    def g(key, default):
        v = meta.get(key, default)
        try:
            return float(v) if v not in (None, "null", "") else float(default)
        except (ValueError, TypeError):
            return float(default)

    raw = {
        "body_height":        g("metadata.model.body_height", 161),
        "breast_size_female": g("metadata.model.breast_size_female", 87),
        "waist_size":         g("metadata.model.waist_size", 74),
        "hip_seize":          g("metadata.model.hip_seize", 93),
        "shoulders_width":    g("metadata.model.shoulders_width", 39),
        "arm_length":         g("metadata.model.arm_length", 54),
        "waist_height":       g("metadata.model.waist_height", 97),
        "back_length":        g("metadata.model.back_length", 42),
        "weight":             g("metadata.model.weight", 57),
        "gender":             gender,
    }
    return torch.tensor(
        [(v - BODY_STATS[k][0]) / BODY_STATS[k][1] for k, v in raw.items()],
        dtype=torch.float32
    )


def extract_clothes_measurements(data: dict) -> Optional[torch.Tensor]:
    clothes_meta = data.get("metadata.clothes", {})
    values = []
    for key in CLOTHES_KEYS:
        val = clothes_meta.get(key) or data.get(key)
        if val in (None, "null", ""):
            return None
        try:
            v = float(val)
            lo, hi = CLOTHES_VALID_RANGE[key]
            if not (lo <= v <= hi):
                return None
            mean, std = CLOTHES_STATS[key]
            values.append((v - mean) / std)
        except (ValueError, TypeError):
            return None
    return torch.tensor(values, dtype=torch.float32)


def denormalize_clothes(tensor: torch.Tensor) -> torch.Tensor:
    means = torch.tensor([CLOTHES_STATS[k][0] for k in CLOTHES_KEYS],
                         dtype=torch.float32, device=tensor.device)
    stds  = torch.tensor([CLOTHES_STATS[k][1] for k in CLOTHES_KEYS],
                         dtype=torch.float32, device=tensor.device)
    return tensor * stds + means


class ClothingDataset(Dataset):
    def __init__(
        self,
        json_dir: str,
        image_dir: str,
        cache_dir: Optional[str] = None,      # ← 추가: 전처리 캐시 디렉토리
        categories: List[str] = ["blouse"],
        view_type: str = "wear",
        use_mediapipe: bool = True,
        augment: bool = False,
        split: str = "train",
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ):
        self.json_dir  = json_dir
        self.image_dir = image_dir
        self.cache_dir = cache_dir
        self.view_type = view_type
        self.augment   = augment

        # 캐시가 있으면 MediaPipe 불필요
        self.scale_normalizer = (
            ScaleNormalizer() if (use_mediapipe and cache_dir is None) else None
        )

        # 이미지 to 텐서
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        # augmentation (학습용)
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        ]) if augment else None

        self.samples = self._collect_samples(categories)

        # train/val/test split
        np.random.seed(seed)
        idx = np.random.permutation(len(self.samples))
        n_test = int(len(idx) * test_ratio)
        n_val  = int(len(idx) * val_ratio)
        if split == "test":
            idx = idx[:n_test]
        elif split == "val":
            idx = idx[n_test:n_test + n_val]
        else:
            idx = idx[n_test + n_val:]
        self.samples = [self.samples[i] for i in idx]
        logger.info(f"{split} 샘플 수: {len(self.samples)}")

    def _collect_samples(self, categories: List[str]) -> List[str]:
        samples = []
        skipped_type = 0
        skipped_null = 0
        skipped_img  = 0

        for cat in categories:
            search_dirs = [
                self.json_dir,
                os.path.join(self.json_dir, f"label_{cat}"),
                os.path.join(self.json_dir, cat),
            ]
            found_jsons = []
            for d in search_dirs:
                pattern = os.path.join(d, "*.json")
                found_jsons.extend(glob.glob(pattern))

            if not found_jsons:
                logger.warning(f"JSON 없음: {self.json_dir} (카테고리: {cat})")
                continue

            for jpath in found_jsons:
                try:
                    with open(jpath, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    json_type = data.get("metadata.model", {}).get(
                        "metadata.model.type", ""
                    ).lower()

                    if self.view_type != "all" and json_type != self.view_type:
                        skipped_type += 1
                        continue

                    if extract_clothes_measurements(data) is None:
                        skipped_null += 1
                        continue

                    img_path = self._json_to_img_path(data, jpath)
                    if not img_path or not os.path.exists(img_path):
                        skipped_img += 1
                        continue

                    samples.append(jpath)

                except Exception:
                    continue

        return samples

    def _json_to_img_path(self, data: dict, jpath: str) -> Optional[str]:
        rel = data.get("dataset", {}).get("dataset.image_path", "")
        if rel:
            fname = os.path.basename(rel)
            candidate = os.path.join(self.image_dir, fname)
            if os.path.exists(candidate):
                return candidate

        json_fname = os.path.basename(jpath).replace(".json", ".jpg")
        candidate = os.path.join(self.image_dir, json_fname)
        if os.path.exists(candidate):
            return candidate

        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        jpath = self.samples[idx]

        with open(jpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_path = self._json_to_img_path(data, jpath)
        meta = data.get("metadata.model", {})
        body_vec = normalize_body(meta)
        shoulder_width_cm = float(meta.get("metadata.model.shoulders_width", 39) or 39)

        # ── 이미지 로딩: 캐시 우선, 없으면 원본 처리 ──────────────────────
        if self.cache_dir is not None:
            cache_name = os.path.splitext(os.path.basename(jpath))[0] + ".npy"
            cache_path = os.path.join(self.cache_dir, cache_name)
            if os.path.exists(cache_path):
                rgb = np.load(cache_path)   # RGB uint8 [H, W, 3]
            else:
                # 캐시 미스 — letterbox fallback
                image = cv2.imread(img_path) if img_path else None
                if image is None:
                    image = np.full((TARGET_SIZE, TARGET_SIZE, 3), 128, dtype=np.uint8)
                rgb = cv2.cvtColor(letterbox(image, TARGET_SIZE), cv2.COLOR_BGR2RGB)
        else:
            image = cv2.imread(img_path) if img_path else None
            if image is None:
                image = np.full((TARGET_SIZE, TARGET_SIZE, 3), 128, dtype=np.uint8)
            if self.scale_normalizer is not None:
                image, _ = self.scale_normalizer(image, shoulder_width_cm)
            else:
                image = letterbox(image, TARGET_SIZE)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # ──────────────────────────────────────────────────────────────────

        pil = Image.fromarray(rgb)
        if self.aug:
            pil = self.aug(pil)
        image_tensor = self.to_tensor(pil)

        annotations = data.get("annotation", [])
        ds = data.get("dataset", {})
        img_w = ds.get("dataset.width", TARGET_SIZE)
        img_h = ds.get("dataset.height", TARGET_SIZE)

        if annotations and annotations[0].get("annotation_point"):
            pts = annotations[0]["annotation_point"]
            mask_np = polygon_to_mask(pts, img_w, img_h, TARGET_SIZE)
        else:
            mask_np = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)

        mask = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0)

        clothes = extract_clothes_measurements(data)
        if clothes is None:
            clothes = torch.zeros(len(CLOTHES_KEYS), dtype=torch.float32)

        return {
            "image":      image_tensor,
            "body_vec":   body_vec,
            "mask":       mask,
            "clothes":    clothes,
            "path":       jpath,
            "model_type": meta.get("metadata.model.type", ""),
        }


# DataLoader 생성
def get_dataloaders(
    json_dir: str,
    image_dir: str,
    cache_dir: Optional[str] = None,          # ← 추가
    categories: List[str] = ["blouse"],
    view_type: str = "wear",
    use_mediapipe: bool = True,
    batch_size: int = 16,
    num_workers: int = 4,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    loaders = {}
    for split in ["train", "val", "test"]:
        ds = ClothingDataset(
            json_dir=json_dir,
            image_dir=image_dir,
            cache_dir=cache_dir,              # ← 추가
            categories=categories,
            view_type=view_type,
            use_mediapipe=use_mediapipe,
            augment=(split == "train"),
            split=split,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )
    return loaders


# 데이터 분포 확인용
def compute_dataset_stats(json_dir: str, categories: List[str], view_type: str = "wear"):
    body_vals    = {k: [] for k in BODY_STATS}
    clothes_vals = {k: [] for k in CLOTHES_KEYS}
    n_total = n_type = n_null = 0

    for cat in categories:
        search_dirs = [
            json_dir,
            os.path.join(json_dir, f"label_{cat}"),
            os.path.join(json_dir, cat),
        ]
        found = []
        for d in search_dirs:
            found.extend(glob.glob(os.path.join(d, "*.json")))

        for jpath in found:
            try:
                with open(jpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                n_total += 1

                meta = data.get("metadata.model", {})

                json_type = meta.get("metadata.model.type", "").lower()
                if view_type != "all" and json_type != view_type:
                    n_type += 1
                    continue

                def g(key, default):
                    v = meta.get(key, default)
                    try: return float(v) if v not in (None, "null", "") else float(default)
                    except: return float(default)

                body_vals["body_height"].append(g("metadata.model.body_height", 0))
                body_vals["breast_size_female"].append(g("metadata.model.breast_size_female", 0))
                body_vals["waist_size"].append(g("metadata.model.waist_size", 0))
                body_vals["hip_seize"].append(g("metadata.model.hip_seize", 0))
                body_vals["shoulders_width"].append(g("metadata.model.shoulders_width", 0))
                body_vals["arm_length"].append(g("metadata.model.arm_length", 0))
                body_vals["waist_height"].append(g("metadata.model.waist_height", 0))
                body_vals["back_length"].append(g("metadata.model.back_length", 0))
                body_vals["weight"].append(g("metadata.model.weight", 0))
                body_vals["gender"].append(0.0 if meta.get("metadata.model.gender") == "FEMALE" else 1.0)

                for key in CLOTHES_KEYS:
                    v = data.get(key) or data.get("metadata.clothes", {}).get(key)
                    if v not in (None, "null", ""):
                        try: clothes_vals[key].append(float(v))
                        except: pass
                    else:
                        n_null += 1

            except Exception:
                continue

    print(f"\n전체 json: {n_total}")
    print("\n신체 치수 분포")
    for k in list(BODY_STATS.keys()):
        if body_vals[k]:
            arr = np.array(body_vals[k])
            print(f"  {k:30s}: mean={arr.mean():.1f}, std={arr.std():.1f}, "
                  f"min={arr.min():.1f}, max={arr.max():.1f}  (n={len(arr)})")

    print("\n옷 치수 분포")
    for k in CLOTHES_KEYS:
        if clothes_vals[k]:
            arr = np.array(clothes_vals[k])
            print(f"  {k.split('.')[-1]:20s}: mean={arr.mean():.1f}, std={arr.std():.1f}, "
                  f"min={arr.min():.1f}, max={arr.max():.1f}  (n={len(arr)})")


if __name__ == "__main__":
    import sys, argparse
    p = argparse.ArgumentParser()
    p.add_argument("--json_dir",   type=str, default=None)
    p.add_argument("--image_dir",  type=str, default=None)
    p.add_argument("--categories", nargs="+", default=["blouse"])
    p.add_argument("--view_type",  type=str, default="wear")
    p.add_argument("--root",       type=str, default=None)
    args = p.parse_args()

    if args.root and not args.json_dir:
        sys.exit(0)

    if args.json_dir:
        compute_dataset_stats(args.json_dir, args.categories, args.view_type)