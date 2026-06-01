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

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class ScaleNormalizer:
    def __init__(self):
        self.n_total   = 0
        self.n_success = 0
        try:
            import mediapipe as mp
            self.pose = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                min_detection_confidence=0.5,
            )
            self.available = True
            logger.info("MediaPipe 초기화")
        except ImportError:
            self.available = False
            logger.warning("mediapipe 없이 letterbox resize만 사용")

    def success_rate(self) -> float:
        if self.n_total == 0:
            return 0.0
        return self.n_success / self.n_total

    def __call__(self, image, shoulder_width_cm, target_px_per_cm=5.0):
        self.n_total += 1
        if not self.available or shoulder_width_cm <= 0:
            return letterbox(image, TARGET_SIZE), False
        rgb    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)
        if not result.pose_landmarks:
            return letterbox(image, TARGET_SIZE), False
        h, w = image.shape[:2]
        lm   = result.pose_landmarks.landmark
        lx, ly = int(lm[11].x * w), int(lm[11].y * h)
        rx, ry = int(lm[12].x * w), int(lm[12].y * h)
        shoulder_px = np.sqrt((rx - lx)**2 + (ry - ly)**2)
        if shoulder_px < 10:
            return letterbox(image, TARGET_SIZE), False
        scale = target_px_per_cm / (shoulder_px / shoulder_width_cm)
        new_w, new_h = int(w * scale), int(h * scale)
        if new_w < 10 or new_h < 10:
            return letterbox(image, TARGET_SIZE), False
        self.n_success += 1
        return center_crop_or_pad(cv2.resize(image, (new_w, new_h)), TARGET_SIZE), True


def letterbox(image: np.ndarray, size: int = TARGET_SIZE) -> np.ndarray:
    h, w  = image.shape[:2]
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


def polygon_to_mask(points, img_w, img_h, size=TARGET_SIZE):
    pts  = np.array(points, dtype=np.float32).reshape(-1, 2).astype(np.int32)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], color=1)
    scale  = size / max(img_h, img_w)
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
        try:    return float(v) if v not in (None, "null", "") else float(default)
        except: return float(default)

    raw = {
        "body_height":        g("metadata.model.body_height",        161),
        "breast_size_female": g("metadata.model.breast_size_female",  87),
        "waist_size":         g("metadata.model.waist_size",          74),
        "hip_seize":          g("metadata.model.hip_seize",           93),
        "shoulders_width":    g("metadata.model.shoulders_width",     39),
        "arm_length":         g("metadata.model.arm_length",          54),
        "waist_height":       g("metadata.model.waist_height",        97),
        "back_length":        g("metadata.model.back_length",         42),
        "weight":             g("metadata.model.weight",              57),
        "gender":             gender,
    }
    return torch.tensor(
        [(v - BODY_STATS[k][0]) / BODY_STATS[k][1] for k, v in raw.items()],
        dtype=torch.float32,
    )  # [10] — 원본과 동일


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


# ──────────────────────────────────────────────────────────────
# 원본과 달라진 부분은 여기뿐입니다.
#
# 기존: json_dir(str), image_dir(str) → 단일 폴더
# 변경: json_dirs(List[str]), image_dirs(List[str]) → 여러 폴더
#       각 인덱스가 쌍으로 대응됩니다.
#       예) json_dirs[0]="/content/label_blouse"
#           image_dirs[0]="/content/image_blouse"
#           json_dirs[1]="/content/label_shirt"
#           image_dirs[1]="/content/image_shirt"
# ──────────────────────────────────────────────────────────────
class ClothingDataset(Dataset):
    def __init__(
        self,
        json_dirs:  List[str],   # ← 변경: 복수 폴더 리스트
        image_dirs: List[str],   # ← 변경: 복수 폴더 리스트
        view_type:  str   = "wear",
        use_mediapipe: bool = True,
        augment:    bool  = False,
        split:      str   = "train",
        val_ratio:  float = 0.1,
        test_ratio: float = 0.1,
        seed:       int   = 42,
    ):
        self.view_type = view_type
        self.augment   = augment
        self.scale_normalizer = ScaleNormalizer() if use_mediapipe else None

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        ]) if augment else None

        # (jpath, image_dir) 쌍으로 수집 — 이미지 폴더를 함께 기억
        self.samples = self._collect_samples(json_dirs, image_dirs)

        np.random.seed(seed)
        idx    = np.random.permutation(len(self.samples))
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

    def _collect_samples(
        self,
        json_dirs: List[str],
        image_dirs: List[str],
    ) -> List[Tuple[str, str]]:
        """(jpath, image_dir) 튜플 리스트 반환."""
        samples      = []
        skipped_type = skipped_null = skipped_img = 0

        for j_dir, i_dir in zip(json_dirs, image_dirs):
            found_jsons = glob.glob(os.path.join(j_dir, "*.json"))
            if not found_jsons:
                logger.warning(f"JSON 없음: {j_dir}")
                continue

            for jpath in found_jsons:
                try:
                    with open(jpath, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    json_type = data.get("metadata.model", {}).get(
                        "metadata.model.type", "").lower()
                    if self.view_type != "all" and json_type != self.view_type:
                        skipped_type += 1
                        continue

                    if extract_clothes_measurements(data) is None:
                        skipped_null += 1
                        continue

                    img_path = self._find_image(data, jpath, i_dir)
                    if not img_path:
                        skipped_img += 1
                        continue

                    samples.append((jpath, i_dir))

                except Exception:
                    continue

            logger.info(f"  {j_dir}: {sum(1 for s in samples if s[1]==i_dir)}개 수집")

        logger.info(f"전체 {len(samples)}개  "
                    f"(뷰타입제외={skipped_type}, 치수null={skipped_null}, 이미지없음={skipped_img})")
        return samples

    @staticmethod
    def _find_image(data: dict, jpath: str, i_dir: str) -> Optional[str]:
        rel = data.get("dataset", {}).get("dataset.image_path", "")
        if rel:
            cand = os.path.join(i_dir, os.path.basename(rel))
            if os.path.exists(cand):
                return cand
        cand = os.path.join(i_dir, os.path.basename(jpath).replace(".json", ".jpg"))
        return cand if os.path.exists(cand) else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        jpath, i_dir = self.samples[idx]

        with open(jpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_path = self._find_image(data, jpath, i_dir)
        image    = cv2.imread(img_path)
        if image is None:
            image = np.full((TARGET_SIZE, TARGET_SIZE, 3), 128, dtype=np.uint8)

        meta              = data.get("metadata.model", {})
        body_vec          = normalize_body(meta)   # [10] — 원본과 동일
        shoulder_width_cm = float(meta.get("metadata.model.shoulders_width", 39) or 39)

        if self.scale_normalizer is not None:
            image, _ = self.scale_normalizer(image, shoulder_width_cm)
        else:
            image = letterbox(image, TARGET_SIZE)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        if self.aug:
            pil = self.aug(pil)
        image_tensor = self.to_tensor(pil)

        annotations = data.get("annotation", [])
        ds    = data.get("dataset", {})
        img_w = ds.get("dataset.width",  TARGET_SIZE)
        img_h = ds.get("dataset.height", TARGET_SIZE)

        if annotations and annotations[0].get("annotation_point"):
            mask_np = polygon_to_mask(
                annotations[0]["annotation_point"], img_w, img_h, TARGET_SIZE)
        else:
            mask_np = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)

        mask    = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0)
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


def get_dataloaders(
    json_dirs:  List[str],   # ← 변경
    image_dirs: List[str],   # ← 변경
    view_type:  str   = "wear",
    use_mediapipe: bool = True,
    batch_size: int   = 16,
    num_workers: int  = 4,
    val_ratio:  float = 0.1,
    test_ratio: float = 0.1,
    seed:       int   = 42,
) -> Dict[str, DataLoader]:
    loaders = {}
    for split in ["train", "val", "test"]:
        ds = ClothingDataset(
            json_dirs=json_dirs,
            image_dirs=image_dirs,
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


def compute_dataset_stats(json_dirs: List[str], view_type: str = "wear"):
    body_vals    = {k: [] for k in BODY_STATS}
    clothes_vals = {k: [] for k in CLOTHES_KEYS}
    n_total = n_type = 0

    for j_dir in json_dirs:
        for jpath in glob.glob(os.path.join(j_dir, "*.json")):
            try:
                with open(jpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                n_total += 1
                meta      = data.get("metadata.model", {})
                json_type = meta.get("metadata.model.type", "").lower()
                if view_type != "all" and json_type != view_type:
                    n_type += 1
                    continue

                def g(key, default):
                    v = meta.get(key, default)
                    try:    return float(v) if v not in (None,"null","") else float(default)
                    except: return float(default)

                for k in list(BODY_STATS.keys())[:-1]:
                    body_vals[k].append(g(f"metadata.model.{k}", 0))
                body_vals["gender"].append(
                    0.0 if meta.get("metadata.model.gender") == "FEMALE" else 1.0)

                for key in CLOTHES_KEYS:
                    v = data.get(key) or data.get("metadata.clothes", {}).get(key)
                    if v not in (None, "null", ""):
                        try: clothes_vals[key].append(float(v))
                        except: pass
            except Exception:
                continue

    print(f"\n전체 json: {n_total}")
    print("\n신체 치수 분포")
    for k in BODY_STATS:
        if body_vals.get(k):
            arr = np.array(body_vals[k])
            print(f"  {k:30s}: mean={arr.mean():.1f}, std={arr.std():.1f}")
    print("\n옷 치수 분포")
    for k in CLOTHES_KEYS:
        if clothes_vals[k]:
            arr = np.array(clothes_vals[k])
            print(f"  {k.split('.')[-1]:20s}: mean={arr.mean():.1f}, std={arr.std():.1f}  (n={len(arr)})")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--json_dirs",  nargs="+", required=True)
    p.add_argument("--view_type",  type=str, default="wear")
    args = p.parse_args()
    compute_dataset_stats(args.json_dirs, args.view_type)