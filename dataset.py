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

# ──────────────────────────────────────────────────────────────
# ② 카테고리별 CLOTHES_STATS 분리
#    stats_analysis.py를 카테고리별로 실행한 뒤 여기에 붙여넣으세요.
#    현재값은 추정치입니다.
# ──────────────────────────────────────────────────────────────
CLOTHES_STATS_BY_CATEGORY: Dict[str, Dict[str, Tuple]] = {
    "blouse": {
        "metadata.top.shoulder_width": (40.5,  5.7),
        "metadata.top.front_length":   (61.4,  7.9),
        "metadata.top.chest_size":     (96.2, 11.5),
        "metadata.top.waist_size":     (98.9, 13.4),
        "metadata.top.sleeve_length":  (35.0, 10.2),
    },
    "shirt": {
        "metadata.top.shoulder_width": (41.0,  5.5),
        "metadata.top.front_length":   (72.0,  8.5),
        "metadata.top.chest_size":     (100.0, 12.0),
        "metadata.top.waist_size":     (100.0, 13.0),
        "metadata.top.sleeve_length":  (58.0,  8.5),
    },
    "coat": {
        "metadata.top.shoulder_width": (42.0,  5.0),
        "metadata.top.front_length":   (90.0, 15.0),
        "metadata.top.chest_size":     (105.0, 13.0),
        "metadata.top.waist_size":     (105.0, 14.0),
        "metadata.top.sleeve_length":  (60.0,  7.0),
    },
}

CLOTHES_STATS_DEFAULT = {
    "metadata.top.shoulder_width": (40.5,  5.7),
    "metadata.top.front_length":   (61.4,  7.9),
    "metadata.top.chest_size":     (96.2, 11.5),
    "metadata.top.waist_size":     (98.9, 13.4),
    "metadata.top.sleeve_length":  (42.2, 15.1),
}

CLOTHES_STATS = CLOTHES_STATS_DEFAULT   # 하위 호환용
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

# ──────────────────────────────────────────────────────────────
# ① 카테고리 임베딩
#    body_vec 뒤에 one-hot을 붙임
#    body_dim = 10(신체) + 3(카테고리) = 13
# ──────────────────────────────────────────────────────────────
CATEGORY_MAP: Dict[str, int] = {
    "blouse": 0,
    "shirt":  1,
    "coat":   2,
}
CATEGORY_DIM = len(CATEGORY_MAP)   # 3
BODY_DIM     = 10 + CATEGORY_DIM  # 13

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

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_clothes_stats(category: str) -> Dict[str, Tuple]:
    return CLOTHES_STATS_BY_CATEGORY.get(category, CLOTHES_STATS_DEFAULT)


def denormalize_clothes(tensor: torch.Tensor, category: str = "blouse") -> torch.Tensor:
    stats = get_clothes_stats(category)
    means = torch.tensor([stats[k][0] for k in CLOTHES_KEYS],
                         dtype=torch.float32, device=tensor.device)
    stds  = torch.tensor([stats[k][1] for k in CLOTHES_KEYS],
                         dtype=torch.float32, device=tensor.device)
    return tensor * stds + means


def denormalize_clothes_batch(tensor: torch.Tensor,
                               categories: List[str]) -> torch.Tensor:
    """배치 내 카테고리가 섞여 있을 때 각각 역변환."""
    out = torch.zeros_like(tensor)
    for i, cat in enumerate(categories):
        out[i] = denormalize_clothes(tensor[i].unsqueeze(0), cat).squeeze(0)
    return out


# ──────────────────────────────────────────────────────────────
# 경로 헬퍼
# ──────────────────────────────────────────────────────────────
def _resolve_dir(dir_input, category: str) -> str:
    """
    dir_input이 Dict이면 category 키로 조회,
    str이면 그대로 반환.

    사용 예:
        단일:  json_dir = "/content/label_blouse"
        복수:  json_dir = {"blouse": "/content/label_blouse",
                           "shirt":  "/content/label_shirt"}
    """
    if isinstance(dir_input, dict):
        if category not in dir_input:
            raise KeyError(f"카테고리 '{category}'에 대한 경로가 없습니다. "
                           f"전달된 키: {list(dir_input.keys())}")
        return dir_input[category]
    return dir_input   # str


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
        return self.n_success / self.n_total if self.n_total else 0.0

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
        rescaled = cv2.resize(image, (new_w, new_h))
        self.n_success += 1
        return center_crop_or_pad(rescaled, TARGET_SIZE), True


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
        return image[(h-size)//2:(h-size)//2+size, (w-size)//2:(w-size)//2+size]
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


def normalize_body(meta: dict, category: str = "blouse") -> torch.Tensor:
    """신체 10차원 + 카테고리 one-hot 3차원 = 13차원."""
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
    body_part = [(v - BODY_STATS[k][0]) / BODY_STATS[k][1] for k, v in raw.items()]

    cat_onehot          = [0.0] * CATEGORY_DIM
    cat_onehot[CATEGORY_MAP.get(category, 0)] = 1.0

    return torch.tensor(body_part + cat_onehot, dtype=torch.float32)  # [13]


def extract_clothes_measurements(data: dict,
                                  category: str = "blouse") -> Optional[torch.Tensor]:
    stats        = get_clothes_stats(category)
    clothes_meta = data.get("metadata.clothes", {})
    values       = []
    for key in CLOTHES_KEYS:
        val = clothes_meta.get(key) or data.get(key)
        if val in (None, "null", ""):
            return None
        try:
            v = float(val)
            lo, hi = CLOTHES_VALID_RANGE[key]
            if not (lo <= v <= hi):
                return None
            mean, std = stats[key]
            values.append((v - mean) / std)
        except (ValueError, TypeError):
            return None
    return torch.tensor(values, dtype=torch.float32)


# ──────────────────────────────────────────────────────────────
# Dataset
#   json_dir  : str  또는  Dict[category, path]
#   image_dir : str  또는  Dict[category, path]
#
#   str  → 모든 카테고리가 같은 폴더 (기존 동작 유지)
#   Dict → 카테고리마다 다른 폴더   (신규 멀티카테고리 동작)
# ──────────────────────────────────────────────────────────────
class ClothingDataset(Dataset):
    def __init__(
        self,
        json_dir,           # str | Dict[str, str]
        image_dir,          # str | Dict[str, str]
        categories: List[str] = ["blouse"],
        view_type:  str = "wear",
        use_mediapipe: bool = True,
        augment:    bool = False,
        split:      str  = "train",
        val_ratio:  float = 0.1,
        test_ratio: float = 0.1,
        seed:       int   = 42,
    ):
        self.json_dir  = json_dir
        self.image_dir = image_dir
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

        # samples: List[Tuple[jpath, category, image_dir_for_this_cat]]
        self.samples = self._collect_samples(categories)

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
        logger.info(f"[{split}] {len(self.samples)}개 샘플 준비 완료")

    def _collect_samples(self, categories):
        """(jpath, category, img_dir) 튜플 리스트 반환."""
        samples      = []
        skip_type    = skip_null = skip_img = 0

        for cat in categories:
            j_dir = _resolve_dir(self.json_dir,  cat)
            i_dir = _resolve_dir(self.image_dir, cat)

            # JSON 파일 탐색 — 루트, label_{cat}/, {cat}/ 순서
            search_dirs = [
                j_dir,
                os.path.join(j_dir, f"label_{cat}"),
                os.path.join(j_dir, cat),
            ]
            found_jsons = []
            for d in search_dirs:
                found_jsons.extend(glob.glob(os.path.join(d, "*.json")))

            if not found_jsons:
                logger.warning(f"[{cat}] JSON 없음: {j_dir}")
                continue

            for jpath in found_jsons:
                try:
                    with open(jpath, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    vt = data.get("metadata.model", {}).get(
                        "metadata.model.type", "").lower()
                    if self.view_type != "all" and vt != self.view_type:
                        skip_type += 1
                        continue

                    if extract_clothes_measurements(data, category=cat) is None:
                        skip_null += 1
                        continue

                    img_path = self._find_image(data, jpath, i_dir)
                    if not img_path:
                        skip_img += 1
                        continue

                    samples.append((jpath, cat, i_dir))

                except Exception:
                    continue

            logger.info(f"[{cat}] json_dir={j_dir} | image_dir={i_dir} "
                        f"| 수집={sum(1 for s in samples if s[1]==cat)}개")

        logger.info(f"전체 수집: {len(samples)}개  "
                    f"(뷰타입제외={skip_type}, 치수null={skip_null}, 이미지없음={skip_img})")
        return samples

    @staticmethod
    def _find_image(data: dict, jpath: str, i_dir: str) -> Optional[str]:
        """JSON의 image_path 필드 또는 json 파일명으로 이미지 경로를 찾는다."""
        rel = data.get("dataset", {}).get("dataset.image_path", "")
        if rel:
            fname = os.path.basename(rel)
            cand  = os.path.join(i_dir, fname)
            if os.path.exists(cand):
                return cand

        fname = os.path.basename(jpath).replace(".json", ".jpg")
        cand  = os.path.join(i_dir, fname)
        if os.path.exists(cand):
            return cand
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        jpath, category, i_dir = self.samples[idx]

        with open(jpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_path = self._find_image(data, jpath, i_dir)
        image    = cv2.imread(img_path)
        if image is None:
            image = np.full((TARGET_SIZE, TARGET_SIZE, 3), 128, dtype=np.uint8)

        meta     = data.get("metadata.model", {})
        body_vec = normalize_body(meta, category=category)  # [13]

        shoulder_cm = float(meta.get("metadata.model.shoulders_width", 39) or 39)
        if self.scale_normalizer is not None:
            image, _ = self.scale_normalizer(image, shoulder_cm)
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
        clothes = extract_clothes_measurements(data, category=category)
        if clothes is None:
            clothes = torch.zeros(len(CLOTHES_KEYS), dtype=torch.float32)

        return {
            "image":      image_tensor,
            "body_vec":   body_vec,      # [13]
            "mask":       mask,
            "clothes":    clothes,
            "path":       jpath,
            "category":   category,
            "model_type": meta.get("metadata.model.type", ""),
        }


# ──────────────────────────────────────────────────────────────
# get_dataloaders
#   json_dir / image_dir 모두 str 또는 Dict[str, str] 허용
# ──────────────────────────────────────────────────────────────
def get_dataloaders(
    json_dir,           # str | Dict[str, str]
    image_dir,          # str | Dict[str, str]
    categories: List[str] = ["blouse"],
    view_type:  str = "wear",
    use_mediapipe: bool = True,
    batch_size: int = 16,
    num_workers: int = 4,
    val_ratio:  float = 0.1,
    test_ratio: float = 0.1,
    seed:       int   = 42,
) -> Dict[str, DataLoader]:
    loaders = {}
    for split in ["train", "val", "test"]:
        ds = ClothingDataset(
            json_dir=json_dir,
            image_dir=image_dir,
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


def compute_dataset_stats(json_dir, categories, view_type="wear"):
    body_vals    = {k: [] for k in BODY_STATS}
    clothes_vals = {k: [] for k in CLOTHES_KEYS}
    n_total = n_type = 0

    for cat in categories:
        j_dir = _resolve_dir(json_dir, cat)
        search_dirs = [j_dir,
                       os.path.join(j_dir, f"label_{cat}"),
                       os.path.join(j_dir, cat)]
        found = []
        for d in search_dirs:
            found.extend(glob.glob(os.path.join(d, "*.json")))

        for jpath in found:
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

                for k in list(BODY_STATS.keys())[:-1]:   # gender 제외
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
    p.add_argument("--json_dir",   type=str, default=None)
    p.add_argument("--image_dir",  type=str, default=None)
    p.add_argument("--categories", nargs="+", default=["blouse"])
    p.add_argument("--view_type",  type=str, default="wear")
    args = p.parse_args()
    if args.json_dir:
        compute_dataset_stats(args.json_dir, args.categories, args.view_type)