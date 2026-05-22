import os
import glob
import json
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET_SIZE = 224

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


def build_detector():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    import urllib.request

    model_path = "/tmp/pose_landmarker.task"
    if not os.path.exists(model_path):
        logger.info("Pose Landmarker 모델 다운로드 중...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/"
            "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
            model_path,
        )
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
    )
    return vision.PoseLandmarker.create_from_options(options)


def process_one(args_tuple):
    import mediapipe as mp

    jpath, image_path, shoulder_width_cm, cache_path, detector = args_tuple

    if os.path.exists(cache_path):
        return jpath, True, None   

    image = cv2.imread(image_path)
    if image is None:
        image = np.full((TARGET_SIZE, TARGET_SIZE, 3), 128, dtype=np.uint8)
        np.save(cache_path, image.astype(np.uint8))
        return jpath, True, False

    used_mediapipe = False
    result_image = None

    if detector is not None and shoulder_width_cm > 0:
        try:
            h, w = image.shape[:2]
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)

            if result.pose_landmarks:
                lms = result.pose_landmarks[0]
                lx, ly = int(lms[11].x * w), int(lms[11].y * h)
                rx, ry = int(lms[12].x * w), int(lms[12].y * h)
                shoulder_px = np.sqrt((rx - lx) ** 2 + (ry - ly) ** 2)

                if shoulder_px >= 10:
                    target_px_per_cm = 5.0
                    scale = target_px_per_cm / (shoulder_px / shoulder_width_cm)
                    new_w, new_h = int(w * scale), int(h * scale)
                    if new_w >= 10 and new_h >= 10:
                        rescaled = cv2.resize(image, (new_w, new_h))
                        result_image = center_crop_or_pad(rescaled, TARGET_SIZE)
                        used_mediapipe = True
        except Exception:
            pass

    if result_image is None:
        result_image = letterbox(image, TARGET_SIZE)

    result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    np.save(cache_path, result_rgb.astype(np.uint8))
    return jpath, True, used_mediapipe


def collect_jobs(json_dir, image_dir, categories, view_type, cache_dir):
    jobs = []
    skipped = 0

    for cat in categories:
        search_dirs = [
            json_dir,
            os.path.join(json_dir, f"label_{cat}"),
            os.path.join(json_dir, cat),
        ]
        found_jsons = []
        for d in search_dirs:
            found_jsons.extend(glob.glob(os.path.join(d, "*.json")))

        for jpath in found_jsons:
            try:
                with open(jpath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                json_type = data.get("metadata.model", {}).get(
                    "metadata.model.type", ""
                ).lower()
                if view_type != "all" and json_type != view_type:
                    skipped += 1
                    continue

                rel = data.get("dataset", {}).get("dataset.image_path", "")
                img_path = None
                if rel:
                    candidate = os.path.join(image_dir, os.path.basename(rel))
                    if os.path.exists(candidate):
                        img_path = candidate
                if img_path is None:
                    fname = os.path.basename(jpath).replace(".json", ".jpg")
                    candidate = os.path.join(image_dir, fname)
                    if os.path.exists(candidate):
                        img_path = candidate
                if img_path is None:
                    skipped += 1
                    continue

                meta = data.get("metadata.model", {})
                sw = float(meta.get("metadata.model.shoulders_width", 39) or 39)

                cache_name = os.path.splitext(os.path.basename(jpath))[0] + ".npy"
                cache_path = os.path.join(cache_dir, cache_name)

                jobs.append((jpath, img_path, sw, cache_path))

            except Exception:
                continue

    logger.info(f"전처리 대상: {len(jobs)}장  /  스킵: {skipped}장")
    return jobs


def run(args):
    os.makedirs(args.cache_dir, exist_ok=True)

    jobs = collect_jobs(
        args.json_dir, args.image_dir,
        args.categories, args.view_type,
        args.cache_dir,
    )
    if not jobs:
        logger.warning("처리할 파일이 없습니다.")
        return

    try:
        detector = build_detector()
        logger.info("MediaPipe 초기화 완료")
    except Exception as e:
        detector = None
        logger.warning(f"MediaPipe 없이 letterbox만 사용: {e}")

    n_mediapipe = 0
    n_letterbox = 0
    n_skip      = 0

    pbar = tqdm(total=len(jobs), desc="전처리 중")
    for job in jobs:
        jpath, img_path, sw, cache_path = job
        _, ok, used_mp = process_one((jpath, img_path, sw, cache_path, detector))
        if used_mp is None:
            n_skip += 1
        elif used_mp:
            n_mediapipe += 1
        else:
            n_letterbox += 1
        pbar.update(1)
    pbar.close()

    total = n_mediapipe + n_letterbox
    logger.info(f"\n완료!")
    logger.info(f"  MediaPipe 성공: {n_mediapipe}/{total} "
                f"({n_mediapipe/max(total,1)*100:.1f}%)")
    logger.info(f"  letterbox fallback: {n_letterbox}/{total}")
    logger.info(f"  이미 캐시 있어서 스킵: {n_skip}")
    logger.info(f"  캐시 저장 위치: {args.cache_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--json_dir",   type=str, required=True)
    p.add_argument("--image_dir",  type=str, required=True)
    p.add_argument("--cache_dir",  type=str, required=True,
                   help="전처리된 .npy 파일 저장 디렉토리")
    p.add_argument("--categories", nargs="+", default=["blouse"])
    p.add_argument("--view_type",  type=str, default="front",
                   choices=["front", "wear", "all"])
    args = p.parse_args()
    run(args)