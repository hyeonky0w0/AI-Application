"""
stats_analysis.py — 카테고리별 신체/의류 치수 분포 분석

사용법:
    # 단일 카테고리
    python stats_analysis.py --json_dir /content/label_blouse --view_type wear

    # 복수 카테고리 (각 디렉토리를 쉼표 구분으로)
    python stats_analysis.py \
        --json_dirs /content/label_blouse,/content/label_shirt \
        --categories blouse,shirt \
        --view_type wear

출력 결과를 dataset.py 의 CLOTHES_STATS_BY_CATEGORY 에 붙여넣으세요.
"""

import sys, json, glob, os, argparse
import numpy as np

CLOTHES_KEYS = [
    "metadata.top.shoulder_width",
    "metadata.top.front_length",
    "metadata.top.chest_size",
    "metadata.top.waist_size",
    "metadata.top.sleeve_length",
]

BODY_KEYS = [
    "body_height", "breast_size_female", "waist_size", "hip_seize",
    "shoulders_width", "arm_length", "waist_height", "back_length", "weight",
]

VALID_CLOTHES = {
    "metadata.top.shoulder_width": (30,  60),
    "metadata.top.front_length":   (40, 120),
    "metadata.top.chest_size":     (70, 160),
    "metadata.top.waist_size":     (60, 160),
    "metadata.top.sleeve_length":  (10,  80),
}
VALID_BODY = {
    "body_height":        (140, 195),
    "breast_size_female": (60,  120),
    "waist_size":         (50,  110),
    "hip_seize":          (70,  130),
    "shoulders_width":    (30,   55),
    "arm_length":         (40,   70),
    "waist_height":       (70,  120),
    "back_length":        (30,   55),
    "weight":             (35,  110),
}


def analyze_one(json_dir: str, view_type: str, category: str = "unknown"):
    """단일 디렉토리에 대한 분포 분석."""
    body_vals    = {k: [] for k in BODY_KEYS}
    clothes_vals = {k: [] for k in CLOTHES_KEYS}
    n_total = n_type = n_clothes_outlier = n_body_outlier = 0

    for jpath in glob.glob(os.path.join(json_dir, "*.json")):
        try:
            with open(jpath, encoding="utf-8") as f:
                data = json.load(f)

            n_total += 1
            meta = data.get("metadata.model", {})

            json_type = meta.get("metadata.model.type", "").lower()
            if view_type != "all" and json_type != view_type:
                n_type += 1
                continue

            def g(key, default=0):
                v = meta.get(f"metadata.model.{key}", default)
                try:
                    return float(v) if v not in (None, "null", "") else float(default)
                except:
                    return float(default)

            body_row = {k: g(k) for k in BODY_KEYS}
            body_ok  = all(
                VALID_BODY[k][0] <= v <= VALID_BODY[k][1]
                for k, v in body_row.items()
                if k in VALID_BODY
            )
            if body_ok:
                for k, v in body_row.items():
                    body_vals[k].append(v)
            else:
                n_body_outlier += 1

            clothes_meta = data.get("metadata.clothes", {})
            clothes_ok   = True
            clothes_row  = {}
            for k in CLOTHES_KEYS:
                v = clothes_meta.get(k) or data.get(k)
                if v in (None, "null", ""):
                    clothes_ok = False; break
                fv = float(v)
                lo, hi = VALID_CLOTHES[k]
                if not (lo <= fv <= hi):
                    clothes_ok = False; break
                clothes_row[k] = fv

            if clothes_ok:
                for k, v in clothes_row.items():
                    clothes_vals[k].append(v)
            else:
                n_clothes_outlier += 1

        except Exception:
            continue

    worn = n_total - n_type
    print(f"\n[카테고리: {category}]")
    print(f"  전체 JSON:     {n_total:,}개")
    print(f"  타입 제외:     {n_type:,}개  (착장 샘플={worn:,}개)")
    print(f"  신체 이상치:   {n_body_outlier:,}개")
    print(f"  옷 이상치:     {n_clothes_outlier:,}개")
    clean_n = len(clothes_vals[CLOTHES_KEYS[0]])
    print(f"  최종 유효:     {clean_n:,}개")

    print(f"\n  신체 치수 분포")
    body_stats = {}
    for k in BODY_KEYS:
        if body_vals[k]:
            arr = np.array(body_vals[k])
            body_stats[k] = (round(arr.mean(), 1), round(arr.std(), 1))
            print(f"    {k:25s}: mean={arr.mean():.1f}, std={arr.std():.1f}, "
                  f"min={arr.min():.0f}, max={arr.max():.0f}")

    print(f"\n  옷 치수 분포")
    clothes_stats = {}
    for k in CLOTHES_KEYS:
        name = k.split(".")[-1]
        if clothes_vals[k]:
            arr = np.array(clothes_vals[k])
            clothes_stats[k] = (round(arr.mean(), 1), round(arr.std(), 1))
            print(f"    {name:20s}: mean={arr.mean():.1f}, std={arr.std():.1f}, "
                  f"min={arr.min():.0f}, max={arr.max():.0f}  (n={len(arr)})")
        else:
            print(f"    {name:20s}: 데이터 없음")

    return body_stats, clothes_stats


def print_code_snippet(all_clothes_stats: dict, all_body_stats: dict):
    """dataset.py 에 붙여넣을 코드 조각 출력."""
    print("\n\n" + "="*60)
    print("# ▼ dataset.py 의 CLOTHES_STATS_BY_CATEGORY 를 아래로 교체")
    print("="*60)
    print("CLOTHES_STATS_BY_CATEGORY: Dict[str, Dict[str, Tuple]] = {")
    for cat, stats in all_clothes_stats.items():
        print(f'    "{cat}": {{')
        for k in CLOTHES_KEYS:
            if k in stats:
                m, s = stats[k]
                print(f'        "{k}": ({m}, {s}),')
        print("    },")
    print("}")

    # body_stats는 공유하므로 첫 번째 카테고리 기준으로 출력
    first_body = next(iter(all_body_stats.values()), {})
    if first_body:
        print("\n\n" + "="*60)
        print("# ▼ dataset.py 의 BODY_STATS 를 아래로 교체")
        print("="*60)
        print("BODY_STATS = {")
        for k in BODY_KEYS:
            if k in first_body:
                m, s = first_body[k]
                print(f'    "{k}": ({m}, {s}),')
        print('    "gender": (0.5, 0.5),')
        print("}")


def main():
    p = argparse.ArgumentParser()
    # 단일 디렉토리 (기존 방식)
    p.add_argument("--json_dir",    type=str, default=None,
                   help="JSON 디렉토리 (단일 카테고리)")
    p.add_argument("--view_type",   type=str, default="wear")

    # 복수 카테고리 (쉼표 구분)
    p.add_argument("--json_dirs",   type=str, default=None,
                   help="JSON 디렉토리들, 쉼표 구분 (예: /data/label_blouse,/data/label_shirt)")
    p.add_argument("--categories",  type=str, default=None,
                   help="카테고리 이름들, 쉼표 구분 (예: blouse,shirt)")
    args = p.parse_args()

    all_clothes_stats: dict = {}
    all_body_stats:    dict = {}

    if args.json_dirs and args.categories:
        dirs = [d.strip() for d in args.json_dirs.split(",")]
        cats = [c.strip() for c in args.categories.split(",")]
        if len(dirs) != len(cats):
            print("ERROR: --json_dirs 와 --categories 의 수가 일치해야 합니다.")
            sys.exit(1)
        for d, cat in zip(dirs, cats):
            bs, cs = analyze_one(d, args.view_type, category=cat)
            all_clothes_stats[cat] = cs
            all_body_stats[cat]    = bs

    elif args.json_dir:
        cat = "blouse"
        bs, cs = analyze_one(args.json_dir, args.view_type, category=cat)
        all_clothes_stats[cat] = cs
        all_body_stats[cat]    = bs
    else:
        p.print_help()
        sys.exit(0)

    print_code_snippet(all_clothes_stats, all_body_stats)


if __name__ == "__main__":
    main()