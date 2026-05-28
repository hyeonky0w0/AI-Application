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

# 이상치 필터
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


def main(json_dir, view_type):
    body_vals   = {k: [] for k in BODY_KEYS}
    clothes_vals = {k: [] for k in CLOTHES_KEYS}
    n_total = n_type = n_clothes_outlier = n_body_outlier = 0

    for jpath in glob.glob(os.path.join(json_dir, "*.json")):
        try:
            with open(jpath, encoding="utf-8") as f:
                data = json.load(f)

            n_total += 1
            meta = data.get("metadata.model", {})

            # view_type 필터
            json_type = meta.get("metadata.model.type", "").lower()
            if view_type != "all" and json_type != view_type:
                n_type += 1
                continue

            def g(key, default=0):
                v = meta.get(f"metadata.model.{key}", default)
                try: return float(v) if v not in (None, "null", "") else float(default)
                except: return float(default)

            body_row = {k: g(k) for k in BODY_KEYS}
            body_ok = all(
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
            clothes_row = {}
            clothes_ok = True
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
    # print(f"\n전체 JSON:       {n_total:,}개")
    # print(f"타입 제외:       {n_type:,}개")
    # print(f"착장 샘플:       {worn:,}개")
    # print(f"신체 이상치:     {n_body_outlier:,}개 제거")
    # print(f"옷치수 이상치:   {n_clothes_outlier:,}개 제거")
    # clean_n = len(clothes_vals["metadata.top.chest_size"])
    # print(f"최종 유효 샘플:  {clean_n:,}")

    print("\n신체 치수 분포 (이상치 제거 후)")
    body_stats = {}
    for k in BODY_KEYS:
        if body_vals[k]:
            arr = np.array(body_vals[k])
            body_stats[k] = (round(arr.mean(), 1), round(arr.std(), 1))
            print(f"  {k:25s}: mean={arr.mean():.1f}, std={arr.std():.1f}, "
                  f"min={arr.min():.0f}, max={arr.max():.0f}")

    print("\n옷 치수 분포 (이상치 제거 후)")
    clothes_stats = {}
    for k in CLOTHES_KEYS:
        name = k.split(".")[-1]
        if clothes_vals[k]:
            arr = np.array(clothes_vals[k])
            clothes_stats[k] = (round(arr.mean(), 1), round(arr.std(), 1))
            print(f"  {name:20s}: mean={arr.mean():.1f}, std={arr.std():.1f}, "
                  f"min={arr.min():.0f}, max={arr.max():.0f}  (n={len(arr)})")
        else:
            print(f"  {name:20s}: 데이터 없음")

    print("\n")
    #BODY_STATS를 아래 값으로 교체
    print("BODY_STATS = {")
    for k in BODY_KEYS:
        if k in body_stats:
            m, s = body_stats[k]
            print(f'    "{k}": ({m}, {s}),')
    print('    "gender": (0.5, 0.5),')
    print("}")

    print("\n")
    # CLOTHES_STATS를 아래 값으로 교체
    print("CLOTHES_STATS = {")
    for k in CLOTHES_KEYS:
        if k in clothes_stats:
            m, s = clothes_stats[k]
            print(f'    "{k}": ({m}, {s}),')
    print("}")

    print("\n")
    for k, (lo, hi) in VALID_CLOTHES.items():
        name = k.split(".")[-1]
        print(f"  {name:20s}: {lo} ~ {hi} cm")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--json_dir",   type=str, required=True)
    p.add_argument("--view_type",  type=str, default="wear")
    args = p.parse_args()
    main(args.json_dir, args.view_type)