/content/SiCoimport torchimport cv2import globimport numpy as npfrom PIL import Imageimport sysimport osimport requestsimport ioimport jsonfrom segment_anything_hq import sam_model_registry, SamPredictor

DILATE_ITERATIONS = 3

── AI Hub 블라우스 데이터셋 통계 ──────────

STATS = {'shoulder': {'mean': 40.31, 'std': 6.73},'chest': {'mean': 95.35, 'std': 12.70},'waist': {'mean': 98.16, 'std': 14.57},'sleeve': {'mean': 39.31, 'std': 17.68},'length': {'mean': 60.95, 'std': 8.28},}

════════════════════════════════════════════════════════

FitNet 연동

════════════════════════════════════════════════════════

def fitnet_predict_from_image(garment_image_path, model_path="./weights/fitnet.pth"):"""[모드 1] 옷 이미지 → 5개 치수 예측FitNet 모델이 없으면 더미값 반환

Returns:
    dict: {shoulder, length, chest, waist, sleeve} (cm)
"""
if not os.path.exists(model_path):
    print(f"[WARNING] FitNet 모델 없음 ({model_path}). 더미값 사용.")
    return {
        'shoulder': 38.0,
        'length':   62.0,
        'chest':    90.0,
        'waist':    76.0,
        'sleeve':   58.0,
    }

# ── 실제 FitNet 연결 시 아래 주석 해제 ──────────────────────────
# from model import FitNet  # 본인 모델 클래스
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = FitNet()
# model.load_state_dict(torch.load(model_path, map_location=device))
# model.eval()
#
# img = preprocess_garment(garment_image_path)  # 전처리
# with torch.no_grad():
#     preds = model(img)  # [어깨, 총장, 가슴, 허리, 소매] (정규화값)
#
# # z-score 역변환 → cm
# keys = ['shoulder', 'length', 'chest', 'waist', 'sleeve']
# result = {}
# for i, k in enumerate(keys):
#     result[k] = preds[i].item() * STATS[k]['std'] + STATS[k]['mean']
# return result
# ────────────────────────────────────────────────────────────────

return {'shoulder': 38.0, 'length': 62.0, 'chest': 90.0, 'waist': 76.0, 'sleeve': 58.0}

def fitnet_predict_from_body(body_measurements, model_path="./weights/fitnet.pth"):"""[모드 2] 신체 치수(10개) → 옷 치수(5개) 예측직접 입력 테스트용 / FitNet 연결 전 단계

Args:
    body_measurements: dict {
        body_height, breast_size, waist_size, hip_size,
        shoulders_width, arm_length, waist_height,
        back_length, weight, gender(0=여/1=남)
    }

Returns:
    dict: {shoulder, length, chest, waist, sleeve} (cm)
"""
if not os.path.exists(model_path):
    print(f"[WARNING] FitNet 모델 없음. 신체치수 기반 추정값 사용.")
    # 신체치수로 간단 추정 (FitNet 없을 때 임시)
    chest    = body_measurements.get('breast_size', 90.0)
    waist    = body_measurements.get('waist_size', 76.0)
    shoulder = body_measurements.get('shoulders_width', 38.0)
    sleeve   = body_measurements.get('arm_length', 58.0)
    length   = body_measurements.get('back_length', 62.0)
    return {
        'shoulder': shoulder,
        'length':   length,
        'chest':    chest,
        'waist':    waist,
        'sleeve':   sleeve,
    }

# ── 실제 FitNet 연결 시 아래 주석 해제 ──────────────────────────
# from model import FitNet
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = FitNet()
# model.load_state_dict(torch.load(model_path, map_location=device))
# model.eval()
#
# keys_in = ['body_height','breast_size','waist_size','hip_size',
#            'shoulders_width','arm_length','waist_height',
#            'back_length','weight','gender']
# x = torch.tensor([[body_measurements[k] for k in keys_in]], dtype=torch.float32)
# with torch.no_grad():
#     preds = model(x)
# keys_out = ['shoulder','length','chest','waist','sleeve']
# return {k: preds[0][i].item() * STATS[k]['std'] + STATS[k]['mean']
#         for i, k in enumerate(keys_out)}
# ────────────────────────────────────────────────────────────────

chest    = body_measurements.get('breast_size', 90.0)
waist    = body_measurements.get('waist_size', 76.0)
shoulder = body_measurements.get('shoulders_width', 38.0)
sleeve   = body_measurements.get('arm_length', 58.0)
length   = body_measurements.get('back_length', 62.0)
return {'shoulder': shoulder, 'length': length,
        'chest': chest, 'waist': waist, 'sleeve': sleeve}

════════════════════════════════════════════════════════

부위별 z-score 계산

════════════════════════════════════════════════════════

def compute_relative_z(garment_dims, user_body):"""garment_dims: FitNet이 예측한 옷 치수 dictuser_body:    사용자 신체 치수 dict

Returns:
    dict: {shoulder, chest, waist, sleeve, length} z-score
"""
z = {}
# 어깨: 옷 어깨너비 vs 사용자 어깨너비
z['shoulder'] = (garment_dims['shoulder'] - user_body.get('shoulder', 38.0)) / STATS['shoulder']['std']
# 가슴: 옷 가슴둘레 vs 사용자 가슴둘레
z['chest']    = (garment_dims['chest']    - user_body.get('chest', 90.0))     / STATS['chest']['std']
# 허리: 옷 허리둘레 vs 사용자 허리둘레
z['waist']    = (garment_dims['waist']    - user_body.get('waist', 76.0))      / STATS['waist']['std']
# 소매: 옷 소매길이 vs 사용자 팔길이
z['sleeve']   = (garment_dims['sleeve']   - user_body.get('arm', 58.0))      / STATS['sleeve']['std']
# 총장: 옷 총장 vs 사용자 등길이
z['length']   = (garment_dims['length']   - user_body.get('back', 62.0))     / STATS['length']['std']

print("[Z-Score] 부위별 relative_z:")
for k, v in z.items():
    fit = "헐렁" if v > 0.5 else ("타이트" if v < -0.5 else "적합")
    print(f"  {k:10s}: {v:+.2f} ({fit})")
return z

════════════════════════════════════════════════════════

부위별 마스크 조절

════════════════════════════════════════════════════════

def scale_mask_bbox(mask, scale_x=1.0, scale_y=1.0):"""mask의 bounding box를 기준으로 가로/세로 비율 조정.전체 이미지를 resize하는 게 아니라, 해당 부위 mask 영역만 scale 조절."""import cv2import numpy as np

mask = (mask > 0).astype(np.uint8) * 255
h, w = mask.shape[:2]

ys, xs = np.where(mask > 0)
if len(xs) == 0 or len(ys) == 0:
    return mask

x1, x2 = xs.min(), xs.max()
y1, y2 = ys.min(), ys.max()

crop = mask[y1:y2+1, x1:x2+1]
ch, cw = crop.shape[:2]

new_w = max(1, int(cw * scale_x))
new_h = max(1, int(ch * scale_y))

resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

canvas = np.zeros_like(mask)

cx = (x1 + x2) // 2
cy = (y1 + y2) // 2

nx1 = cx - new_w // 2
ny1 = cy - new_h // 2
nx2 = nx1 + new_w
ny2 = ny1 + new_h

src_x1 = max(0, -nx1)
src_y1 = max(0, -ny1)
dst_x1 = max(0, nx1)
dst_y1 = max(0, ny1)

dst_x2 = min(w, nx2)
dst_y2 = min(h, ny2)

src_x2 = src_x1 + (dst_x2 - dst_x1)
src_y2 = src_y1 + (dst_y2 - dst_y1)

if dst_x2 > dst_x1 and dst_y2 > dst_y1:
    canvas[dst_y1:dst_y2, dst_x1:dst_x2] = resized[src_y1:src_y2, src_x1:src_x2]

return canvas

def z_to_scale(z, alpha=0.08, min_scale=0.75, max_scale=1.25):"""z-score를 mask scale 비율로 변환.z=0이면 1.0 유지.z가 양수면 확대, 음수면 축소."""scale = 1.0 + alpha * zreturn max(min_scale, min(max_scale, scale))

def adjust_mask_by_region(masks_dict, z_scores):"""DensePose 기반 부위별 mask를 z-score에 따라 비율 조절.torso      ← chest + waistupper_arm  ← shoulderlower_arm  ← sleeve"""import cv2import numpy as np

result_masks = []

if 'torso' in masks_dict:
    torso = masks_dict['torso'].copy()
    z_body = (z_scores['chest'] + z_scores['waist']) / 2.0

    # torso는 가로 중심으로만 조절
    scale_x = z_to_scale(z_body, alpha=0.10, min_scale=0.88, max_scale=1.12)
    scale_y = 1.0

    print(f"[mask scale] torso: z_body={z_body:+.2f}, scale_x={scale_x:.2f}, scale_y={scale_y:.2f}")

    torso = scale_mask_bbox(torso, scale_x=scale_x, scale_y=scale_y)

    # 상의가 하의/골반 영역까지 내려가는 것 방지
    ys, xs = np.where(torso > 0)
    if len(ys) > 0:
        y_min = ys.min()
        y_max = ys.max()

        # 기본적으로 torso 하단 12% 제거
        # z_body가 작거나 음수면 덜 자르고, 크면 조금 더 자름
        clip_ratio = 0.90 - min(0.06, max(0.0, z_body * 0.02))
        clip_ratio = max(0.84, min(0.92, clip_ratio))

        cut_y = int(y_min + (y_max - y_min) * clip_ratio)
        torso[cut_y:, :] = 0

        print(f"[mask clip] torso: clip_ratio={clip_ratio:.2f}, cut_y={cut_y}")

    result_masks.append(torso)

if 'upper_arm' in masks_dict:
    upper_arm = masks_dict['upper_arm'].copy()
    z_sh = z_scores['shoulder']

    # 어깨/팔은 너무 과하게 키우지 않음
    scale_x = z_to_scale(z_sh, alpha=0.02, min_scale=0.97, max_scale=1.04)
    scale_y = 1.0

    print(f"[mask scale] upper_arm: z_shoulder={z_sh:+.2f}, scale_x={scale_x:.2f}")
    upper_arm = scale_mask_bbox(upper_arm, scale_x=scale_x, scale_y=scale_y)
    result_masks.append(upper_arm)

if 'lower_arm' in masks_dict:
    lower_arm = masks_dict['lower_arm'].copy()
    z_sl = z_scores['sleeve']

    scale_x = 1.0
    scale_y = z_to_scale(z_sl, alpha=0.02, min_scale=0.96, max_scale=1.04)

    print(f"[mask scale] lower_arm: z_sleeve={z_sl:+.2f}, scale_y={scale_y:.2f}")
    lower_arm = scale_mask_bbox(lower_arm, scale_x=scale_x, scale_y=scale_y)
    result_masks.append(lower_arm)

if not result_masks:
    return np.zeros((1024, 1024), dtype=np.uint8)

combined = np.sum(result_masks, axis=0)
combined = np.uint8(combined > 0) * 255

# 너무 붙는 현상 완화용 padding은 약하게만 적용
combined = cv2.dilate(
    combined,
    kernel=np.ones((3, 3), np.uint8),
    iterations=1
)

return combined



def z_to_prompt(z_scores):"""z-score → Fooocus 프롬프트 문자열"""z_avg = np.mean([z_scores['chest'], z_scores['waist'], z_scores['shoulder']])

if z_avg >= 2.0:
    return "big baggy loose overfitted"
elif z_avg >= 1.0:
    return "loose fitting oversized"
elif z_avg >= 0.5:
    return "slightly loose fitting"
elif z_avg <= -2.0:
    return "very tight cropped"
elif z_avg <= -1.0:
    return "tight fitting form fitting"
elif z_avg <= -0.5:
    return "slightly tight fitting"
else:
    return "fitted regular fit"

════════════════════════════════════════════════════════

유틸

════════════════════════════════════════════════════════

def make_dir(path):if not os.path.exists(path):os.makedirs(path)

def image_to_bytes(image_path):if type(image_path) == str:with open(image_path, "rb") as img_file:return img_file.read()else:buf = io.BytesIO()Image.fromarray(image_path).save(buf, format='JPEG')buf.seek(0)return buf.read()

════════════════════════════════════════════════════════

Fooocus API 호출

════════════════════════════════════════════════════════

def fooocus_api(user_image_path,mask_image_path,out_path,garment_image_path,garment_type,z_scores,        # dict or Noneprompt,mask_only,masks_dict=None, # 부위별 마스크 dict (garment 있을 때)):if garment_image_path is None:# 나체 생성 단계input_mask = np.array(Image.open(mask_image_path))if mask_only:return input_mask

    input_image_bytes = image_to_bytes(user_image_path)
    input_mask_bytes  = image_to_bytes(mask_image_path)

    additional_params = {
        "prompt": prompt,
        "image_prompts": [{"cn_stop": 0.7, "cn_weight": 1.6,
                           "cn_type": "PyraCanny", "cn_img": input_mask_bytes}],
        "cn_stop1": 0.7, "cn_weight1": 1.6,
        "cn_type1": "PyraCanny", "cn_img1": input_mask_bytes,
        "advanced_params": json.dumps({
            "mixing_image_prompt_and_inpaint": "true",
            "inpaint_engine": "v2.6",
        }),
    }
    files = {
        "input_image": input_image_bytes,
        "input_mask":  input_mask_bytes,
        "cn_img1":     input_mask_bytes,
    }

else:
    # 옷 합성 단계
    # ── 부위별 마스크 조절 ──────────────────────────────────────
    if masks_dict is not None and z_scores is not None:
        input_mask = adjust_mask_by_region(masks_dict, z_scores)
    else:
        input_mask = np.array(Image.open(mask_image_path))
        input_mask = cv2.dilate(input_mask, kernel=np.ones((5, 5)),
                                iterations=DILATE_ITERATIONS)
        input_mask = cv2.dilate(input_mask, kernel=np.ones((5, 5)), iterations=1)

    # ── 프롬프트 ─────────────────────────────────────────────────
    if z_scores is not None:
        prompt_fit = z_to_prompt(z_scores)
    else:
        prompt_fit = "fitted"

    if mask_only:
        return input_mask

    assert garment_type in ['top', 'pants', 'skirt', 'dress', 'jump suit']
    full_prompt = f'{garment_type}+++ , {prompt_fit}'

    input_image_bytes   = image_to_bytes(user_image_path)
    input_mask_bytes    = image_to_bytes(input_mask)
    garment_image_bytes = image_to_bytes(garment_image_path)

    additional_params = {
        "prompt": "a person wearing " + full_prompt,
        "image_prompts": [
            {"cn_stop": 1.0, "cn_weight": 1.6,
             "cn_type": "ImagePrompt", "cn_img": garment_image_bytes},
            {"cn_stop": 0.7, "cn_weight": 1.6,
             "cn_type": "PyraCanny",   "cn_img": input_mask_bytes},
        ],
        "cn_stop1": 1.0, "cn_weight1": 1.6,
        "cn_type1": "ImagePrompt", "cn_img1": garment_image_bytes,
        "cn_stop2": 0.7, "cn_weight2": 1.6,
        "cn_type2": "PyraCanny",   "cn_img2": input_mask_bytes,
        "advanced_params": json.dumps({
            "mixing_image_prompt_and_inpaint": "true",
            "inpaint_engine": "v2.6",
        }),
    }
    files = {
        "input_image": input_image_bytes,
        "input_mask":  input_mask_bytes,
        "cn_img1":     garment_image_bytes,
        "cn_img2":     input_mask_bytes,
    }

host   = "http://127.0.0.1:8888"
url    = f"{host}/v1/generation/image-prompt"
params = {
    "negative_prompt": "(worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art:1.4), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), morbid, ugly, asymmetrical, mutated malformed, mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts, out of focus, glitch, duplicate, (airbrushed, cartoon, anime, semi-realistic, cgi, render, blender, digital art, manga, amateur:1.3), (3D ,3D Game, 3D Game Scene, 3D Character:1.1), (bad hands, bad anatomy, bad body, bad face, bad teeth, bad arms, bad legs, deformities:1.3)",
    "inpaint_additional_prompt": prompt,
    "performance_selection": "Speed",
    "style_selections": ["Fooocus V2", "Fooocus Enhance", "Fooocus Sharp"],
    "loras": json.dumps([]),
    "guidance_scale": 7.5,
}
params.update(additional_params)

response = requests.post(url=url, data=params, files=files, timeout=600)
res_json = response.json()
print("Fooocus response:", res_json)

if isinstance(res_json, list):
    if len(res_json) == 0:
        raise ValueError("Fooocus returned empty list.")
    img_url = res_json[0]["url"]
elif isinstance(res_json, dict):
    img_url = (res_json.get("url") or
               res_json.get("image_url") or
               res_json.get("result", [{}])[0].get("url"))
else:
    raise ValueError(f"Unexpected Fooocus response: {res_json}")

if img_url is None:
    raise ValueError(f"Cannot find image url: {res_json}")

os.system(f'wget -O "{out_path}" "{img_url}"')

════════════════════════════════════════════════════════

SAM / DensePose

════════════════════════════════════════════════════════

def select_random_points(binary_mask, k, upper):binary_mask = cv2.erode(binary_mask, kernel=np.ones((5, 5)), iterations=7)non_zero_indices = np.transpose(np.nonzero(binary_mask))mean_x = np.mean(non_zero_indices[:, 1])min_x  = np.min(non_zero_indices[:, 1])max_x  = np.max(non_zero_indices[:, 1])if upper:filtered = non_zero_indices[(min_x + 10 < non_zero_indices[:, 1]) &(non_zero_indices[:, 1] < mean_x - 10)]else:filtered = non_zero_indices[(max_x - 10 > non_zero_indices[:, 1]) &(non_zero_indices[:, 1] > mean_x + 10)]selected = filtered[np.random.choice(len(filtered),size=min(k, len(filtered)), replace=False)].tolist()return [(y, x) for (x, y) in selected]

def bounding_box(mask):rows = np.any(mask, axis=1)cols = np.any(mask, axis=0)ymin, ymax = np.where(rows)[0][[0, -1]]xmin, xmax = np.where(cols)[0][[0, -1]]return [xmin, ymin, xmax, ymax]

@torch.no_grad()def sam(user_image_path, garment_mask_path, upper):sam_model = sam_model_registry"vit_tiny"device = 'cuda' if torch.cuda.is_available() else 'cpu'sam_model = sam_model.to(device)predictor = SamPredictor(sam_model)

image = np.array(Image.open(user_image_path).convert('RGB').resize((1024, 1024)))
predictor.set_image(image)
mask  = np.array(Image.open(garment_mask_path).convert('L').resize((1024, 1024)))
input_point = np.array(select_random_points(mask, k=1, upper=upper))
input_label = np.ones(input_point.shape[0])
input_box   = np.array(bounding_box(mask))

ret, _, _ = predictor.predict(
    point_coords=input_point, point_labels=input_label,
    box=input_box, multimask_output=False, hq_token_only=True)
return ret

@torch.no_grad()def get_body_mask(image_path, uid):os.system(f"python ./DensePose/apply_net.py show "f"./DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml "f"https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl "f"{image_path} dp_segm --output ./cache/{uid}/densepose.png")

def combine_body_mask(garment_type, top_sleeve_length=None,bottom_leg_length=None, uid=None):body_mask_list = glob.glob(f'./cache/{uid}/densepose.*')assert garment_type in ['top', 'pants', 'skirt', 'jump suit', 'dress']assert top_sleeve_length in [None, 'no', 'short', 'long']assert bottom_leg_length in [None, 'no', 'short', 'long']mask_list = []

if garment_type != 'top':
    assert bottom_leg_length in ['short', 'long']
    bot_list = [f for f in body_mask_list if 'upper_leg' in f]
    if bottom_leg_length == 'long':
        bot_list += [f for f in body_mask_list if 'lower_leg' in f]
    bot = np.sum([np.array(Image.open(f).convert('L')) for f in bot_list], axis=0)
    bot = np.uint8(bot > 0)
    if bottom_leg_length == 'short':
        idx = np.argwhere(bot > 0)
        min_x, _ = idx.min(axis=0); max_x, _ = idx.max(axis=0)
        bot[max(0, max_x - (max_x-min_x)//3):, :] = 0
    if garment_type in ['skirt', 'dress']:
        idx = np.argwhere(bot > 0)
        min_x, _ = idx.min(axis=0); max_x, _ = idx.max(axis=0)
        for i in range(min_x, max_x+1):
            row = bot[i][None]
            ri = np.argwhere(row > 0)
            if len(ri): bot[i, ri.min():ri.max()+1] = 1
    mask_list.append(bot)

if garment_type not in ['pants', 'skirt']:
    top_list = ([f for f in body_mask_list if 'upper_arm' in f]
                if top_sleeve_length != 'no' else [])
    if top_sleeve_length == 'long':
        top_list += [f for f in body_mask_list if 'lower_arm' in f]
    if top_list:
        top = np.sum([np.array(Image.open(f).convert('L')) for f in top_list], axis=0)
        top = np.uint8(top > 0)
        if top_sleeve_length == 'short':
            idx = np.argwhere(top > 0)
            min_x, _ = idx.min(axis=0); max_x, _ = idx.max(axis=0)
            top[max(0, max_x - (max_x-min_x)//3):, :] = 0
        mask_list.append(top)

torso = np.uint8(np.array(Image.open(f'./cache/{uid}/densepose.torso.png').convert('L')) > 0)
if garment_type not in ['dress', 'jump suit']:
    idx = np.argwhere(torso > 0)
    min_x, _ = idx.min(axis=0); max_x, _ = idx.max(axis=0)
    if garment_type == 'top':
        torso[max_x - (max_x-min_x)//6:] = 0
    else:
        torso[:min_x + (max_x-min_x)*2//3] = 0
mask_list.append(torso)

combined = np.sum(mask_list, axis=0)
return np.uint8(combined > 0) * 255

def load_region_masks(uid, body_mask_list, garment_type, top_sleeve_length):"""부위별 마스크를 dict로 반환"""masks = {}

# 몸통
torso_path = f'./cache/{uid}/densepose.torso.png'
if os.path.exists(torso_path):
    masks['torso'] = np.uint8(
        np.array(Image.open(torso_path).convert('L')) > 0) * 255

# 상박 (어깨)
upper_arm_files = [f for f in body_mask_list if 'upper_arm' in f]
if upper_arm_files and top_sleeve_length != 'no':
    ua = np.sum([np.array(Image.open(f).convert('L')) for f in upper_arm_files], axis=0)
    masks['upper_arm'] = np.uint8(ua > 0) * 255

# 하박 (소매)
lower_arm_files = [f for f in body_mask_list if 'lower_arm' in f]
if lower_arm_files and top_sleeve_length == 'long':
    la = np.sum([np.array(Image.open(f).convert('L')) for f in lower_arm_files], axis=0)
    masks['lower_arm'] = np.uint8(la > 0) * 255

return masks

════════════════════════════════════════════════════════

main

════════════════════════════════════════════════════════

def main():"""실행 방법:

[직접 입력 모드 — 테스트용]
python backbone_standalone.py \\
    <user_image> <garment_image> <uid> \\
    <garment_type> <top_sleeve_length> <bottom_leg_length> \\
    --body_height=170 --breast_size=88 --waist_size=72 --hip_size=94 \\
    --shoulders_width=38 --arm_length=58 --waist_height=100 \\
    --back_length=40 --weight=60 --gender=0

예시:
python backbone_standalone.py models/man_1.jpg examples/shirt.jpg 1 \\
    top short none \\
    --body_height=170 --breast_size=88 --waist_size=72 --hip_size=94 \\
    --shoulders_width=38 --arm_length=58 --waist_height=100 \\
    --back_length=40 --weight=60 --gender=0
"""
user_image_path    = sys.argv[1]
garment_image_path = sys.argv[2]
uid                = sys.argv[3]
garment_type       = sys.argv[4]
top_sleeve_length  = sys.argv[5] if sys.argv[5] != 'none' else None
bottom_leg_length  = sys.argv[6] if sys.argv[6] != 'none' else None

# ── 의류 치수 + 사용자 신체 치수 파싱 (--key=value 형식) ─────────────
# garment_* 는 현재 직접 입력. 추후 FitNet 출력값으로 자동 대체 예정.
garment_dims = {
    'shoulder': 42.0,
    'length':   65.0,
    'chest':   102.0,
    'waist':    98.0,
    'sleeve':   58.0,
}

# target_* 는 실제로 옷을 입혀볼 사용자의 신체 치수
target_body = {
    'shoulder': 38.0,
    'back':     40.0,
    'chest':    88.0,
    'waist':    72.0,
    'arm':      58.0,
}

for arg in sys.argv[7:]:
    if arg.startswith('--'):
        k, v = arg[2:].split('=')
        v = float(v)

        if k == 'garment_shoulder':
            garment_dims['shoulder'] = v
        elif k == 'garment_length':
            garment_dims['length'] = v
        elif k == 'garment_chest':
            garment_dims['chest'] = v
        elif k == 'garment_waist':
            garment_dims['waist'] = v
        elif k == 'garment_sleeve':
            garment_dims['sleeve'] = v

        elif k == 'target_shoulder':
            target_body['shoulder'] = v
        elif k == 'target_back':
            target_body['back'] = v
        elif k == 'target_chest':
            target_body['chest'] = v
        elif k == 'target_waist':
            target_body['waist'] = v
        elif k == 'target_arm':
            target_body['arm'] = v

print("[의류 치수]", {k: f"{v:.1f}cm" for k, v in garment_dims.items()})
print("[사용자 신체 치수]", {k: f"{v:.1f}cm" for k, v in target_body.items()})

# ── 부위별 z-score 계산 ──────────────────────────────────────────
z_scores = compute_relative_z(garment_dims, target_body)

# ── 마스크 생성 ──────────────────────────────────────────────────
body_mask_labels = f"{garment_type}_{top_sleeve_length}_{bottom_leg_length}"
body_mask_args = {
    'garment_type':      garment_type,
    'top_sleeve_length': top_sleeve_length,
    'bottom_leg_length': bottom_leg_length,
    'uid': uid,
}

make_dir(f'./cache/{uid}')
make_dir(f'./cache/{uid}/{body_mask_labels}')
garment_mask_path              = f'./cache/{uid}/{body_mask_labels}/mask.png'
existing_garment_bound_mask_path = f'./cache/{uid}/{body_mask_labels}/bound_mask.png'

if not os.path.exists(f"./cache/{uid}/densepose.torso.png"):
    get_body_mask(user_image_path, uid)

body_mask_list = glob.glob(f'./cache/{uid}/densepose.*')

if not os.path.exists(garment_mask_path):
    body_mask = combine_body_mask(**body_mask_args)
    Image.fromarray(body_mask).save(garment_mask_path)

if not os.path.exists(existing_garment_bound_mask_path):
    existing_garment_bound_mask = combine_body_mask(
        garment_type='jump suit', top_sleeve_length='long',
        bottom_leg_length='long', uid=uid)
    existing_garment_bound_mask = cv2.dilate(
        existing_garment_bound_mask, kernel=np.ones((5, 5)),
        iterations=DILATE_ITERATIONS * 2)
    Image.fromarray(existing_garment_bound_mask).save(existing_garment_bound_mask_path)

existing_garment_bound_mask = np.array(
    Image.open(existing_garment_bound_mask_path).convert('L'))

# ── SAM으로 기존 옷 마스크 ───────────────────────────────────────
prompt = "a naked person wearing"
body_mask_path = f'./cache/{uid}/{body_mask_labels}/body_mask.png'
if not os.path.exists(body_mask_path):
    masks = sam(user_image_path, garment_mask_path,
                upper=(garment_type == "top"))
    masks = Image.fromarray(masks[0].astype(np.uint8) * 255)
    masks.save(body_mask_path)
    body_mask = np.array(
        Image.open(body_mask_path).convert('L').resize(
            Image.open(user_image_path).size))
    body_mask = cv2.dilate(body_mask, kernel=np.ones((5, 5)),
                           iterations=DILATE_ITERATIONS - 1)
    body_mask[existing_garment_bound_mask <= 0] = 0
    Image.fromarray((body_mask > 0).astype(np.uint8) * 255).save(body_mask_path)

# ── 부위별 마스크 로드 ───────────────────────────────────────────
region_masks = load_region_masks(uid, body_mask_list,
                                 garment_type, top_sleeve_length)
print(f"[부위별 마스크] 로드된 부위: {list(region_masks.keys())}")

# ── 나체 생성 (필요시) ───────────────────────────────────────────
naked_out_path = f'cache/{uid}/{body_mask_labels}/user_body_{prompt}.jpg'
old_garment_mask = fooocus_api(
    user_image_path=user_image_path,
    mask_image_path=body_mask_path,
    out_path=naked_out_path,
    garment_image_path=None,
    garment_type=None,
    z_scores=None,
    prompt=prompt,
    mask_only=True,
)

make_dir(f'./results/{uid}')
basename = str(len(glob.glob(f'results/{uid}/*.jpg')) + 1).zfill(10) + ".jpg"
final_out_path = os.path.join('results', uid, basename)

# ── 새 옷 마스크 (부위별 z-score 적용) ──────────────────────────
new_garment_mask = fooocus_api(
    user_image_path=user_image_path,
    mask_image_path=garment_mask_path,
    out_path=final_out_path,
    garment_image_path=garment_image_path,
    garment_type=garment_type,
    z_scores=z_scores,
    prompt=None,
    mask_only=True,
    masks_dict=region_masks,
)

if np.min(1. * new_garment_mask - old_garment_mask) < 0:
    fooocus_api(
        user_image_path=user_image_path,
        mask_image_path=body_mask_path,
        out_path=naked_out_path,
        garment_image_path=None,
        garment_type=None,
        z_scores=None,
        prompt=prompt,
        mask_only=False,
    )
    final_input_path = naked_out_path
else:
    final_input_path = user_image_path

# ── 최종 합성 ────────────────────────────────────────────────────
fooocus_api(
    user_image_path=final_input_path,
    mask_image_path=garment_mask_path,
    out_path=final_out_path,
    garment_image_path=garment_image_path,
    garment_type=garment_type,
    z_scores=z_scores,
    prompt=None,
    mask_only=False,
    masks_dict=region_masks,
)

if name == 'main':main()
