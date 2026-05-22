import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights


# ────────────────────────────────────────────────
# FiLM Layer
# ────────────────────────────────────────────────
class FiLM(nn.Module):
    def __init__(self, num_channels: int, body_dim: int):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(body_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_channels * 2),
        )
        nn.init.zeros_(self.generator[-1].weight)
        nn.init.ones_(self.generator[-1].bias[:num_channels])
        nn.init.zeros_(self.generator[-1].bias[num_channels:])

    def forward(self, feature: torch.Tensor, body_vec: torch.Tensor) -> torch.Tensor:
        params = self.generator(body_vec)
        gamma, beta = params.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * feature + beta


# ────────────────────────────────────────────────
# ConvBlock
# ────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers += [
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ────────────────────────────────────────────────
# Encoder (ResNet50 backbone + FiLM)
# ────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, body_dim: int, pretrained: bool = True):
        super().__init__()
        weights  = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)

        self.enc1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.enc2 = backbone.layer1  
        self.enc3 = backbone.layer2  
        self.enc4 = backbone.layer3  
        self.enc5 = backbone.layer4  

        self.film3 = FiLM(512,  body_dim)
        self.film4 = FiLM(1024, body_dim)
        self.film5 = FiLM(2048, body_dim)

    def forward(self, x: torch.Tensor, body_vec: torch.Tensor):
        e1 = self.enc1(x)                                  
        e2 = self.enc2(self.pool(e1))                      
        e3 = self.film3(self.enc3(e2), body_vec)      
        e4 = self.film4(self.enc4(e3), body_vec)           
        e5 = self.film5(self.enc5(e4), body_vec)           
        return e1, e2, e3, e4, e5


# ────────────────────────────────────────────────
# Bottleneck
# ────────────────────────────────────────────────
class Bottleneck(nn.Module):
    def __init__(self, body_dim: int, dropout: float = 0.2):
        super().__init__()
        self.block = ConvBlock(2048, 1024, dropout=dropout)
        self.film  = FiLM(1024, body_dim)

    def forward(self, x: torch.Tensor, body_vec: torch.Tensor) -> torch.Tensor:
        return self.film(self.block(x), body_vec)          


# ────────────────────────────────────────────────
# Decoder  (ResNet50 skip 채널 기준)
# ────────────────────────────────────────────────
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up4  = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = ConvBlock(512 + 1024, 512)

        self.up3  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(256 + 512,  256)

        self.up2  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(128 + 256,  128)

        self.up1  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(64  + 64,   64)

        self.up0  = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec0 = ConvBlock(32, 32)

        self.seg_head = nn.Conv2d(32, 1, 1)

    def forward(self, neck, e1, e2, e3, e4):
        d4 = self.dec4(torch.cat([self.up4(neck), e4], dim=1)) 
        d3 = self.dec3(torch.cat([self.up3(d4),   e3], dim=1))  
        d2 = self.dec2(torch.cat([self.up2(d3),   e2], dim=1)) 
        d1 = self.dec1(torch.cat([self.up1(d2),   e1], dim=1))  
        d0 = self.dec0(self.up0(d1))                            
        seg = torch.sigmoid(self.seg_head(d0))
        return seg, d1, d2, d3, d4  


# ────────────────────────────────────────────────────────────────────────
# SPP (Spatial Pyramid Pooling)
# ────────────────────────────────────────────────────────────────────────
class SPP(nn.Module):
    def __init__(self, pool_sizes: list = (1, 2, 4)):
        super().__init__()
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(s) for s in pool_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        return torch.cat([p(x).view(B, -1) for p in self.pools], dim=1)


# ────────────────────────────────────────────────────────────────────────
# MaskGuidedAttention
# ────────────────────────────────────────────────────────────────────────
class MaskGuidedAttention(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.attn_conv = nn.Conv2d(in_channels, 1, 1)
        self.mask_temp = nn.Parameter(torch.ones(1))
        self.bg_bias   = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self,
        feat: torch.Tensor,         
        seg_mask: torch.Tensor,    
    ) -> torch.Tensor:

        mask_r = F.interpolate(
            seg_mask, size=feat.shape[2:], mode='bilinear', align_corners=False
        ) 

        attn_logit = self.attn_conv(feat) + self.mask_temp * mask_r + self.bg_bias
        attn_weight = torch.sigmoid(attn_logit)  

        pooled = (feat * attn_weight).sum(dim=[2, 3]) / (
            attn_weight.sum(dim=[2, 3]) + 1e-6
        )  
        return pooled


# ────────────────────────────────────────────────────────────────────────
# RegressionHead — SPP + MaskGuidedAttention + 다중 스케일 피처 융합
# ────────────────────────────────────────────────────────────────────────
class RegressionHead(nn.Module):
    def __init__(
        self,
        body_dim: int = 10,
        num_measurements: int = 5,
        d1_ch: int = 64,
        d2_ch: int = 128,
        d3_ch: int = 256,
        d4_ch: int = 512,
        compress_ch: int = 64,
        spp_sizes: tuple = (1, 2, 4),
    ):
        super().__init__()

        self.attn_d1 = MaskGuidedAttention(d1_ch)

        self.spp_d2  = SPP(spp_sizes)
        n_spp_bins   = sum(s * s for s in spp_sizes)  
        d2_spp_dim   = d2_ch * n_spp_bins             

        self.compress_d3 = nn.Sequential(
            nn.Conv2d(d3_ch, compress_ch, 1, bias=False),
            nn.BatchNorm2d(compress_ch),
            nn.ReLU(inplace=True),
        )
        self.compress_d4 = nn.Sequential(
            nn.Conv2d(d4_ch, compress_ch, 1, bias=False),
            nn.BatchNorm2d(compress_ch),
            nn.ReLU(inplace=True),
        )
        self.spp_d3  = SPP(spp_sizes)
        self.spp_d4  = SPP(spp_sizes)
        d3_spp_dim   = compress_ch * n_spp_bins        
        d4_spp_dim   = compress_ch * n_spp_bins       

        fusion_dim = d1_ch + d2_spp_dim + d3_spp_dim + d4_spp_dim + body_dim

        self.fc = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_measurements),
        )

    def forward(
        self,
        seg_feat: torch.Tensor,         
        body_vec: torch.Tensor,         
        seg_mask: torch.Tensor = None, 
        d2_feat: torch.Tensor = None,   
        d3_feat: torch.Tensor = None,
        d4_feat: torch.Tensor = None,  
    ) -> torch.Tensor:

        if seg_mask is not None:
            feat_d1 = self.attn_d1(seg_feat, seg_mask)  
        else:
            feat_d1 = seg_feat.mean(dim=[2, 3])          

        if d2_feat is not None:
            feat_d2 = self.spp_d2(d2_feat)             
        else:
            feat_d2 = torch.zeros(
                seg_feat.size(0), self.spp_d2.pools[0].output_size[0] ** 0 * 128,
                device=seg_feat.device
            )

        if d3_feat is not None:
            feat_d3 = self.spp_d3(self.compress_d3(d3_feat)) 
        else:
            feat_d3 = torch.zeros(
                seg_feat.size(0), 1344, device=seg_feat.device
            )

        if d4_feat is not None:
            feat_d4 = self.spp_d4(self.compress_d4(d4_feat))  
        else:
            feat_d4 = torch.zeros(
                seg_feat.size(0), 1344, device=seg_feat.device
            )

        fused = torch.cat([feat_d1, feat_d2, feat_d3, feat_d4, body_vec], dim=1)
        return self.fc(fused)


# ────────────────────────────────────────────────
# 추론 전용 전처리기 (학습과 동일한 MediaPipe 전처리)
# ────────────────────────────────────────────────
class _ImagePreprocessor:
    TARGET_SIZE   = 224
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(self):
        import os, urllib.request
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            model_path = "/tmp/pose_landmarker.task"
            if not os.path.exists(model_path):
                urllib.request.urlretrieve(
                    "https://storage.googleapis.com/mediapipe-models/"
                    "pose_landmarker/pose_landmarker_lite/float16/1/"
                    "pose_landmarker_lite.task",
                    model_path,
                )
            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=False,
            )
            self.detector  = vision.PoseLandmarker.create_from_options(options)
            self.available = True
            print("[_ImagePreprocessor] MediaPipe 초기화 완료")
        except Exception as e:
            self.detector  = None
            self.available = False
            print(f"[_ImagePreprocessor] MediaPipe 없음 → letterbox fallback: {e}")

    def _letterbox(self, image):
        import cv2
        h, w = image.shape[:2]
        s = self.TARGET_SIZE
        scale = s / max(h, w)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top, left = (s - nh) // 2, (s - nw) // 2
        return cv2.copyMakeBorder(
            resized,
            top, s - nh - top, left, s - nw - left,
            cv2.BORDER_CONSTANT, value=(128, 128, 128),
        )

    def _center_crop_or_pad(self, image):
        h, w = image.shape[:2]
        s = self.TARGET_SIZE
        if h >= s and w >= s:
            return image[(h - s) // 2:(h - s) // 2 + s,
                         (w - s) // 2:(w - s) // 2 + s]
        return self._letterbox(image)

    def __call__(self, image_bgr, shoulder_width_cm: float = 39.0):
        import cv2
        import numpy as np
        from torchvision import transforms
        from PIL import Image

        result_image = None

        if self.available and shoulder_width_cm > 0:
            try:
                import mediapipe as mp
                h, w = image_bgr.shape[:2]
                rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = self.detector.detect(mp_image)

                if result.pose_landmarks:
                    lms = result.pose_landmarks[0]
                    lx = int(lms[11].x * w); ly = int(lms[11].y * h)
                    rx = int(lms[12].x * w); ry = int(lms[12].y * h)
                    shoulder_px = np.sqrt((rx - lx) ** 2 + (ry - ly) ** 2)

                    if shoulder_px >= 10:
                        scale = 5.0 / (shoulder_px / shoulder_width_cm)
                        new_w, new_h = int(w * scale), int(h * scale)
                        if new_w >= 10 and new_h >= 10:
                            rescaled = cv2.resize(image_bgr, (new_w, new_h))
                            result_image = self._center_crop_or_pad(rescaled)
            except Exception:
                pass

        if result_image is None:
            result_image = self._letterbox(image_bgr)

        rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])
        return to_tensor(pil).unsqueeze(0)  


# ────────────────────────────────────────────────
# 메인 모델
# ────────────────────────────────────────────────
class ClothingMeasurementNet(nn.Module):
    def __init__(self, body_dim: int = 10, num_measurements: int = 5,
                 pretrained: bool = True):
        super().__init__()
        self.encoder    = Encoder(body_dim, pretrained=pretrained)
        self.bottleneck = Bottleneck(body_dim)
        self.decoder    = Decoder()
        self.reg_head   = RegressionHead(body_dim, num_measurements)

        self._preprocessor = None

    def forward(self, image: torch.Tensor, body_vec: torch.Tensor):
        e1, e2, e3, e4, e5 = self.encoder(image, body_vec)
        neck = self.bottleneck(e5, body_vec)
        seg, d1, d2, d3, d4 = self.decoder(neck, e1, e2, e3, e4)
        meas = self.reg_head(
            d1, body_vec,
            seg_mask=seg,
            d2_feat=d2,
            d3_feat=d3,
            d4_feat=d4,
        )
        return seg, meas

    def predict(self, image_bgr, body_info: dict, device=None):
        from dataset import normalize_body, denormalize_clothes, CLOTHES_KEYS

        if device is None:
            device = next(self.parameters()).device

        if self._preprocessor is None:
            self._preprocessor = _ImagePreprocessor()

        shoulder_width_cm = float(body_info.get("shoulders_width", 39) or 39)
        image_tensor = self._preprocessor(image_bgr, shoulder_width_cm).to(device)

        meta = {f"metadata.model.{k}": v for k, v in body_info.items()}
        meta["metadata.model.gender"] = body_info.get("gender", "FEMALE")
        body_vec = normalize_body(meta).unsqueeze(0).to(device)

        self.eval()
        with torch.no_grad():
            _, meas_pred = self.forward(image_tensor, body_vec)

        meas_cm    = denormalize_clothes(meas_pred.cpu())[0]
        item_names = [k.split(".")[-1] for k in CLOTHES_KEYS]
        return {name: round(meas_cm[i].item(), 1) for i, name in enumerate(item_names)}


# ────────────────────────────────────────────────
# Loss
# ────────────────────────────────────────────────
class CombinedLoss(nn.Module):
    def __init__(self, lambda1: float = 1.0, lambda2: float = 0.5):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.bce = nn.BCELoss()
        self.meas_weights = torch.tensor([1.0, 1.0, 1.2, 1.2, 2.0])

    def forward(self, seg_pred, seg_gt, meas_pred, meas_gt):
        loss_seg  = self.bce(seg_pred, seg_gt)
        w = self.meas_weights.to(meas_pred.device)
        loss_meas = ((meas_pred - meas_gt) ** 2 * w).mean()
        total = self.lambda1 * loss_seg + self.lambda2 * loss_meas
        return total, loss_seg, loss_meas


# ────────────────────────────────────────────────
# Phase별 Freeze 유틸리티
# ────────────────────────────────────────────────
def get_optimizer_phase1(model: ClothingMeasurementNet,
                         lr_decoder: float = 1e-3) -> torch.optim.Optimizer:
    for param in model.encoder.parameters():
        param.requires_grad = False
    trainable = (list(model.bottleneck.parameters()) +
                 list(model.decoder.parameters())    +
                 list(model.reg_head.parameters()))
    return torch.optim.Adam(trainable, lr=lr_decoder, weight_decay=1e-4)


def get_optimizer_phase2(model: ClothingMeasurementNet,
                         lr_encoder: float = 1e-4,
                         lr_others:  float = 1e-3) -> torch.optim.Optimizer:
    for param in model.parameters():
        param.requires_grad = True
    enc_low  = (list(model.encoder.enc1.parameters()) +
                list(model.encoder.enc2.parameters()))
    enc_high = (list(model.encoder.enc3.parameters()) +
                list(model.encoder.enc4.parameters()) +
                list(model.encoder.enc5.parameters()) +
                list(model.encoder.film3.parameters()) +
                list(model.encoder.film4.parameters()) +
                list(model.encoder.film5.parameters()))
    others   = (list(model.bottleneck.parameters()) +
                list(model.decoder.parameters())    +
                list(model.reg_head.parameters()))
    return torch.optim.Adam([
        {"params": enc_low,  "lr": lr_encoder * 0.1},
        {"params": enc_high, "lr": lr_encoder},
        {"params": others,   "lr": lr_others},
    ], weight_decay=1e-4)


# ────────────────────────────────────────────────
# Ablation 모델들
# ────────────────────────────────────────────────
class BaselineMLP(nn.Module):
    def __init__(self, body_dim: int = 10, num_measurements: int = 5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(body_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_measurements),
        )

    def forward(self, image: torch.Tensor, body_vec: torch.Tensor):
        meas = self.fc(body_vec)
        seg  = torch.zeros(image.shape[0], 1, 224, 224, device=image.device)
        return seg, meas


class SegOnlyModel(nn.Module):
    def __init__(self, body_dim: int = 10, num_measurements: int = 5):
        super().__init__()
        self.model = ClothingMeasurementNet(body_dim, num_measurements,
                                            pretrained=False)

    def forward(self, image, body_vec):
        return self.model(image, body_vec)


class NoFiLMModel(nn.Module):
    def __init__(self, body_dim: int = 10, num_measurements: int = 5):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3,   64,  3, padding=1, bias=False), nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.Conv2d(64,  64,  3, padding=1, bias=False), nn.BatchNorm2d(64),  nn.ReLU(True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64,  128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True),
        )
        self.neck = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(True),
            nn.Conv2d(1024,1024, 3, padding=1, bias=False), nn.BatchNorm2d(1024), nn.ReLU(True),
        )
        self.up4   = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4  = ConvBlock(512 + 512,  512)
        self.up3   = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3  = ConvBlock(256 + 256,  256)
        self.up2   = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2  = ConvBlock(128 + 128,  128)
        self.up1   = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1  = ConvBlock(64  + 64,   64)
        self.seg_head = nn.Conv2d(64, 1, 1)
        self.reg_head = RegressionHead(body_dim, num_measurements)
        self.pool     = nn.MaxPool2d(2)

    def forward(self, image, body_vec):
        e1 = self.enc1(image)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        neck = self.neck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(neck), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4),   e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3),   e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2),   e1], dim=1))
        seg = torch.sigmoid(self.seg_head(d1))
        meas = self.reg_head(d1, body_vec, seg_mask=seg,
                             d2_feat=d2, d3_feat=d3, d4_feat=d4)
        return seg, meas


# ────────────────────────────────────────────────
# Shape 검증
# ────────────────────────────────────────────────
if __name__ == "__main__":
    BODY_DIM = 10
    B = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"

    models_to_test = [
        ("ClothingMeasurementNet", ClothingMeasurementNet(BODY_DIM, pretrained=False)),
        ("BaselineMLP",            BaselineMLP(BODY_DIM)),
        ("SegOnlyModel",           SegOnlyModel(BODY_DIM)),
        ("NoFiLMModel",            NoFiLMModel(BODY_DIM)),
    ]

    print("=== 모델 Shape 검증 ===\n")
    for name, model in models_to_test:
        model = model.to(device)
        img  = torch.randn(B, 3, 224, 224).to(device)
        body = torch.randn(B, BODY_DIM).to(device)
        seg, meas = model(img, body)
        params = sum(p.numel() for p in model.parameters())
        print(f"{name}")
        print(f"  seg : {seg.shape}  meas: {meas.shape}")
        print(f"  params: {params:,}\n")

    # Loss 검증
    model = ClothingMeasurementNet(BODY_DIM, pretrained=False).to(device)
    img   = torch.randn(B, 3, 224, 224).to(device)
    body  = torch.randn(B, BODY_DIM).to(device)
    seg, meas = model(img, body)
    seg_gt    = torch.randint(0, 2, seg.shape).float().to(device)
    meas_gt   = torch.randn(B, 5).to(device)

    criterion = CombinedLoss()
    total, seg_l, meas_l = criterion(seg, seg_gt, meas, meas_gt)
    print(f"Loss 검증:")
    print(f"  Total : {total.item():.4f}")
    print(f"  BCE   : {seg_l.item():.4f}  (x{criterion.lambda1})")
    print(f"  MSE   : {meas_l.item():.4f}  (x{criterion.lambda2})")

    # predict() 사용 예시
    print("\n=== predict() 사용 예시 ===")
    import numpy as np
    model_inf = ClothingMeasurementNet(BODY_DIM, pretrained=False)
    model_inf.eval()
    dummy_image = np.full((600, 400, 3), 128, dtype=np.uint8)
    body_info = {
        "body_height":        165,
        "breast_size_female": 88,
        "waist_size":         68,
        "hip_seize":          93,
        "shoulders_width":    38,
        "arm_length":         53,
        "waist_height":       97,
        "back_length":        41,
        "weight":             55,
        "gender":             "FEMALE",
    }
    result = model_inf.predict(dummy_image, body_info)
    print("예측 결과 (cm):", result)