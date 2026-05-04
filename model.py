import torch
import torch.nn as nn
import torch.nn.functional as F

# FiLM Layer
class FiLM(nn.Module):
    def __init__(self, num_channels: int, body_dim: int):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(body_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_channels * 2),
        )
        # γ를 1 근방으로 초기화: 학습 초반 안정성
        nn.init.zeros_(self.generator[-1].weight)
        nn.init.ones_(self.generator[-1].bias[:num_channels])
        nn.init.zeros_(self.generator[-1].bias[num_channels:])

    def forward(self, feature: torch.Tensor, body_vec: torch.Tensor) -> torch.Tensor:
        params = self.generator(body_vec)             
        gamma, beta = params.chunk(2, dim=1)          
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * feature + beta

#ConvBlock
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

# 인코더
class Encoder(nn.Module):
    def __init__(self, body_dim: int, dropout: float = 0.1):
        super().__init__()
        self.enc1 = ConvBlock(3,   64)
        self.enc2 = ConvBlock(64,  128)
        self.enc3 = ConvBlock(128, 256, dropout=dropout)
        self.enc4 = ConvBlock(256, 512, dropout=dropout)
        self.pool = nn.MaxPool2d(2)
        # enc3, enc4에 FiLM
        self.film3 = FiLM(256, body_dim)
        self.film4 = FiLM(512, body_dim)

    def forward(self, x: torch.Tensor, body_vec: torch.Tensor):
        e1 = self.enc1(x)                           # [B, 64,  224, 224]
        e2 = self.enc2(self.pool(e1))               # [B, 128, 112, 112]
        e3 = self.film3(self.enc3(self.pool(e2)), body_vec)  # [B, 256, 56, 56]
        e4 = self.film4(self.enc4(self.pool(e3)), body_vec)  # [B, 512, 28, 28]
        return e1, e2, e3, e4

# 보틀넥
class Bottleneck(nn.Module):
    def __init__(self, body_dim: int, dropout: float = 0.2):
        super().__init__()
        self.block = ConvBlock(512, 1024, dropout=dropout)
        self.film  = FiLM(1024, body_dim)

    def forward(self, x: torch.Tensor, body_vec: torch.Tensor) -> torch.Tensor:
        return self.film(self.block(x), body_vec)   # [B, 1024, 14, 14]

#Decoder
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up4  = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = ConvBlock(1024, 512)

        self.up3  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(512,  256)

        self.up2  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256,  128)

        self.up1  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(128,  64)

        # 최종 세그멘테이션
        self.seg_head = nn.Conv2d(64, 1, 1)

    def forward(self, neck, e1, e2, e3, e4):
        d4 = self.dec4(torch.cat([self.up4(neck), e4], dim=1)) #인코더의 중간 피쳐 연결하는 부분이 skip connection
        d3 = self.dec3(torch.cat([self.up3(d4),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3),  e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2),  e1], dim=1))
        seg = torch.sigmoid(self.seg_head(d1))
        return seg, d1

#Regression Head
class RegressionHead(nn.Module):
    def __init__(self, body_dim: int, num_measurements: int = 5):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(64 + body_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_measurements),
        )

    def forward(self, seg_feat: torch.Tensor, body_vec: torch.Tensor) -> torch.Tensor:
        pooled   = self.gap(seg_feat).flatten(1)
        combined = torch.cat([pooled, body_vec], dim=1)
        return self.fc(combined)


class ClothingMeasurementNet(nn.Module):
    def __init__(self, body_dim: int = 10, num_measurements: int = 5):
        super().__init__()
        self.encoder    = Encoder(body_dim)
        self.bottleneck = Bottleneck(body_dim)
        self.decoder    = Decoder()
        self.reg_head   = RegressionHead(body_dim, num_measurements)

    def forward(self, image: torch.Tensor, body_vec: torch.Tensor):
        e1, e2, e3, e4 = self.encoder(image, body_vec)
        neck = self.bottleneck(self.encoder.pool(e4), body_vec)
        seg, feat = self.decoder(neck, e1, e2, e3, e4)
        meas = self.reg_head(feat, body_vec)
        return seg, meas


#Ablation용 모델
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
        seg = torch.zeros(image.shape[0], 1, 224, 224, device=image.device)
        return seg, meas

class SegOnlyModel(nn.Module):
    #전처리 없는 데이터로 학습하는 버전
    def __init__(self, body_dim: int = 10, num_measurements: int = 5):
        super().__init__()
        self.model = ClothingMeasurementNet(body_dim, num_measurements)

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
        self.decoder  = Decoder()
        self.reg_head = RegressionHead(body_dim, num_measurements)
        self.pool     = nn.MaxPool2d(2)

    def forward(self, image, body_vec):
        e1 = self.enc1(image)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        neck = self.neck(self.pool(e4))
        seg, feat = self.decoder(neck, e1, e2, e3, e4)
        meas = self.reg_head(feat, body_vec)
        return seg, meas

#loss 함수
class CombinedLoss(nn.Module):
    def __init__(self, lambda1: float = 1.0, lambda2: float = 0.1):
        super().__init__()
        self.lambda1  = lambda1
        self.lambda2  = lambda2
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()

    def forward(self, seg_pred, seg_gt, meas_pred, meas_gt):
        loss_seg  = self.bce(seg_pred, seg_gt)
        loss_meas = self.mse(meas_pred, meas_gt)
        total = self.lambda1 * loss_seg + self.lambda2 * loss_meas
        return total, loss_seg, loss_meas


if __name__ == "__main__":
    BODY_DIM = 10
    B = 2

    print("=== 모델 Shape 검증 ===\n")
    for name, model in [
        ("ClothingMeasurementNet (Full)",  ClothingMeasurementNet(BODY_DIM)),
        ("BaselineMLP (Exp1)",             BaselineMLP(BODY_DIM)),
        ("NoFiLMModel (Exp3)",             NoFiLMModel(BODY_DIM)),
    ]:
        img  = torch.randn(B, 3, 224, 224)
        body = torch.randn(B, BODY_DIM)
        seg, meas = model(img, body)
        params = sum(p.numel() for p in model.parameters())
        print(f"{name}")
        print(f"  seg:  {seg.shape}  meas: {meas.shape}")
        print(f"  파라미터: {params:,}\n")

    # Loss 확인
    model = ClothingMeasurementNet(BODY_DIM)
    img   = torch.randn(B, 3, 224, 224)
    body  = torch.randn(B, BODY_DIM)
    seg, meas = model(img, body)

    seg_gt  = torch.randint(0, 2, seg.shape).float()
    meas_gt = torch.randn(B, 5)

    criterion = CombinedLoss(lambda1=1.0, lambda2=0.1)
    total, seg_loss, meas_loss = criterion(seg, seg_gt, meas, meas_gt)
    print(f"Loss 확인:")
    print(f"  Total: {total.item():.4f}")
    print(f"  BCE:   {seg_loss.item():.4f}  (×1.0)")
    print(f"  MSE:   {meas_loss.item():.4f}  (×0.1)")
