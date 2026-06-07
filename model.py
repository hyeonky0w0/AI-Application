import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights


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

class BodyCrossAttention(nn.Module):
    def __init__(self, feat_dim: int, body_dim: int,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.body_dim = body_dim
        self.pos_embed = nn.Embedding(body_dim, feat_dim)
        self.val_embed = nn.Linear(1, feat_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.norm    = nn.LayerNorm(feat_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, feature: torch.Tensor, body_vec: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feature.shape

        feat_seq = feature.flatten(2).permute(0, 2, 1)

        idx        = torch.arange(self.body_dim, device=body_vec.device)
        pos_emb    = self.pos_embed(idx)                      
        val_emb    = self.val_embed(body_vec.unsqueeze(-1))   
        body_tokens = val_emb + pos_emb                       

        attended, _ = self.cross_attn(
            query=feat_seq,
            key=body_tokens,
            value=body_tokens,
        )

        out = self.norm(feat_seq + self.dropout(attended))
        return out.permute(0, 2, 1).reshape(B, C, H, W)


class IdentityConditioning(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, feature: torch.Tensor, body_vec: torch.Tensor) -> torch.Tensor:
        return feature


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

class Encoder(nn.Module):
    def __init__(self, body_dim: int, pretrained: bool = True,
                 use_cross_attn: bool = False, use_film: bool = True):
        super().__init__()
        weights  = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)

        self.enc1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.enc2 = backbone.layer1   # 256ch
        self.enc3 = backbone.layer2   # 512ch
        self.enc4 = backbone.layer3   # 1024ch
        self.enc5 = backbone.layer4   # 2048ch

        if use_cross_attn:
            cond_cls = BodyCrossAttention
        elif use_film:
            cond_cls = FiLM
        else:
            cond_cls = IdentityConditioning
        self.cond3 = cond_cls(512,  body_dim)
        self.cond4 = cond_cls(1024, body_dim)
        self.cond5 = cond_cls(2048, body_dim)

    def forward(self, x: torch.Tensor, body_vec: torch.Tensor):
        e1 = self.enc1(x)                         
        e2 = self.enc2(self.pool(e1))             
        e3 = self.cond3(self.enc3(e2), body_vec)  
        e4 = self.cond4(self.enc4(e3), body_vec)  
        e5 = self.cond5(self.enc5(e4), body_vec)  
        return e1, e2, e3, e4, e5


class Bottleneck(nn.Module):
    def __init__(self, body_dim: int, dropout: float = 0.2,
                 use_cross_attn: bool = False, use_film: bool = True):
        super().__init__()
        self.block = ConvBlock(2048, 1024, dropout=dropout)
        if use_cross_attn:
            cond_cls = BodyCrossAttention
        elif use_film:
            cond_cls = FiLM
        else:
            cond_cls = IdentityConditioning
        self.cond  = cond_cls(1024, body_dim)

    def forward(self, x: torch.Tensor, body_vec: torch.Tensor) -> torch.Tensor:
        return self.cond(self.block(x), body_vec)   


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
        return seg, d1


class RegressionHead(nn.Module):
    def __init__(self, body_dim: int, num_measurements: int = 5):
        super().__init__()
        self.attn = nn.Conv2d(64, 1, 1)
        self.fc = nn.Sequential(
            nn.Linear(64 + body_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_measurements),
        )

    def forward(self, seg_feat: torch.Tensor, body_vec: torch.Tensor,
                seg_mask: torch.Tensor = None) -> torch.Tensor:
        if seg_mask is not None:
            seg_resized = F.interpolate(seg_mask, size=seg_feat.shape[2:],
                                        mode='bilinear', align_corners=False)
            attn = torch.sigmoid(self.attn(seg_feat)) * seg_resized
        else:
            attn = torch.softmax(
                self.attn(seg_feat).flatten(2), dim=2
            ).view_as(seg_feat)
        pooled   = (seg_feat * attn).sum(dim=[2, 3]) / (attn.sum(dim=[2, 3]) + 1e-6)
        combined = torch.cat([pooled, body_vec], dim=1)
        return self.fc(combined)


class MeasurementQueryTransformer(nn.Module):
    def __init__(self, feat_dim: int = 64, body_dim: int = 10,
                 num_measurements: int = 5, num_heads: int = 4,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.body_dim        = body_dim
        self.num_measurements = num_measurements

        self.meas_queries = nn.Parameter(torch.randn(num_measurements, feat_dim))
        nn.init.trunc_normal_(self.meas_queries, std=0.02)

        if body_dim > 0:
            self.body_pos_embed = nn.Embedding(body_dim, feat_dim)
            self.body_val_embed = nn.Linear(1, feat_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feat_dim,
            nhead=num_heads,
            dim_feedforward=feat_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 쿼리 1개 → 치수 1개 (scalar)
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 1),
        )

    def forward(self, seg_feat: torch.Tensor, body_vec: torch.Tensor,
                seg_mask: torch.Tensor = None) -> torch.Tensor:
        B = seg_feat.shape[0]

        img_tokens = seg_feat.flatten(2).permute(0, 2, 1)

        # seg_mask로 옷 영역 토큰 강조
        if seg_mask is not None:
            w = F.interpolate(seg_mask, size=seg_feat.shape[2:],
                              mode='bilinear', align_corners=False)
            img_tokens = img_tokens * (1.0 + w.flatten(2).permute(0, 2, 1))

        if self.body_dim > 0:
            idx         = torch.arange(self.body_dim, device=body_vec.device)
            pos_emb     = self.body_pos_embed(idx)                    
            val_emb     = self.body_val_embed(body_vec.unsqueeze(-1)) 
            body_tokens = val_emb + pos_emb                           
            context = torch.cat([img_tokens, body_tokens], dim=1)     
        else:
            context = img_tokens                                     

        queries = self.meas_queries.unsqueeze(0).expand(B, -1, -1)

        out = self.transformer(queries, context)  

        meas = self.head(out).squeeze(-1)          
        return meas


class ClothingMeasurementNet(nn.Module):
    def __init__(self, body_dim: int = 10, num_measurements: int = 5,
                 pretrained: bool = True, use_cross_attn: bool = False,
                 use_film: bool = True, head_type: str = "attn"):
        super().__init__()
        self.encoder    = Encoder(body_dim, pretrained=pretrained,
                                  use_cross_attn=use_cross_attn, use_film=use_film)
        self.bottleneck = Bottleneck(body_dim, use_cross_attn=use_cross_attn,
                                     use_film=use_film)
        self.decoder    = Decoder()

        if head_type == "qformer":
            self.reg_head = MeasurementQueryTransformer(
                feat_dim=64, body_dim=body_dim,
                num_measurements=num_measurements,
            )
        else:
            self.reg_head = RegressionHead(body_dim, num_measurements)

    def forward(self, image: torch.Tensor, body_vec: torch.Tensor):
        e1, e2, e3, e4, e5 = self.encoder(image, body_vec)
        neck = self.bottleneck(e5, body_vec)
        seg, feat = self.decoder(neck, e1, e2, e3, e4)
        meas = self.reg_head(feat, body_vec, seg_mask=seg)
        return seg, meas


class ClothingMeasurementNetCA(ClothingMeasurementNet):
    """Cross-Attention 버전 (Exp5)."""
    def __init__(self, body_dim: int = 10, num_measurements: int = 5,
                 pretrained: bool = True):
        super().__init__(body_dim, num_measurements, pretrained,
                         use_cross_attn=True, use_film=True, head_type="attn")


class ClothingMeasurementNetQFormer(ClothingMeasurementNet):
    """Full model: FiLM + Q-Former 헤드 (Exp6, 최종 모델)."""
    def __init__(self, body_dim: int = 10, num_measurements: int = 5,
                 pretrained: bool = True):
        super().__init__(body_dim, num_measurements, pretrained,
                         use_cross_attn=False, use_film=True, head_type="qformer")


class ImageOnlyQFormerModel(ClothingMeasurementNet):
    """Exp2: 이미지만, body 무시, FiLM 없음, Q-Former 헤드."""
    def __init__(self, body_dim: int = 10, num_measurements: int = 5,
                 pretrained: bool = True):
        super().__init__(0, num_measurements, pretrained,
                         use_cross_attn=False, use_film=False, head_type="qformer")


class ImageBodyNoFiLMQFormerModel(ClothingMeasurementNet):
    """Exp3: 이미지 + body, FiLM 없음, Q-Former 헤드."""
    def __init__(self, body_dim: int = 10, num_measurements: int = 5,
                 pretrained: bool = True):
        super().__init__(body_dim, num_measurements, pretrained,
                         use_cross_attn=False, use_film=False, head_type="qformer")


class CombinedLoss(nn.Module):
    def __init__(self, lambda1: float = 1.0, lambda2: float = 0.5):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.bce = nn.BCELoss()
        # [shoulder, front, chest, waist, sleeve]
        self.meas_weights = torch.tensor([1.0, 1.0, 1.8, 1.8, 2.0])

    def forward(self, seg_pred, seg_gt, meas_pred, meas_gt):
        loss_seg  = self.bce(seg_pred, seg_gt)
        w = self.meas_weights.to(meas_pred.device)
        loss_meas = ((meas_pred - meas_gt) ** 2 * w).mean()
        total = self.lambda1 * loss_seg + self.lambda2 * loss_meas
        return total, loss_seg, loss_meas


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
                list(model.encoder.cond3.parameters()) +
                list(model.encoder.cond4.parameters()) +
                list(model.encoder.cond5.parameters()))
    others   = (list(model.bottleneck.parameters()) +
                list(model.decoder.parameters())    +
                list(model.reg_head.parameters()))
    return torch.optim.Adam([
        {"params": enc_low,  "lr": lr_encoder * 0.1},
        {"params": enc_high, "lr": lr_encoder},
        {"params": others,   "lr": lr_others},
    ], weight_decay=1e-4)


# Ablation 모델들
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
        self.dec4  = ConvBlock(512 + 512,  512)   # 1024ch (e4=512)

        self.up3   = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3  = ConvBlock(256 + 256,  256)   # 512ch  (e3=256)

        self.up2   = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2  = ConvBlock(128 + 128,  128)   # 256ch  (e2=128)

        self.up1   = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1  = ConvBlock(64  + 64,   64)    # 128ch  (e1=64)

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

        meas = self.reg_head(d1, body_vec, seg_mask=seg)
        return seg, meas


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

    #loss
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
    print(f"  BCE   : {seg_l.item():.4f}  (×{criterion.lambda1})")
    print(f"  MSE   : {meas_l.item():.4f}  (×{criterion.lambda2})")