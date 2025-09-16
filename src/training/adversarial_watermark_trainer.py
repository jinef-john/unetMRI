# adversarial_watermark_trainer.py
"""
Robust Adversarial Watermarking for MRI Images  (fixed twin-head C2)
...
"""

import os, sys, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import torchvision.transforms as T, numpy as np, cv2, logging, pathlib
from torch.utils.data import DataLoader
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.efficientnet_cbam import EfficientNetB3_CBAM_Bottleneck   
from models.autoencoder           import AutoEncoder
from models.u2net                 import U2NET
from utils.data_loader            import MRIDataset
from utils.metrics                import calculate_ssim, calculate_psnr
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.cuda.empty_cache()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
NUM_CLASSES = len(CLASSES)

BATCH_SIZE      = 8
EPOCHS_WARMUP   = 3
EPOCHS_ADV      = 5
EPOCHS_TOTAL    = 8
LR_MODE         = 1e-3
LR_WM           = 1e-3
LR_GEN          = 2e-4
WATERMARK_INTENSITY_BASE = 0.15
WATERMARK_INTENSITY_MAX  = 0.40
# ----------------------------------------------

# -------- entropy helper (a) ------------------
def entropy(p):                 # p = softmax logits
    p = F.softmax(p, dim=1)
    return -torch.sum(p * torch.log(p + 1e-8), dim=1).mean()
# ----------------------------------------------

# ---------- small CBAM block ------------------
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction, 1), nn.ReLU(),
            nn.Conv2d(channels//reduction, channels, 1))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)
    def forward(self, x):
        ca = torch.sigmoid(self.mlp(x))
        x = x * ca
        sa = torch.sigmoid(self.spatial(torch.cat([x.mean(dim=1,keepdim=True), x.max(dim=1,keepdim=True)[0]],1)))
        return x * sa
# ----------------------------------------------

class DualHeadC2Classifier(nn.Module):
    """
    Twin-head C2:  shared stem  →  clean branch (detached)  →  WM branch
    """
    def __init__(self, arch='tf_efficientnet_b3', num_classes=NUM_CLASSES):
        super().__init__()
        import timm
        base = timm.create_model(arch, pretrained=True, features_only=False)

        # 1-channel input
        old_conv = base.conv_stem
        base.conv_stem = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        # shared stem (blocks 0-1)
        self.stem = nn.Sequential(
            base.conv_stem, base.bn1, nn.SiLU(inplace=True), *base.blocks[:2]
        )

        # Get stem output channels (32 for EfficientNet-B3 blocks[:2])
        self.stem_channels = 32  # From the terminal output, blocks[1] outputs 32 channels

        # WM-only trunk
        self.wm_trunk = nn.Sequential(
            *base.blocks[2:],
            base.conv_head,  # 384 -> 1536 channels
            base.bn2,        # batch norm after conv_head
            nn.SiLU(inplace=True),
            CBAM(base.num_features),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        # heads
        self.mode_det_clean = nn.Linear(self.stem_channels, 2)      # clean uses stem features
        self.clean_class    = nn.Linear(self.stem_channels, num_classes)  # ← NEW
        self.mode_det_wm    = nn.Linear(base.num_features, 2)       # wm uses full features
        self.wm_class       = nn.Linear(base.num_features, num_classes)   # wm classification

    def forward(self, x, mode='clean'):
        f = self.stem(x)                                # B, C=32, H, W
        gap = F.adaptive_avg_pool2d(f, 1).flatten(1)    # B, 32

        if mode == 'clean':
            gap = gap.detach()  # gradient isolation for stem
            # ensure clean_class gets a detached feature input (clean head must not train features)
            return self.mode_det_clean(gap), self.clean_class(gap.detach())
        else:  # mode == 'wm'
            f_full = self.wm_trunk(f)                   # B, 1536 (full features)
            mode_logits = self.mode_det_wm(f_full)
            class_logits = self.wm_class(f_full)
            return mode_logits, class_logits


class FrequencyWatermarkGenerator(nn.Module):
    """Image-dependent watermark generator — no label embedding"""
    def __init__(self, latent_dim=1024):
        super().__init__()
        self.watermark_net = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, latent_dim, 3, padding=1), nn.Tanh())
        # NOTE: removed class_embed to avoid label-dependent watermarks
        self.freq_gens = nn.ModuleList([nn.Conv2d(latent_dim, latent_dim, 3, padding=1)
                                        for _ in [0.1,0.3,0.7]])   

    # forward signature changed: (img, intensity) — no labels
    def forward(self, img, intensity):
        img = F.interpolate(img, size=(32,32), mode='bilinear', align_corners=False)
        w = self.watermark_net(img) * intensity
        for i,fg in enumerate(self.freq_gens):
            w = w + fg(w) * [0.1,0.3,0.7][i]
        return w


class AdversarialWatermarkTrainer:
    def __init__(self, data_root:str, pretrained_dir:str):
        self.data_root, self.pretrained_dir = data_root, pretrained_dir
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
        self.logger = logging.getLogger(__name__)
        self._load_models()
        self._setup_optimizers()
        self._setup_data()
        self.norm = T.Normalize(mean=[0.5], std=[0.5])
        self.current_epoch = 0
        self.watermark_intensity = WATERMARK_INTENSITY_BASE

    # ---------------- model loading -----------------
    def _load_models(self):
        self.logger.info("Loading models...")
        # C1 (frozen)
        self.c1 = EfficientNetB3_CBAM_Bottleneck(num_classes=NUM_CLASSES)
        first_conv = self.c1.base.features[0][0]
        self.c1.base.features[0][0] = nn.Conv2d(1, first_conv.out_channels,3,1,1,bias=False)
        ckpt = torch.load(os.path.join(self.pretrained_dir,"MRI-C1EfficientNet_B3_CBAM.pth"), map_location=DEVICE)
        self.c1.load_state_dict({k.replace('module.',''):v for k,v in ckpt.items()})
        self.c1.eval().requires_grad_(False).to(DEVICE)

        # auto-encoder (frozen)
        self.ae = AutoEncoder()
        ckpt = torch.load(os.path.join(self.pretrained_dir,"autoencoder_epoch7.pth"), map_location=DEVICE)
        self.ae.load_state_dict({k.replace('module.',''):v for k,v in ckpt.items()})
        self.ae.eval().requires_grad_(False).to(DEVICE)

        # U²-Net (frozen)
        self.u2net = U2NET()
        u2path = os.path.join(self.pretrained_dir,"u2net.pth")
        if os.path.exists(u2path): self.u2net.load_state_dict(torch.load(u2path, map_location=DEVICE))
        self.u2net.eval().requires_grad_(False).to(DEVICE)

        # trainable nets
        self.c2  = DualHeadC2Classifier().to(DEVICE)
        self.gen = FrequencyWatermarkGenerator().to(DEVICE)

        if torch.cuda.device_count()>1:
            self.c1  = nn.DataParallel(self.c1)
            self.ae  = nn.DataParallel(self.ae)
            self.u2net=nn.DataParallel(self.u2net)
            self.c2  = nn.DataParallel(self.c2)
            self.gen = nn.DataParallel(self.gen)
        self.logger.info(f"Using {torch.cuda.device_count()} GPUs")

    # ---------------- optimisers -----------------
    def _setup_optimizers(self):
        # ---- freeze clean-class head ----
        clean_head = (self.c2.module.clean_class if hasattr(self.c2, 'module')
                      else self.c2.clean_class)
        for p in clean_head.parameters():
            p.requires_grad = False
        # ---------------------------------

        # 1. mode detectors – both clean and WM mode detection heads
        if hasattr(self.c2, 'module'):
            mode_params = list(self.c2.module.mode_det_clean.parameters()) + \
                          list(self.c2.module.mode_det_wm.parameters())
            wm_params = list(self.c2.module.wm_trunk.parameters()) + \
                        list(self.c2.module.wm_class.parameters())
        else:
            mode_params = list(self.c2.mode_det_clean.parameters()) + \
                          list(self.c2.mode_det_wm.parameters())
            wm_params = list(self.c2.wm_trunk.parameters()) + \
                        list(self.c2.wm_class.parameters())

        self.opt_mode = optim.Adam(mode_params, lr=LR_MODE, weight_decay=1e-4)
        self.opt_wm   = optim.Adam(wm_params,   lr=LR_WM, weight_decay=0)
        self.opt_gen  = optim.Adam(self.gen.parameters(), lr=LR_GEN, weight_decay=1e-5)

    # ---------------- data -----------------
    def _setup_data(self):
        from torch.utils.data import Subset
        dataset = MRIDataset(self.data_root, CLASSES)
        if len(dataset)>1000:
            import random
            random.seed(42)
            per_class = 250
            idxs=[]
            for c in range(NUM_CLASSES):
                c_idx = [i for i,label in enumerate(dataset.labels) if label==c]
                idxs+=random.sample(c_idx, per_class) if len(c_idx)>=per_class else c_idx
            dataset = Subset(dataset, idxs)
        self.loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                                 num_workers=2, pin_memory=False)
        self.logger.info(f"Dataset: {len(dataset)} samples")

    # ---------------- helpers -----------------
    def encode(self, x):            # → latent 32×32×1024
        with torch.no_grad():
            x = x*2-1
            latents, skip64 = self.ae.module.encoder(x) if hasattr(self.ae,'module') else self.ae.encoder(x)
            return latents, skip64
    def decode(self, lat, skip):    # ← latent  →  image
        with torch.no_grad():
            return self.ae.module.decoder(lat, skip) if hasattr(self.ae,'module') else self.ae.decoder(lat, skip)

    # ---- Replaced brain_mask per patch: soft invert + per-sample random leak ----
    def brain_mask(self, x):        # 1 = background, 0 = brain
        with torch.no_grad():
            x3 = x.repeat(1,3,1,1) if x.size(1)==1 else x
            m = self.u2net(x3)
            if isinstance(m,(list,tuple)): m=m[0]
            mask = torch.sigmoid(m)
            mask = (1 - mask) ** 2  # soft invert
            leak = torch.rand_like(mask) * 0.05
            return (mask + leak).clamp(0, 1)

    # ---------------- training -----------------
    def train_epoch(self, adversarial: bool):
        def to_float(x):
            if isinstance(x, (float, int)):
                return float(x)
            return float(x.item())

        self.c2.train()
        self.gen.train() if adversarial else self.gen.eval()

        metrics = {
            'loss_mode': 0,
            'loss_wm': 0,
            'loss_clean_cls': 0,
            'loss_clean_ent': 0,
            'loss_gen': 0,
            'acc_c1_clean': 0,
            'acc_c1_wm': 0,
            'acc_mode': 0,
            'acc_clean': 0,
            'acc_wm': 0,
            'ssim': 0,
            'psnr': 0,
        }
        N = 0

        for idx, (img, labels, _, _) in enumerate(self.loader):
            img, labels = img.to(DEVICE), labels.to(DEVICE)
            B = img.size(0)

            # ---- clean pipeline ----
            latents, skip64 = self.encode(img)
            clean_img = self.decode(latents, skip64)
            clean_norm = self.norm(clean_img)
            with torch.no_grad():
                c1_clean_logits = self.c1(clean_norm)

            if adversarial:
                # build watermark
                mask = self.brain_mask(img)
                mask_32 = F.interpolate(mask, size=(32, 32), mode='bilinear', align_corners=False)

                # generator now takes (img, intensity) — no label leakage
                w = self.gen(img, self.watermark_intensity)
                latents_wm = latents + w * mask_32
                wm_img = self.decode(latents_wm, skip64)
                wm_norm = self.norm(wm_img)

                # ---- forward C2 ----
                noise = torch.randn_like(clean_norm) * 0.02  # 2 % noise
                clean_noisy = clean_norm + noise
                mode_clean, cls_clean = self.c2(clean_noisy, mode='clean')
                mode_logits_wm, class_logits_wm = self.c2(wm_norm, mode='wm')

                # targets
                tgt_mode_clean = torch.zeros(B, dtype=torch.long, device=DEVICE)
                tgt_mode_wm = torch.ones(B, dtype=torch.long, device=DEVICE)

                # losses (C2)
                loss_mode = (
                    F.cross_entropy(mode_clean, tgt_mode_clean)
                    + F.cross_entropy(mode_logits_wm, tgt_mode_wm)
                )
                loss_wm = F.cross_entropy(class_logits_wm, labels)
                loss_clean_cls = F.cross_entropy(cls_clean, labels)

                # ---------- entropy reg ----------
                loss_clean_ent = entropy(cls_clean)
                # detach version for generator usage
                loss_clean_ent_detached = loss_clean_ent.detach()

                # --- update C2 ---
                self.opt_mode.zero_grad()
                self.opt_wm.zero_grad()
                (loss_mode + loss_wm + loss_clean_cls).backward()
                self.opt_mode.step()
                self.opt_wm.step()

                # --- feature divergence ---
                stem_net = self.c2.module.stem if hasattr(self.c2, "module") else self.c2.stem
                with torch.no_grad():
                    f_clean = stem_net(clean_norm)
                    f_wm = stem_net(wm_norm)
                # detach clean features to avoid double-backward
                loss_feat_div = -F.cosine_similarity(
                    f_clean.detach().flatten(1), f_wm.flatten(1), dim=1
                ).mean()

                # Entropy clamp (force C2_clean entropy ≥ log(#classes))
                target_entropy = np.log(NUM_CLASSES)
                entropy_loss = F.relu(target_entropy - loss_clean_ent.detach())

                # --- update Generator ---
                w_gen = self.gen(img, self.watermark_intensity)
                print(
                    f"DEBUG batch {idx}: |w| inside mask = {(w_gen.abs() * mask_32).mean().item():.4f}  "
                    f"| outside = {(w_gen.abs() * (1 - mask_32)).mean().item():.4f}"
                )
                latents_gen = latents + w_gen * mask_32
                wm_img_gen = self.decode(latents_gen, skip64)
                wm_norm_gen = self.norm(wm_img_gen)
                _, cls_gen = self.c2(wm_norm_gen, mode='wm')

                loss_gen = (
                    F.cross_entropy(cls_gen, labels)
                    + 1.0 * (w_gen.abs() * (1 - mask_32)).mean()
                    - 5.0 * loss_clean_ent_detached
                    + 2.0 * loss_feat_div
                    + 3.0 * entropy_loss
                )

                self.opt_gen.zero_grad()
                loss_gen.backward()
                self.opt_gen.step()

                # --- metrics ---
                with torch.no_grad():
                    c1_wm_logits = self.c1(wm_norm)
                    acc_c1_clean = (c1_clean_logits.argmax(1) == labels).float().mean()
                    acc_c1_wm = (c1_wm_logits.argmax(1) == labels).float().mean()
                    acc_mode = (
                        torch.cat([mode_clean.argmax(1), mode_logits_wm.argmax(1)])
                        == torch.cat([tgt_mode_clean, tgt_mode_wm])
                    ).float().mean()
                    acc_clean = (cls_clean.argmax(1) == labels).float().mean()
                    acc_wm = (class_logits_wm.argmax(1) == labels).float().mean()
                    ssim = calculate_ssim(clean_img, wm_img)
                    psnr = calculate_psnr(clean_img, wm_img)

                loss_gen_val = loss_gen.detach()

            else:
                # warmup – only mode detector on clean
                mode_logits, _ = self.c2(clean_norm, mode='clean')
                tgt_mode = torch.zeros(B, dtype=torch.long, device=DEVICE)
                loss_mode = F.cross_entropy(mode_logits, tgt_mode)
                self.opt_mode.zero_grad()
                loss_mode.backward()
                self.opt_mode.step()

                with torch.no_grad():
                    acc_c1_clean = (c1_clean_logits.argmax(1) == labels).float().mean()
                    acc_mode = (mode_logits.argmax(1) == tgt_mode).float().mean()
                    acc_clean = torch.tensor(0.0, device=DEVICE)
                    acc_wm = torch.tensor(0.0, device=DEVICE)
                    ssim, psnr = 1.0, 50.0

                loss_wm = torch.tensor(0.0, device=DEVICE)
                loss_gen = torch.tensor(0.0, device=DEVICE)
                loss_clean_cls = torch.tensor(0.0, device=DEVICE)
                loss_clean_ent = torch.tensor(0.0, device=DEVICE)
                acc_c1_wm = acc_c1_clean
                loss_gen_val = torch.tensor(0.0, device=DEVICE)

            # ---- accumulate ----
            for k, v in zip(
                metrics.keys(),
                [
                    to_float(loss_mode),
                    to_float(loss_wm),
                    to_float(loss_clean_cls),
                    to_float(loss_clean_ent),
                    to_float(loss_gen_val) if adversarial else to_float(loss_gen),
                    to_float(acc_c1_clean),
                    to_float(acc_c1_wm),
                    to_float(acc_mode),
                    to_float(acc_clean),
                    to_float(acc_wm),
                    float(ssim),
                    float(psnr),
                ],
            ):
                metrics[k] += v

            N += 1
            if idx % 10 == 0:
                if adversarial:
                    self.logger.info(
                        f"batch {idx}: "
                        f"C2_mode={metrics['acc_mode']/N:.3f}  "
                        f"C2_clean={metrics['acc_clean']/N:.3f}  "
                        f"C2_wm={metrics['acc_wm']/N:.3f}  "
                        f"SSIM={metrics['ssim']/N:.4f}"
                    )
                else:
                    self.logger.info(
                        f"warmup batch {idx}: "
                        f"C1={metrics['acc_c1_clean']/N:.3f}  "
                        f"C2_mode={metrics['acc_mode']/N:.3f}"
                    )

        return {k: v / N for k, v in metrics.items()}



    # ---------------- main train loop -----------------
    def train(self, save_dir:str="./output/adversarial_training"):
        os.makedirs(save_dir, exist_ok=True)
        for epoch in range(1, EPOCHS_TOTAL+1):
            self.current_epoch=epoch
            adv = epoch > EPOCHS_WARMUP
            self.logger.info(f"\n=== EPOCH {epoch}/{EPOCHS_TOTAL} ({'ADV' if adv else 'WARMUP'}) ===")
            m = self.train_epoch(adversarial=adv)

            if adv:
                self.logger.info(f"Results:  C1_clean={m['acc_c1_clean']:.3f}  C1_wm={m['acc_c1_wm']:.3f} | "
                                 f"C2_mode={m['acc_mode']:.3f}  C2_clean={m['acc_clean']:.3f}  C2_wm={m['acc_wm']:.3f} | "
                                 f"SSIM={m['ssim']:.4f}  PSNR={m['psnr']:.2f}")
                # improved intensity scheduler: ramp until 90% wm acc
                if m['acc_wm'] < 0.90:
                    # self.watermark_intensity = min(self.watermark_intensity * 1.3, 0.30)  # ← cap at 0.30/
                    self.watermark_intensity = 0.30


                    self.logger.info(f"↑ intensity → {self.watermark_intensity:.3f}")
            else:
                self.logger.info(f"Warmup:  C1_clean={m['acc_c1_clean']:.3f}  C2_mode={m['acc_mode']:.3f}")

            # save every epoch
            ckpt = {'epoch':epoch,
                    'c2_state_dict': self.c2.state_dict(),
                    'gen_state_dict':self.gen.state_dict(),
                    'opt_mode':self.opt_mode.state_dict(),
                    'opt_wm':self.opt_wm.state_dict(),
                    'opt_gen':self.opt_gen.state_dict(),
                    'intensity':self.watermark_intensity,
                    'metrics':m}
            torch.save(ckpt, os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth"))
        return save_dir


# ------------------------------ main ---------------------------------
def main():
    data_root   = "/teamspace/studios/this_studio/unetMRI/dataset/brain-tumor-mri-dataset/Training"
    pretrained  = "/teamspace/studios/this_studio/unetMRI/pt models"
    out_dir     = "/teamspace/studios/this_studio/unetMRI/output/adversarial_training_robust"
    trainer = AdversarialWatermarkTrainer(data_root, pretrained)
    trainer.train(out_dir)

if __name__ == "__main__":
    main()
