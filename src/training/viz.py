import os
import torch
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np
from PIL import Image
import torchvision.transforms as T

from adversarial_watermark_trainer import DualHeadC2Classifier, FrequencyWatermarkGenerator
from models.autoencoder import AutoEncoder
from models.u2net import U2NET
from utils.data_loader import MRIDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- helpers --------------------
def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    return new_state_dict

def load_models(ckpt_path, device=DEVICE):
    ckpt = torch.load(ckpt_path, map_location=device)
    c2 = DualHeadC2Classifier().to(device)
    gen = FrequencyWatermarkGenerator().to(device)
    c2_state = remove_module_prefix(ckpt['c2_state_dict'])
    gen_state = remove_module_prefix(ckpt['gen_state_dict'])
    c2.load_state_dict(c2_state, strict=False)
    gen.load_state_dict(gen_state, strict=False)
    intensity = ckpt.get("intensity", 0.15)
    c2.eval(); gen.eval()
    return c2, gen, intensity

def load_frozen_ae_u2net(pretrained_dir, device=DEVICE):
    ae = AutoEncoder().eval().requires_grad_(False).to(device)
    ae_ckpt = torch.load(os.path.join(pretrained_dir, "autoencoder_epoch7.pth"), map_location=device)
    ae.load_state_dict({k.replace('module.', ''): v for k, v in ae_ckpt.items()})
    u2net = U2NET().eval().requires_grad_(False).to(device)
    u2path = os.path.join(pretrained_dir, "u2net.pth")
    if os.path.exists(u2path):
        u2net.load_state_dict(torch.load(u2path, map_location=device))
    return ae, u2net

def save_images(images, titles=None, out_dir="viz_output", filename="clean_vs_wm.png"):
    os.makedirs(out_dir, exist_ok=True)
    n = len(images)
    ncols = len(images)
    fig, axes = plt.subplots(1, ncols, figsize=(4*ncols,4))
    if ncols == 1: axes=[axes]
    for i, img in enumerate(images):
        if torch.is_tensor(img):
            img = img.detach().cpu().numpy().squeeze()
        axes[i].imshow(img, cmap="gray")
        axes[i].axis("off")
        if titles: axes[i].set_title(titles[i], fontsize=12)
    plt.tight_layout()
    save_path = os.path.join(out_dir, filename)
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved image to {save_path}")

# Use skimage for SSIM
from skimage.metrics import structural_similarity as sk_ssim
def calculate_ssim(img1, img2):
    img1_np = img1.detach().cpu().numpy().squeeze()
    img2_np = img2.detach().cpu().numpy().squeeze()
    ssim_val = sk_ssim(img1_np, img2_np, data_range=1.0)
    return ssim_val

def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2, reduction='mean')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

# -------------------- main --------------------
@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--pretrained", default="/teamspace/studios/this_studio/unetMRI/pt models")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--img", type=str, default=None, help="Optional single MRI slice")
    args = parser.parse_args()

    print(f"Loading models from {args.ckpt} on {DEVICE}...")
    c2, gen, intensity = load_models(args.ckpt, DEVICE)
    ae, u2net = load_frozen_ae_u2net(args.pretrained, DEVICE)

    if args.img is not None:
        # load single image
        x = T.ToTensor()(Image.open(args.img).convert("L").resize((512,512)))
        x = (x.unsqueeze(0) * 2 - 1).to(DEVICE)  # [-1,1]
        samples = [(x, "external")]
    else:
        # load dataset samples
        ds = MRIDataset("/teamspace/studios/this_studio/unetMRI/dataset/brain-tumor-mri-dataset/Training",
                        ['glioma','meningioma','notumor','pituitary'])
        samples = [ds[i] for i in range(min(args.num_samples, len(ds)))]

    for i, item in enumerate(samples):
        if args.img is not None:
            x, *_ = item
        else:
            x, *_ = item
            x = x.unsqueeze(0).to(DEVICE)

        latents, skip64 = ae.encoder(x*2-1)
        mask = (1 - torch.sigmoid(u2net(x.repeat(1,3,1,1))[0]))**2
        mask_32 = F.interpolate(mask,(32,32),mode='bilinear',align_corners=False)
        w = gen(x,intensity)
        lat_wm = latents + w*mask_32
        wm_img = ae.decoder(lat_wm, skip64)

        clean = (x*0.5+0.5).clamp(0,1)
        wm = (wm_img*0.5+0.5).clamp(0,1)
        diff_map = torch.abs(clean-wm)
        diff_map = (diff_map*0.5+0.5).clamp(0,1)

        save_images([clean[0], wm[0], diff_map[0]],
                    titles=["Clean","Watermarked","Difference Map"],
                    out_dir="viz_output",
                    filename=f"sample_{i}_clean_vs_wm.png")

        ssim = calculate_ssim(clean, wm)
        psnr = calculate_psnr(clean, wm)
        print(f"Sample {i} - SSIM: {ssim:.4f}, PSNR: {psnr:.2f} dB")


if __name__=="__main__":
    main()
