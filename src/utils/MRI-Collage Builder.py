import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from sklearn.decomposition import PCA
import subprocess
from PIL import Image, ImageDraw, ImageFont
from MRI_Encoder_Latent_64_128_Train import Decoder  # Импорт декодера

from skimage.metrics import structural_similarity as ssim
from skimage.feature import canny
from skimage.color import rgb2gray


# ========== GPU AUTOPICK ==========
def pick_free_gpu():
    try:
        result = subprocess.check_output(
            'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits',
            shell=True)
        memory_used = [int(x) for x in result.decode('utf-8').strip().split('\n')]
        idx = int(np.argmin(memory_used))
        print(f"[INFO] Using GPU:{idx} (used mem: {memory_used[idx]} MB)")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(idx)
        return idx
    except Exception as e:
        print(f"[WARN] Could not auto-select GPU: {e}")
        return 0

pick_free_gpu()

# ===== ПУТИ =====
NPZ_DIR = r'E:\MRI_LOWMEM\MRI-NPZ_latent_skip64_skip128'
IMG_DIR = r'E:\MRI_LOWMEM\Training'
COLLAGE_DIR = r'E:\MRI_LOWMEM\Collages_latent_skip64_skip128'
MODEL_PATH = r'E:\MRI_LOWMEM\Encoder_latent_64_128\autoencoder_epoch9.pth'  # Подставь свой путь

box_height = 32  # txt height
row1_labels = ["Original", "Recon"]
row2_labels = ["Latent-Mean", "Latent-Max", "Latent-Highest Energy RGB", "Latent-Principal Component Analysis-1024ch"]
row3_labels = ["Skip64-Mean", "Skip64-Max", "Skip64-Highest Energy RGB", "Skip64-Principal Component Analysis-512ch"]
row4_labels = ["Skip128-Mean", "Skip128-Max", "Skip128-Highest Energy RGB", "Skip128-Principal Component Analysis-256ch"]

def add_caption(img, text, font=None, box_height=32):
    w, h = img.size
    new_img = Image.new('RGB', (w, h + box_height), (255, 255, 255))
    new_img.paste(img, (0, 0))
    draw = ImageDraw.Draw(new_img)
    if font is None:
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
    # Вместо textsize используем textbbox для новых PIL
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        # Для старых PIL fallback на textsize
        text_w, text_h = draw.textsize(text, font=font)
    draw.text(((w - text_w) // 2, h + (box_height - text_h) // 2), text, fill=(0,0,0), font=font)
    return new_img

# ====== ФУНКЦИИ ======


def vis_feat(feat, method='mean', out_size=512):
    v = feat.squeeze().float().cpu()
    if method == 'mean':
        arr = v.mean(dim=0).numpy()
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
        return Image.fromarray((arr*255).astype(np.uint8)).resize((out_size, out_size))
    elif method == 'max':
        arr = v.max(dim=0)[0].numpy()
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
        return Image.fromarray((arr*255).astype(np.uint8)).resize((out_size, out_size))
    else:
        raise ValueError(f"Unknown method: {method}")

def vis_feat_best_rgb(feat, out_size=512):
    v = feat.squeeze().float().cpu().numpy()  # Теперь сразу numpy!
    C, H, W = v.shape
    stds = np.abs(v).mean(axis=(1,2))  # std/energy по каналам
    idxs = np.argsort(stds)[-3:][::-1]
    arr = v[idxs, :, :].copy()  # <--- индексация по numpy, .copy() для гарантии
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    arr = np.transpose(arr, (1,2,0))
    img = Image.fromarray((arr*255).astype(np.uint8)).resize((out_size, out_size))
    return img


def ensure_rgb(img):
    if img.mode != 'RGB':
        return img.convert('RGB')
    return img



def ssim_heatmap(orig, rec):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    orig_np = np.array(orig).astype(np.float32) / 255.0
    rec_np = np.array(rec).astype(np.float32) / 255.0
    if orig_np.ndim == 2:
        orig_np = np.stack([orig_np]*3, axis=2)
    if rec_np.ndim == 2:
        rec_np = np.stack([rec_np]*3, axis=2)
    if orig_np.shape[2] != 3 or rec_np.shape[2] != 3:
        raise ValueError(f"Input images must be HxWx3, got {orig_np.shape}, {rec_np.shape}")

    H, W = orig_np.shape[0], orig_np.shape[1]
    ssim_map = []
    for ch in range(3):
        ssim_map_ch, _ = ssim(orig_np[:,:,ch], rec_np[:,:,ch], data_range=1.0, full=True)
        # Фикс: если скаляр — превращаем в поле
        if np.isscalar(ssim_map_ch):
            ssim_map_ch = np.ones((H, W)) * ssim_map_ch
        elif ssim_map_ch.shape != (H, W):
            from skimage.transform import resize
            ssim_map_ch = resize(ssim_map_ch, (H, W), preserve_range=True)
        ssim_map.append(ssim_map_ch)
    ssim_map = np.mean(np.stack(ssim_map, axis=0), axis=0)
    error_map = 1.0 - ssim_map

    # should increase contrast if needed
    amplif = 7.0  # arbtrarily amplify SSIM mismatches with original, but expect very little to show up
    error_map = np.clip(error_map * amplif, 0, 1)

    # Выбери палитру: 'seismic' или 'bwr' или любую другую с ярким контрастом
    cmap = plt.colormaps['seismic']
    rgba_map = cmap(error_map)
    color_map = (rgba_map[:, :, :3] * 255).astype(np.uint8)
    error_img = Image.fromarray(color_map).convert('RGB').resize((512,512))
    return error_img




def edge_map(img_pil):
    img_np = np.array(img_pil).astype(np.float32) / 255.0
    gray = rgb2gray(img_np)
    edges = canny(gray, sigma=2.0)
    edges_img = (edges * 255).astype(np.uint8)
    edge_pil = Image.fromarray(edges_img).convert('RGB').resize((512,512))
    return edge_pil


def vis_feat_pca(feat, out_size=512):
    v = feat.squeeze().float().cpu()
    C, H, W = v.shape
    arr = v.reshape(C, -1).T  # [H*W, C]
    pca = PCA(n_components=3)
    arr_pca = pca.fit_transform(arr)
    arr_pca -= arr_pca.min(axis=0)
    arr_pca /= arr_pca.max(axis=0) + 1e-6
    arr_pca = arr_pca.reshape(H, W, 3)
    arr_img = (arr_pca * 255).astype(np.uint8)
    img = Image.fromarray(arr_img).resize((out_size, out_size))
    return img


def main():
    # --- create folders ---
    for cls in ['glioma', 'meningioma', 'notumor', 'pituitary']:
        os.makedirs(os.path.join(COLLAGE_DIR, cls), exist_ok=True)

    # --- decoder ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    decoder = Decoder().to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    if 'module.decoder.deconv1.weight' in state:
        decoder.load_state_dict({k.replace('module.decoder.', ''): v for k, v in state.items() if k.startswith('module.decoder.')})
    elif 'decoder.deconv1.weight' in state:
        decoder.load_state_dict({k.replace('decoder.', ''): v for k, v in state.items() if k.startswith('decoder.')})
    else:
        decoder.load_state_dict(state)
    decoder.eval()

    to_pil = transforms.ToPILImage()

    # --- file search ---
    for cls in ['glioma', 'meningioma', 'notumor', 'pituitary']:
        npz_cls_dir = os.path.join(NPZ_DIR, cls)
        img_cls_dir = os.path.join(IMG_DIR, cls)
        out_cls_dir = os.path.join(COLLAGE_DIR, cls)


        for file in tqdm(os.listdir(npz_cls_dir), desc=f'Processing {cls}'):
            if not file.endswith('.npz'):
                continue
            npz_path = os.path.join(npz_cls_dir, file)
            base_name = os.path.splitext(file)[0]
            # ищем оригинал
            img_path = os.path.join(img_cls_dir, base_name + '.jpg')
            if not os.path.exists(img_path):
                img_path = os.path.join(img_cls_dir, base_name + '.png')
            if not os.path.exists(img_path):
                print(f"[WARN] Original not found for {file}")
                continue
            # --- Load NPZ ---
            arr = np.load(npz_path)
            latent = torch.from_numpy(arr['latent']).unsqueeze(0).to(device)
            skip64 = torch.from_numpy(arr['skip64']).unsqueeze(0).to(device)
            skip128 = torch.from_numpy(arr['skip128']).unsqueeze(0).to(device)
            # --- Decode ---
            with torch.no_grad():
                rec = decoder(latent, skip64, skip128).squeeze(0).cpu()
            rec_img = to_pil(rec.clamp(0,1))
            # --- Оригинал ---
            orig = Image.open(img_path).convert('RGB').resize((512,512))
            # --- Визуализации ---
            # latent
            # latent
            latent_mean = vis_feat(latent[0], 'mean')
            latent_max = vis_feat(latent[0], 'max')
            latent_best_rgb = vis_feat_best_rgb(latent[0])
            latent_pca = vis_feat_pca(latent[0])
            # skip64
            skip64_mean = vis_feat(skip64[0], 'mean')
            skip64_max = vis_feat(skip64[0], 'max')
            skip64_best_rgb = vis_feat_best_rgb(skip64[0])
            skip64_pca = vis_feat_pca(skip64[0])
            # skip128
            skip128_mean = vis_feat(skip128[0], 'mean')
            skip128_max = vis_feat(skip128[0], 'max')
            skip128_best_rgb = vis_feat_best_rgb(skip128[0])
            skip128_pca = vis_feat_pca(skip128[0])

            # --- Убедись, что оба изображения RGB ---
            orig = ensure_rgb(orig)
            rec_img = ensure_rgb(rec_img)

            # --- Считай SSIM-карту и Edge map только один раз ---
            ssim_img = ssim_heatmap(orig, rec_img)
            ssim_img = add_caption(ssim_img, "SSIM-Heatmap (Blue = Full Original Recon Match)", box_height=box_height)

            edge_img = edge_map(rec_img)
            edge_img = add_caption(edge_img, "Reconstruction Edges", box_height=box_height)

            # --- Формируй первый ряд с подписями ---
            row1_imgs = [
                add_caption(orig, "Original", box_height=box_height),
                add_caption(rec_img, "Recon", box_height=box_height),
                ssim_img,
                edge_img
            ]

            # --- Collage Assemble ---
            # --- Rows with signatures ---

            row2_imgs = [add_caption(img, row2_labels[idx], box_height=box_height)
                         for idx, img in enumerate(
                    [latent_mean.convert('RGB'), latent_max.convert('RGB'), latent_best_rgb, latent_pca])]
            row3_imgs = [add_caption(img, row3_labels[idx], box_height=box_height)
                         for idx, img in enumerate(
                    [skip64_mean.convert('RGB'), skip64_max.convert('RGB'), skip64_best_rgb, skip64_pca])]
            row4_imgs = [add_caption(img, row4_labels[idx], box_height=box_height)
                         for idx, img in enumerate(
                    [skip128_mean.convert('RGB'), skip128_max.convert('RGB'), skip128_best_rgb, skip128_pca])]

            # --- Total colacge sz ---
            row_width = 512 * 4  # 4 pics in a row (except the first — there only 2)
            collage_w = row_width
            collage_h = (512 + box_height) * 4  # 4 rows

            collage = Image.new('RGB', (collage_w, collage_h), (255, 255, 255))
            # --- Inserting rows into final collage ---
            # 1st row is: original + recon (2 white cells)
            for idx, img in enumerate(row1_imgs):
                collage.paste(img, (512 * idx, 0))
            # 2nd row - latents
            for idx, img in enumerate(row2_imgs):
                collage.paste(img, (512 * idx, 512 + box_height))
            # 3rd row - skip64 visuals
            for idx, img in enumerate(row3_imgs):
                collage.paste(img, (512 * idx, 2 * (512 + box_height)))
            # 4th row - skip 128 visuals
            for idx, img in enumerate(row4_imgs):
                collage.paste(img, (512 * idx, 3 * (512 + box_height)))

            # --- empty cells in first row ---


            collage_path = os.path.join(out_cls_dir, base_name + '_collage.jpg')
            collage.save(collage_path, quality=95)

if __name__ == "__main__":
    main()
