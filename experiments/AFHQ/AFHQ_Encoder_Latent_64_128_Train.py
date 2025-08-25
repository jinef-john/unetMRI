import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from tqdm import tqdm
import numpy as np
from pytorch_msssim import ssim  # <-- должно быть именно это!
from torchvision.models import vgg16
import torch.nn.functional as F

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super().__init__()
        vgg = vgg16(weights='IMAGENET1K_V1').features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.resize = resize

    def forward(self, x, y):
        # x, y в [0,1], [B,3,H,W]
        if self.resize:
            x = F.interpolate(x, size=(224,224), mode='bilinear', align_corners=False)
            y = F.interpolate(y, size=(224,224), mode='bilinear', align_corners=False)
        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)
        return F.l1_loss(x_vgg, y_vgg)

# --- CBAM Attention ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out

# --- Encoder ---
class Encoder(nn.Module):
    def __init__(self, bottleneck_channels=1024):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)  # 512 -> 512, 3->64
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)  # 512 -> 256, 64->128
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)  # 256 -> 128, 128->256
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1)  # 128 -> 64, 256->512
        self.conv5 = nn.Conv2d(512, bottleneck_channels, 4, 2, 1)  # 64 -> 32, 512->1024
        self.cbam = CBAM(bottleneck_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))  # 512x512
        x2 = self.relu(self.conv2(x1))  # 256x256
        x3 = self.relu(self.conv3(x2))  # 128x128 <-- skip128
        x4 = self.relu(self.conv4(x3))  # 64x64   <-- skip64
        x5 = self.relu(self.conv5(x4))  # 32x32   <-- bottleneck
        latent = self.cbam(x5)  # 32x32
        return latent, x4, x3  # latent, skip64, skip128

# --- Decoder ---
class Decoder(nn.Module):
    def __init__(self, bottleneck_channels=1024):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(bottleneck_channels, 512, 4, 2, 1)      # 32 -> 64
        self.deconv2 = nn.ConvTranspose2d(512+512, 256, 4, 2, 1)                  # 64 -> 128 (+skip64)
        self.deconv3 = nn.ConvTranspose2d(256+256, 128, 4, 2, 1)                  # 128 -> 256 (+skip128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, 4, 2, 1)                       # 256 -> 512
        self.deconv5 = nn.Conv2d(64, 3, 3, 1, 1)
        self.relu = nn.ReLU(inplace=False)
        self.tanh = nn.Tanh()

    def forward(self, latent, skip64, skip128):
        x = self.relu(self.deconv1(latent))         # [B, 512, 64, 64]
        x = torch.cat([x, skip64], dim=1)           # [B, 1024, 64, 64]
        x = self.relu(self.deconv2(x))              # [B, 256, 128, 128]
        x = torch.cat([x, skip128], dim=1)          # [B, 512, 128, 128]
        x = self.relu(self.deconv3(x))              # [B, 128, 256, 256]
        x = self.relu(self.deconv4(x))              # [B, 64, 512, 512]
        x = self.tanh(self.deconv5(x))
        return (x + 1) / 2   # [-1,1] → [0,1]



# --- AutoEncoder ---
class AutoEncoder(nn.Module):
    def __init__(self, bottleneck_channels=1024):
        super().__init__()
        self.encoder = Encoder(bottleneck_channels)
        self.decoder = Decoder(bottleneck_channels)
    def forward(self, x):
        latent, skip64, skip128 = self.encoder(x)
        x_hat = self.decoder(latent, skip64, skip128)
        return x_hat
    def decode_latents(self, latent, skip64, skip128):
        return self.decoder(latent, skip64, skip128)


# --- Метрики ---
from skimage.metrics import structural_similarity as skimage_ssim
import inspect

def compute_ssim(img1, img2):
    ssim_vals = []
    for i in range(img1.size(0)):
        arr1 = img1[i].detach().cpu().numpy().transpose(1,2,0)
        arr2 = img2[i].detach().cpu().numpy().transpose(1,2,0)
        # Проверяем, какой аргумент поддерживается
        params = inspect.signature(skimage_ssim).parameters
        if 'channel_axis' in params:
            val = skimage_ssim(arr1, arr2, channel_axis=2, data_range=1.0)
        elif 'multichannel' in params:
            val = skimage_ssim(arr1, arr2, multichannel=True, data_range=1.0)
        else:
            val = skimage_ssim(arr1, arr2, data_range=1.0)
        ssim_vals.append(val)
    return float(np.mean(ssim_vals))

def compute_psnr(img1, img2):
    mse = ((img1 - img2) ** 2).mean().item()
    if mse == 0:
        return 100
    return 20 * np.log10(1.0 / np.sqrt(mse))

def pick_free_gpu():
    import subprocess
    try:
        result = subprocess.check_output(
            'nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits',
            shell=True)
        memory_free = [int(x) for x in result.decode('utf-8').strip().split('\n')]
        idx = int(np.argmax(memory_free))
        print(f"[INFO] Using GPU:{idx} (free mem: {memory_free[idx]} MB)")
        print(f"Device: {torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device())})")
        print(f"Allocated memory: {torch.cuda.memory_allocated() // 1024 ** 2} MB")
        print(f"Reserved memory: {torch.cuda.memory_reserved() // 1024 ** 2} MB")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(idx)
    except Exception as e:
        print(f"[WARN] Could not automatically select GPU: {e}")
        pass


# --- Точка входа ---
def main():
    DATA_DIR = r'E:\AFHQ\fullset'
    MODEL_SAVE_DIR = r'E:\AFHQ\\models_latent_64_128'
    LOG_FILE = r'E:\AFHQ\models_latent_64_128\trainlog.csv'
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # Your bottleneck and skip shapes
    B = 2  # batch size for test
    latent = torch.randn(B, 1024, 32, 32)
    skip64 = torch.randn(B, 512, 64, 64)
    skip128 = torch.randn(B, 256, 128, 128)

    decoder = Decoder(bottleneck_channels=1024)
    out = decoder(latent, skip64, skip128)
    print(out.shape)  # [2, 3, 512, 512]

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=14, shuffle=True, num_workers=4, pin_memory=True)

    pick_free_gpu()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoEncoder(bottleneck_channels=1024).to(device)

    # --- DataParallel для 2+ GPU ---
    #if torch.cuda.device_count() > 1:
    #    print(f"Using {torch.cuda.device_count()} GPUs!")
    #    model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    criterion = torch.nn.MSELoss()
    epochs = 10

    perceptual_loss = VGGPerceptualLoss().to(device)

    with open(LOG_FILE, "w") as f:
        f.write("epoch,batch,loss,ssim,psnr\n")

        for epoch in range(epochs):
            model.train()
            running_loss = 0
            running_ssim = 0
            running_psnr = 0
            n_samples = 0

            with tqdm(total=len(loader), desc=f"Epoch {epoch + 1}/{epochs}", ncols=100) as pbar:
                for batch_idx, (x, _) in enumerate(loader):
                    x = x.to(device)
                    out = model(x)
                    mse = ((out - x) ** 2).mean()
                    perc = perceptual_loss(out, x)
                    ssim_loss = 1 - ssim(out, x, data_range=1.0, size_average=True)
                    loss = mse + 0.1 * perc + 0.1 * ssim_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * x.size(0)
                    n_samples += x.size(0)

                    # Метрики для текущего батча
                    ssim_score = compute_ssim(out, x)
                    psnr_score = compute_psnr(out, x)
                    running_ssim += ssim_score * x.size(0)
                    running_psnr += psnr_score * x.size(0)

                    # Лог в файл
                    f.write(f"{epoch + 1},{batch_idx + 1},{loss.item():.6f},{ssim_score:.4f},{psnr_score:.2f}\n")
                    f.flush()

                    # Обновляем прогресс-бар метриками
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'ssim': f'{ssim_score:.3f}',
                        'psnr': f'{psnr_score:.1f}'
                    })
                    pbar.update(1)

            # Итоги по эпохе
            epoch_loss = running_loss / n_samples
            epoch_ssim = running_ssim / n_samples
            epoch_psnr = running_psnr / n_samples
            f.write(f"epoch_summary,{epoch + 1},{epoch_loss:.6f},{epoch_ssim:.4f},{epoch_psnr:.2f}\n")
            f.flush()

            # Сохраняем модель
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, f'autoencoder_epoch{epoch + 1}.pth'))


if __name__ == "__main__":
    main()
