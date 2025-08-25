import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.models import efficientnet_b3
from torch.amp import autocast, GradScaler
import multiprocessing

# === CBAM модули ===
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
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
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out,_ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# === EfficientNet-B6 + CBAM (после блока 5) ===
class EfficientNetB3_CBAM_Bottleneck(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.base = efficientnet_b3(weights=None, num_classes=num_classes)
        self.features1 = nn.Sequential(*list(self.base.features.children())[:6])   # до блока 5 (output: [B,200,32,32])
        self.cbam = CBAM(136)
        self.features2 = nn.Sequential(*list(self.base.features.children())[6:])
        self.avgpool = self.base.avgpool
        self.classifier = self.base.classifier

    def forward(self, x):
        x = self.features1(x)
        x = self.cbam(x)
        x = self.features2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# === Основной скрипт ===
if __name__ == '__main__':
    multiprocessing.set_start_method("spawn", force=True)

    # ==== Пути и параметры ====
    TRAIN_DIR = r'E:\AFHQ\fullset'
    VAL_DIR = r'E:\AFHQ\val'
    OLD_PTH = r'E:\AFHQ_LOWMEM\C1-B3\C1EfficientNet_B3.pth'
    NEW_PTH = r'E:\AFHQ_LOWMEM\C1-B3-CBAM\C1EfficientNet_B3_CBAM.pth'
    NUM_CLASSES = 3
    IMG_SIZE = 512
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 2e-4

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_ds = datasets.ImageFolder(VAL_DIR, transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EfficientNetB3_CBAM_Bottleneck(num_classes=NUM_CLASSES)
    state_dict = torch.load(OLD_PTH, map_location='cpu')
    model.base.load_state_dict(state_dict, strict=False)  # грузим backbone
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPU w DataParallel")
        model = nn.DataParallel(model)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()
    best_val_acc = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast('cuda'):
                outputs = model(x)
                loss = criterion(outputs, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            preds = outputs.argmax(1)
            train_correct += (preds == y).sum().item()
            train_loss += loss.item() * x.size(0)
            train_total += x.size(0)
            del x, y, outputs, loss, preds
            torch.cuda.empty_cache()
        train_acc = train_correct / train_total
        train_loss /= train_total

        # --- Validation ---
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with autocast('cuda'):
                    outputs = model(x)
                    loss = criterion(outputs, y)
                preds = outputs.argmax(1)
                val_correct += (preds == y).sum().item()
                val_loss += loss.item() * x.size(0)
                val_total += x.size(0)
                del x, y, outputs, loss, preds
                torch.cuda.empty_cache()
        val_acc = val_correct / val_total
        val_loss /= val_total

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), NEW_PTH)
            print(f"Best model saved at epoch {epoch} (val acc={best_val_acc:.4f})")

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best model weights saved to {NEW_PTH}")
