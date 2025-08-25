import os
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from timm.data import Mixup
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
from tqdm import tqdm
from timm.loss import SoftTargetCrossEntropy
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torch.amp import autocast


if __name__ == '__main__':
    # --------- Paths ---------
    TRAIN_DIR = r'E:\ASCC_LOWMEM\Printing'
    VAL_DIR = r'E:\ASCC_LOWMEM\Printing_val'
    MODEL_PATH = r'E:\ASCC_LOWMEM\C1-B3\ASCC-C1EfficientNet_B3.pth'
    LOG_CSV = r'E:\ASCC_LOWMEM\C1-B3\classifierEFN_B3_train_log.csv'
    CONF_MAT_DIR = r'E:\ASCC_LOWMEM\C1-B3\confusion_matrices'

    os.makedirs(CONF_MAT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # --------- Hyperparameters ---------
    BATCH_SIZE = 64
    NUM_CLASSES = 7
    NUM_EPOCHS = 10
    LEARNING_RATE = 2e-4
    TARGET_ACC = 0.9875
    #IMG_SIZE = 512  # EfficientNet-B3 default

    # --------- Device ---------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --------- Augmentations ---------

    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((120, 160)),
        transforms.RandomResizedCrop((120, 160), scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=10, shear=15, scale=(0.8, 1.2), translate=(0.08, 0.08)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((120, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # --------- Datasets & Loaders ---------
    train_ds = ImageFolder(TRAIN_DIR, transform=train_transform)
    val_ds = ImageFolder(VAL_DIR, transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    class_names = train_ds.classes
    print("Class names:", class_names)


    scaler = torch.amp.GradScaler()


    def convert_first_conv_to_grayscale(model):
        """Converts the first conv layer to accept 1 channel using the mean of pretrained RGB weights."""
        conv = model.features[0][0]
        if conv.in_channels == 3:
            with torch.no_grad():
                # Average across RGB channels to get a single grayscale kernel
                gray_weights = conv.weight.mean(dim=1, keepdim=True)  # shape [40, 1, 3, 3]
                new_conv = nn.Conv2d(1, 40, kernel_size=3, stride=2, padding=1, bias=False)
                new_conv.weight.copy_(gray_weights)
                model.features[0][0] = new_conv
                print("First conv layer adapted from RGB to grayscale (weights averaged).")
        else:
            print("First conv already has", conv.in_channels, "channels.")


    # Example usage after model creation:
    weights = EfficientNet_B3_Weights.DEFAULT
    model = efficientnet_b3(weights=weights)
    convert_first_conv_to_grayscale(model)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)


    print("Torchvision EfficientNet-B3 ImageNet weights,", NUM_CLASSES, "classes")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # ---------End Model ---------

    # --------- Mixup/CutMix ---------
    mixup_fn = Mixup(
        mixup_alpha=0.2,
        cutmix_alpha=0.5,
        prob=0.45,
        switch_prob=0.5,
        mode='batch',
        label_smoothing=0.1,
        num_classes=NUM_CLASSES,
    )

    # --------- Optimizer & LR Scheduler ---------
    criterion = nn.CrossEntropyLoss()
    criterion_soft = SoftTargetCrossEntropy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    best_val_acc = 0
    log_records = []

    # --------- Datasets & Loaders ---------
    train_ds = ImageFolder(TRAIN_DIR, transform=train_transform)
    val_ds = ImageFolder(VAL_DIR, transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    class_names = train_ds.classes
    print("=================================\n")
    print("Class names:", class_names)

    # ---- MAPPING CHK ----
    print("\n[CLASS-TO-IDX MAPPING]")
    print(train_ds.class_to_idx)
    print("Train class mapping: ", {v: k for k, v in train_ds.class_to_idx.items()})

    # CHecking all files and class mapping
    #for idx in range(min(16000, len(train_ds.samples))):
     #   img_path, label_idx = train_ds.samples[idx]
      #  label_str = class_names[label_idx]
       # print(f"{os.path.basename(img_path)}  -->  {label_idx} ({label_str})")
    print("--------------------------------------------------")


    # --------- Training loop ---------
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        train_targets, train_preds = [], []

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [train]", ncols=110):
            #(print(y), print(np.bincount(y.cpu().numpy())))
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            with autocast('cuda'):
                #mixup_fn = None # disables mixup cutmix
                if mixup_fn is not None:
                    x, y_mix = mixup_fn(x, y)
                    outputs = model(x)
                    loss = criterion_soft(outputs, y_mix)  # <<< используем soft-критерий
                    preds = outputs.argmax(1)
                    train_targets.extend(y.cpu().numpy())
                    train_preds.extend(preds.cpu().numpy())
                    train_correct += (preds == y).sum().item()
                else:
                    outputs = model(x)
                    loss = criterion(outputs, y)
                    preds = outputs.argmax(1)
                    train_targets.extend(y.cpu().numpy())
                    train_preds.extend(preds.cpu().numpy())
                    train_correct += (preds == y).sum().item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * x.size(0)
            train_total += x.size(0)
            # 3: очистка памяти
            del x, y, outputs, loss, preds
            torch.cuda.empty_cache()

        train_acc = train_correct / train_total
        train_loss /= train_total

        # --- Validation ---
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        val_targets, val_preds = [], []
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [val]  ", ncols=110):
                x, y = x.to(device), y.to(device)
                with autocast('cuda'):  # 1: AMP for val
                    outputs = model(x)
                    loss = criterion(outputs, y)
                val_loss += loss.item() * x.size(0)
                preds = outputs.argmax(1)
                val_correct += (preds == y).sum().item()
                val_total += x.size(0)
                val_targets.extend(y.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())
                # 3: очистка памяти
                del x, y, outputs, loss, preds
                torch.cuda.empty_cache()
        val_acc = val_correct / val_total
        val_loss /= val_total

        # LR scheduler step
        scheduler.step(val_acc)

        # Confusion matrix
        cm = confusion_matrix(val_targets, val_preds, labels=list(range(NUM_CLASSES)))
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_fn = f'confmat_epoch{epoch:02d}_{datetime.now().strftime("%Y%m%d-%H%M%S")}.csv'
        cm_df.to_csv(os.path.join(CONF_MAT_DIR, cm_fn))

        print(f"\nEpoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        print("Confusion matrix (validation):")
        print(cm_df)

        # Logging
        log_records.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), MODEL_PATH)
            else:
                torch.save(model.state_dict(), MODEL_PATH)
            print(f"Best model saved (val acc={best_val_acc:.4f})")

        # Early stopping
        if val_acc >= TARGET_ACC and train_acc >= TARGET_ACC:
            print(f"\nBoth Target and Train accuracy are now at {TARGET_ACC*100:.2f}% and desired outcome is reached. Stopping training.")
            break

    # --------- Save accuracy/log CSV ---------
    df_log = pd.DataFrame(log_records)
    df_log.to_csv(LOG_CSV, index=False)

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best model weights saved to {MODEL_PATH}")

    # --------- Freeze the model for future use ---------
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print("Classifier 1 (EfficientNet-B3) is now frozen and ready for inference.")

