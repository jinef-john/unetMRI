import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn

from ASCC_Encoder_Latent_64_128_Train import Encoder  # Adjust as needed

# ====== CONFIG ======
DATA_DIR = r'E:\ASCC_LOWMEM\Printing'
SAVE_DIR = r'E:\ASCC_LOWMEM\MRI-NPZ_latent_skip64_skip128'
MODEL_PATH = r'E:\ASCC_LOWMEM\Encoder_latent_64_128\autoencoder_epoch10.pth'
BATCH_SIZE = 32  # Set for good GPU utilization
NUM_WORKERS = 4  # DataLoader workers, not process workers

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

def main():

    # ====== SETUP ======
    os.makedirs(SAVE_DIR, exist_ok=True)
    for cls in ['Extrude', 'Idle', 'Inner_wall_Anomalous', 'Inner_wall_Normal', 'Outer_wall_Anomalous', 'Outer_wall_Normal', 'Print']:
        os.makedirs(os.path.join(SAVE_DIR, cls), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    full_state_dict = torch.load(MODEL_PATH, map_location=device)

    # If it’s a DataParallel model, adjust the keys
    if 'module.encoder.conv1.weight' in full_state_dict:
        encoder_state_dict = {k.replace('module.encoder.', ''): v
                              for k, v in full_state_dict.items() if k.startswith('module.encoder.')}
    elif 'encoder.conv1.weight' in full_state_dict:
        encoder_state_dict = {k.replace('encoder.', ''): v
                              for k, v in full_state_dict.items() if k.startswith('encoder.')}
    else:
        raise RuntimeError("Encoder weights not found in checkpoint!")

    encoder = Encoder().to(device)
    encoder.load_state_dict(encoder_state_dict)
    encoder.eval()

    if torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs')
        encoder = nn.DataParallel(encoder)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    # ====== MAIN LOOP ======
    counter = {cls: 0 for cls in idx_to_class.values()}
    fail_counter = 0
    with torch.no_grad():
        for batch_idx, (img_tensor, labels) in enumerate(tqdm(loader, desc='Encoding')):
            img_tensor = img_tensor.to(device)
            latents, skip64s, skip128s = encoder(img_tensor)  # [B, 1024, 32, 32], [B, 512, 64, 64], [B, 256, 128, 128]
            latents_np = latents.cpu().numpy()
            skip64s_np = skip64s.cpu().numpy()
            skip128s_np = skip128s.cpu().numpy()

            # Save each image in the batch
            for i in range(img_tensor.size(0)):
                cls_name = idx_to_class[labels[i].item()]
                orig_path, _ = dataset.samples[batch_idx * BATCH_SIZE + i]
                fname = os.path.splitext(os.path.basename(orig_path))[0] + '.npz'
                outpath = os.path.join(SAVE_DIR, cls_name, fname)
                try:
                    np.savez_compressed(outpath,
                                        latent=latents_np[i],
                                        skip64=skip64s_np[i],
                                        skip128=skip128s_np[i]
                                        )

                    # Validation
                    arr = np.load(outpath)
                    assert arr['latent'].shape == (1024, 32, 32), f"latent shape wrong in {fname}"
                    assert arr['skip64'].shape == (512, 64, 64), f"skip64 shape wrong in {fname}"
                    assert arr['skip128'].shape == (256, 128, 128), f"skip128 shape wrong in {fname}"
                    counter[cls_name] += 1
                except Exception as e:
                    print(f"[ERROR] {fname}: {e}")
                    fail_counter += 1

    print("Done!")
    for cls in counter:
        print(f"{cls}: {counter[cls]} images saved/validated")
    print(f"Failed: {fail_counter}")

    # (Optional) Save counts to file:
    with open(os.path.join(SAVE_DIR, "npz_counts.txt"), "w") as f:
        for cls in counter:
            f.write(f"{cls}: {counter[cls]}\n")
        f.write(f"Failed: {fail_counter}\n")

if __name__ == "__main__":
    main()
