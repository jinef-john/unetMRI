import torch

pth_path = r'E:\MRI_LOWMEM\Encoder_latent_64_128\autoencoder_epoch9.pth'
state = torch.load(pth_path, map_location='cpu')

print("All keys in checkpoint:")
for k in state:
    print(k)

print("\n== Shapes of first conv and last deconv (если есть):")

for k in state:
    if 'conv1.weight' in k or 'features.0.0.weight' in k:
        print(f"{k}: {state[k].shape}")
    if 'deconv5.weight' in k or 'final.weight' in k:
        print(f"{k}: {state[k].shape}")
