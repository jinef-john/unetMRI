import torch

# Путь к твоему pth-файлу
import torch

pth = torch.load('E:/MRI_LOWMEM/Encoder_latent_64_128/autoencoder_epoch9.pth', map_location='cpu')
for k, v in pth.items():
    if 'conv1.weight' in k:
        print(f"{k}: shape={v.shape}")
        if v.shape[1] == 3:
            print("Модель ожидает RGB (3-канальные входы)")
        elif v.shape[1] == 1:
            print("Модель ожидает grayscale (1-канальные входы)")
        else:
            print(f"Неожиданное число каналов: {v.shape[1]}")
        break


pth_path = r"E:\MRI_LOWMEM\Encoder_latent_64_128\autoencoder_epoch9.pth"

state_dict = torch.load(pth_path, map_location="cpu")

print("="*40)
print(f"Список всех слоёв в {pth_path}:")
print("="*40)
for k, v in state_dict.items():
    print(f"{k:40s}  {tuple(v.shape)}")

print("\n=== Анализ skip-коннектов и декодера ===")
skip_candidates = []
decoder_candidates = []
for k, v in state_dict.items():
    lowk = k.lower()
    if "skip" in lowk or "up" in lowk or "cat" in lowk:
        skip_candidates.append((k, v.shape))
    if "decoder" in lowk or "deconv" in lowk or "dec" in lowk or "up" in lowk:
        decoder_candidates.append((k, v.shape))

if skip_candidates:
    print("\n-- Слои, похожие на skip connections:")
    for k, s in skip_candidates:
        print(f"  {k:35s} {tuple(s)}")
else:
    print("  Нет явных слоёв с названием skip/up/cat")

if decoder_candidates:
    print("\n-- Слои, похожие на decoder/deconv/upsample:")
    for k, s in decoder_candidates:
        print(f"  {k:35s} {tuple(s)}")
else:
    print("  Нет явных слоёв декодера/де-конволюции")

print("\n=== Подсказки по каналам ===")
# Прямо вывести input/output каналы для всех ConvTranspose2d/Conv2d слоёв декодера
for k, v in state_dict.items():
    if ".weight" in k and len(v.shape) == 4:
        # Conv2d/ConvTranspose2d: [out_channels, in_channels, k, k]
        print(f"{k:35s} out_channels={v.shape[0]}, in_channels={v.shape[1]}")

print("\n=== КРАТКАЯ РЕКОМЕНДАЦИЯ ===")
print("- Сравни out_channels/in_channels ConvTranspose2d в decoder с твоей архитектурой.")
print("- На вход декодеру нужно подавать тензор shape, равный in_channels первого deconv/decode слоя.")
print("- Если есть concat со skip — значит после cat каналов в 2 раза больше.")

print("\n=== Пример shape для входа в decoder ===")
first_deconv = None
for k, v in state_dict.items():
    if "deconv" in k and ".weight" in k:
        print(f"{k} --> in_channels: {v.shape[1]}, out_channels: {v.shape[0]}")
        if not first_deconv:
            first_deconv = v.shape[1]
if first_deconv:
    print(f"\n==> На вход decoder должен идти тензор shape: [B, {first_deconv}, H, W]")
