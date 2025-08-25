import os
import re

ROOT = r"E:\MRI_LOWMEM\u2net_masks_png"
pattern = re.compile(r"(.*)\.nii_slice\d{3}\.png$", re.IGNORECASE)

total_renamed = 0
folder_stats = {}

for dirpath, dirnames, filenames in os.walk(ROOT):
    renamed_here = 0
    for fname in filenames:
        m = pattern.match(fname)
        if m:
            new_name = m.group(1) + ".png"
            src = os.path.join(dirpath, fname)
            dst = os.path.join(dirpath, new_name)
            if os.path.exists(dst):
                print(f"SKIP (already exists): {dst}")
            else:
                os.rename(src, dst)
                renamed_here += 1
                print(f"Renamed: {src} -> {dst}")
    if renamed_here > 0:
        print(f"[{dirpath}] Renamed: {renamed_here}")
    folder_stats[dirpath] = renamed_here
    total_renamed += renamed_here

print("\n====== Folder-wise rename stats ======")
for folder, cnt in folder_stats.items():
    if cnt > 0:
        print(f"{folder}: {cnt}")

print(f"\nTOTAL FILES RENAMED: {total_renamed}")
