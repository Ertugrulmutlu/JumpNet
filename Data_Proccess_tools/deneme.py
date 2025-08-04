import numpy as np

# 🔧 Buraya dosya yolunu ver
npz_path = r".\collected\Final_data\Geodashreelfinaldata.npz"

data = np.load(npz_path, allow_pickle=True)
entries = data["data"]

print(f"📦 Toplam örnek sayısı: {len(entries)}")
print("🧩 İlk 5 örnek:\n")

for i, entry in enumerate(entries[:5]):
    print(f"📍 Örnek #{i+1}")
    print(f"  image.shape     : {entry[0].shape}")
    print(f"  label           : {entry[1]}")
    print(f"  keys_raw        : {entry[2]}")
    print(f"  hold_duration   : {entry[3]}")
    print(f"  phase           : {entry[4]}")
    print(f"  frame_index     : {entry[5]}")
    print("-" * 40)
