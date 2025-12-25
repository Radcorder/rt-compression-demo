# =================================================================
#  🚀 RT-Viewer Generator (Ultimate Edition)
#  機能: Variant 7修正 + Variant 8新配分 + バグ修正完全版
# =================================================================

import os, gzip, json, time
import numpy as np
import pydicom
from scipy.ndimage import zoom, binary_fill_holes, label

# --- 設定エリア ---
INPUT_ROOT = "./dicom_data"
OUTPUT_DIR = "."
# ----------------

def get_ct_volume_optimized(folder_path):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.dcm')]
    temp_list = []
    for f in files:
        try:
            d = pydicom.dcmread(f, stop_before_pixels=True)
            if d.Modality == 'CT': temp_list.append((float(d.ImagePositionPatient[2]), f))
        except: pass
    temp_list.sort(key=lambda x: x[0], reverse=True)
    if not temp_list: return None, None

    total = len(temp_list)
    center = total // 2
    selected_indices = list(range(max(0, center - 5), min(total, center + 5), 2))
    if not selected_indices: selected_indices = [0]
    
    vol = []
    d0 = pydicom.dcmread(temp_list[0][1])
    slope = float(getattr(d0, 'RescaleSlope', 1))
    intercept = float(getattr(d0, 'RescaleIntercept', 0))
    for idx in selected_indices:
        d = pydicom.dcmread(temp_list[idx][1])
        px = d.pixel_array * slope + intercept
        vol.append(px)
    return np.array(vol, dtype=np.int16), float(d0.PixelSpacing[0])

def save_bin(data, filename, folder):
    path = os.path.join(folder, filename)
    if os.path.exists(path): os.remove(path)
    with gzip.open(path, 'wb', compresslevel=6) as f: f.write(data.tobytes())
    return os.path.getsize(path)

# --- 画像処理ロジック ---

def _apply_background_fx(v_s, mode='none'):
    if mode == 'none': return v_s
    d, r, c = v_s.shape
    roi = np.zeros_like(v_s, dtype=bool)
    roi[:, r//4:r*3//4, c//4:c*3//4] = True
    
    tis = v_s > -300
    lb, n = label(tis)
    mask = np.zeros_like(v_s, dtype=bool)
    if n > 0:
        sz = np.bincount(lb.ravel()); sz[0]=0
        body = (lb == sz.argmax())
        for i in range(len(body)): mask[i] = binary_fill_holes(body[i])

    if mode == 'mask_only': v_s[~mask] = -1000
    elif mode == 'mosaic':
        v_s[~mask] = -1000
        target = mask & (~roi)
        if np.any(target):
            low_res = zoom(zoom(v_s, [1, 0.5, 0.5], order=1), [1, 2, 2], order=0)
            v_s[target] = low_res[:d,:r,:c][target]
    elif mode == 'silhouette':
        v_s[:] = -1000; v_s[mask] = 0
    return v_s

# ★ 8番: ご指定の4段階配分ロジック
def quantize_custom_5bit(v_s):
    # 範囲定義: [-1000, -200, 0, 100, 1000]
    R = [-1000, -200, 0, 100, 1000]
    
    # 階調配分: 
    # -1000~-200 (2階調) -> 0~2
    # -200~0     (5階調) -> 2~7
    # 0~100      (10階調)-> 7~17
    # 100~1000   (14階調)-> 17~31
    P = [0, 2, 7, 17, 31]

    cond = [
        v_s <= R[1],
        (v_s > R[1]) & (v_s <= R[2]),
        (v_s > R[2]) & (v_s <= R[3]),
        v_s > R[3]
    ]
    
    def f1(x): return (np.clip(x,R[0],R[1])-R[0])/(R[1]-R[0])*(P[1]-P[0])+P[0]
    def f2(x): return (x-R[1])/(R[2]-R[1])*(P[2]-P[1])+P[1]
    def f3(x): return (x-R[2])/(R[3]-R[2])*(P[3]-P[2])+P[2]
    def f4(x): return (np.clip(x,R[3],R[4])-R[3])/(R[4]-R[3])*(P[4]-P[3])+P[3]
    
    idx = np.piecewise(v_s, cond, [f1,f2,f3,f4]).round().astype(np.uint8)
    return idx, "lut_5bit_custom"

def quantize_nonlinear(v_s, bits):
    R = [-1000, -150, 350, 1000]
    if bits == 5: P = [0, 2, 13, 31]
    elif bits == 8: P = [0, 10, 245, 255] # 8bit用
    
    cond = [v_s<=R[1], (v_s>R[1])&(v_s<=R[2]), v_s>R[2]]
    def f1(x): return (np.clip(x,R[0],R[1])-R[0])/(R[1]-R[0])*(P[1]-P[0])+P[0]
    def f2(x): return (x-R[1])/(R[2]-R[1])*(P[2]-P[1])+P[1]
    def f3(x): return (np.clip(x,R[2],R[3])-R[2])/(R[3]-R[2])*(P[3]-P[2])+P[2]
    return np.piecewise(v_s, cond, [f1,f2,f3]).round().astype(np.uint8), f"lut_{bits}bit"

def to_linear_5bit_wide(vol):
    v_s = zoom(vol, [1, 0.5, 0.5], order=1)
    img = np.clip(v_s, -1000, 3096)
    img = (img - (-1000)) / (3096 - (-1000)) * 31
    return img.round().astype(np.uint8), "linear_5bit_wide", v_s.shape

def create_variant(vol, mode_type, bg_mode, downscale=True):
    if downscale: v_proc = zoom(vol, [1, 0.5, 0.5], order=1)
    else: v_proc = vol.copy()
    
    v_proc = _apply_background_fx(v_proc, mode=bg_mode)

    if mode_type == 16:
        return v_proc.astype(np.int16), "int16", v_proc.shape
    elif mode_type == '8bit_linear': 
        img = np.clip(v_proc, -150, 350)
        img = (img - (-150)) / 500 * 255
        return img.astype(np.uint8), "uint8", v_proc.shape
    elif mode_type == 'silhouette':
        # ★修正: シルエットが黒潰れしないよう、8bitスケールで0を表現する
        idx, _ = quantize_nonlinear(v_proc, 8) 
        return idx, "lut_8bit", v_proc.shape
    elif mode_type == 'custom_5bit':
        idx, tname = quantize_custom_5bit(v_proc)
        return idx, tname, v_proc.shape
    elif mode_type == 'linear_wide':
        return to_linear_5bit_wide(vol)
    else:
        idx, tname = quantize_nonlinear(v_proc, mode_type)
        return idx, tname, v_proc.shape

# --- メイン処理 ---
if not os.path.exists(INPUT_ROOT): print("Error: No dicom_data"); exit()
target_path = None
for root, _, files in os.walk(INPUT_ROOT):
    if any(f.endswith('.dcm') for f in files): target_path = root; break
if not target_path: print("Error: No DICOM found"); exit()

print(f"Target: {os.path.basename(target_path)}")
vol_orig, sp = get_ct_volume_optimized(target_path)
if vol_orig is None: exit()

manifest = {"variants": []}
tasks = []

tasks.append({"name": "1. Original (16bit/512px)", "func": create_variant, "args": [vol_orig, 16, 'none', False]})
tasks.append({"name": "2. Reference (16bit/256px)", "func": create_variant, "args": [vol_orig, 16, 'none', True]})
tasks.append({"name": "3. Linear 5-bit (Full 12bit Range)", "func": create_variant, "args": [vol_orig, 'linear_wide', 'none', True]})
tasks.append({"name": "4. Non-Linear 5-bit (Old Logic)", "func": create_variant, "args": [vol_orig, 5, 'none', True]})
tasks.append({"name": "5. Body Masked", "func": create_variant, "args": [vol_orig, 5, 'mask_only', True]})
tasks.append({"name": "6. Full Optimization (Old Logic)", "func": create_variant, "args": [vol_orig, 5, 'mosaic', True]})
tasks.append({"name": "7. Silhouette", "func": create_variant, "args": [vol_orig, 'silhouette', 'silhouette', True]})
tasks.append({"name": "8. Custom 5-bit (Your 4-Step Logic)", "func": create_variant, "args": [vol_orig, 'custom_5bit', 'mosaic', True]})

print(f"Generating {len(tasks)} variants...")
for i, task in enumerate(tasks):
    current_id = i + 1
    print(f"[{current_id}] {task['name']}...")
    data, v_type, dims = task["func"](*task["args"])
    filename = f"v{current_id:02d}_{v_type}.bin.gz"
    s_gz = save_bin(data, filename, OUTPUT_DIR)
    manifest["variants"].append({
        "id": current_id, "name": task['name'], "file": filename, "size_gz": s_gz,
        "type": v_type, "dims": [dims[0], dims[1], dims[2]]
    })

manifest["base_dims"] = manifest["variants"][0]["dims"]
json.dump(manifest, open(os.path.join(OUTPUT_DIR, "manifest.json"), 'w'), indent=2)
print("Done!")