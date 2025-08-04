# Data_preproccer_first_press_pair_release_diag.py
import numpy as np
import os
from datetime import datetime

# === SETTINGS ===
INPUT_NPZ = r".\collected\Geodash2.npz"  # raw input file
TARGET_KEY = "w"
OUTPUT_DIR = "collected"
# ================

def recursive_unwrap(x):
    """Recursively unwraps numpy arrays or objects to native types."""
    while isinstance(x, np.ndarray) and (x.ndim == 0 or x.dtype == object):
        try:
            if x.shape == () or (hasattr(x, "tolist") and isinstance(x.tolist(), list) and len(x.tolist()) == 1):
                x = x.item()
                continue
        except Exception:
            pass
        break
    return x

def scalarize(v):
    """Converts single-element arrays to scalars, otherwise returns the value."""
    try:
        if isinstance(v, np.ndarray):
            if v.size == 1:
                return float(v.item())
            return float(v.flatten()[0])
    except:
        pass
    return v

def normalize_phase(p):
    """Normalizes the phase value to a lowercase string."""
    if isinstance(p, bytes):
        return p.decode("utf-8", errors="ignore").strip().lower()
    if isinstance(p, str):
        return p.strip().lower()
    return str(p).strip().lower()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    raw = np.load(INPUT_NPZ, allow_pickle=True)
    data_raw = raw.get("data")
    keys_raw = raw.get("keys")
    if data_raw is None or keys_raw is None:
        raise RuntimeError("No 'data' or 'keys' found in input .npz file.")
    keys = list(keys_raw)
    if TARGET_KEY not in keys:
        raise RuntimeError(f"'{TARGET_KEY}' not found in keys: {keys}")
    ki = keys.index(TARGET_KEY)

    # Unwrap data container
    data_unwrapped = recursive_unwrap(data_raw)
    if isinstance(data_unwrapped, (list, tuple)):
        entries = list(data_unwrapped)
    elif isinstance(data_unwrapped, np.ndarray):
        try:
            entries = list(data_unwrapped.tolist())
        except:
            entries = list(data_unwrapped)
    else:
        raise RuntimeError(f"Unexpected data structure: {type(data_unwrapped)}")

    out_entries = []
    pending_press = None  # (image, multi_hot, frame_idx)
    stats = {
        "total": 0,
        "press_seen": 0,
        "release_seen": 0,
        "paired": 0,
        "unmatched_final_press": 0,
        "malformed": 0,
    }

    for idx, entry in enumerate(entries):
        stats["total"] += 1
        entry = recursive_unwrap(entry)
        if not (isinstance(entry, (list, tuple)) and len(entry) >= 6):
            stats["malformed"] += 1
            continue
        image, prev_interval, multi_hot, hold_durations, phase, frame_idx = entry
        phase_norm = normalize_phase(phase)

        # Is the target key active?
        target_active = False
        try:
            if scalarize(multi_hot[ki]) >= 0.5:
                target_active = True
        except:
            pass

        if "press" in phase_norm:
            if target_active:
                stats["press_seen"] += 1
                if pending_press is None:
                    pending_press = (image, multi_hot, frame_idx)
                else:
                    # If a press is already pending, ignore this new press
                    pass
        elif "release" in phase_norm:
            stats["release_seen"] += 1
            if pending_press is not None:
                press_image, press_multi_hot, press_frame = pending_press
                # Get hold duration from release
                hold_vec = np.zeros((len(keys),), dtype=np.float32)
                hold_val = scalarize(hold_durations[ki])
                hold_vec[ki] = hold_val
                entry_press = (
                    press_image,
                    1.0,
                    press_multi_hot,
                    hold_vec,
                    "press",
                    press_frame
                )
                out_entries.append(entry_press)
                stats["paired"] += 1
                pending_press = None
            else:
                # Release found, but no previous press, ignore
                pass
        else:
            # Unknown phase, ignore
            pass

    if pending_press is not None:
        press_image, press_multi_hot, press_frame = pending_press
        hold_vec = np.zeros((len(keys),), dtype=np.float32)
        hold_vec[ki] = np.nan
        entry_press = (
            press_image,
            1.0,
            press_multi_hot,
            hold_vec,
            "press",
            press_frame
        )
        out_entries.append(entry_press)
        stats["unmatched_final_press"] += 1
        pending_press = None

    if not out_entries:
        raise RuntimeError("No press–release pairs created.")

    # Save results
    data_array = np.array(out_entries, dtype=object)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.splitext(os.path.basename(INPUT_NPZ))[0].replace(" ", "_")
    out_name = f"firstpress_paired_release_{TARGET_KEY}_{base}_{ts}.npz"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    np.savez(out_path, data=data_array, keys=np.array(keys, dtype=object))

    # Diagnostic summary
    print(f"[+] Saved: {out_path}")
    print("=== Diagnostic ===")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print(f"Paired press–release count: {stats['paired']}")
    if stats["unmatched_final_press"]:
        print(f"Final press with no release (hold=nan): {stats['unmatched_final_press']}")

if __name__ == "__main__":
    main()
