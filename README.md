# 🧠 JumpNet
## 🧠 JumpNet Dataset Builder – Behavior Cloning Data Pipeline

This repository provides a complete **data preparation pipeline** for training a machine learning model to play a jump-based game (e.g., *Geometry Dash*) using **behavior cloning**.

> 📌 Before a model can learn to play, it must first **observe** — that’s why a clean, high-quality dataset is essential.

---

## 📸 What This Pipeline Does

This pipeline converts raw gameplay and keylogging recordings into a structured `.npz` file containing labeled training data:

Each data entry contains:

* ✅ RGB image of the screen
* ✅ Label (1 = jump, 0 = no jump)
* ✅ Keypress vector (multi-hot)
* ✅ Hold duration of key
* ✅ Phase ("press")
* ✅ Frame index for video sync

---

## 🔗 Prerequisites – Start With Data Collection

This pipeline **assumes** you have already recorded raw data using our separate modular tool:

### 🛠 Modular Data Recorder Tool (REQUIRED):

* ✅ Snipping region selector
* ✅ Key logger
* ✅ Real-time frame synchronizer

> 📅 Use this tool **first** to generate your raw `.npz` files:

* 🔗 Dev.to Part 1: [Modular Snip Recorder](https://dev.to/ertugrulmutlu/modular-snip-recorder-a-data-collection-tool-for-behavior-cloning-12-5di8)
* 🔗 Dev.to Part 2: [Advanced Dataset Viewer](https://dev.to/ertugrulmutlu/modular-snip-recorder-a-data-collection-tool-for-behavior-cloning-22-1lgl)
* 🔗 GitHub: [Data Scrap Tool + Viewer](https://github.com/Ertugrulmutlu/-Data-Scrap-Tool-Advanced-Dataset-Viewer)

---

## 🧱 Pipeline Structure

This repo includes several modular scripts to process and structure your data:

| Step | File                        | Purpose                                          |
| ---- | --------------------------- | ------------------------------------------------ |
| 1️⃣  | `Data_preproccer.py`        | Extract clean `press-release` jump pairs         |
| 2️⃣  | `Data_merge.py`             | Merge multiple `.npz` files                      |
| 3️⃣  | `Data_argumentation.py`     | Augment positive samples with noise, flip, shift |
| 4️⃣  | `Data_Negativ_scrapping.py` | Extract "non-jump" frames from raw video         |
| 5️⃣  | `Data_reducier.py`          | Downsample negatives for class balance           |
| 6️⃣  | `Data_merge_final.py`       | Merge final positive + negative dataset          |

---

## 📦 Example Output

A few entries from the final dataset look like this:

```
📍 Entry #1
  image.shape     : (227, 227, 3)
  label           : 1.0
  keys_raw        : [1.]
  hold_duration   : [0.294]
  phase           : press
  frame_index     : 7720
```

---

## 🧰 Installation

**Install required packages:**

```bash
pip install -r Data_Process_requirements.txt
```

---

## ▶️ How to Run the Pipeline

After running each `.py` script, you can optionally inspect the output `.npz` file using the **Modular Viewer Tool** from Part 2:

> 🔍 This allows you to visually verify the result after each stage of the pipeline and detect any anomalies or malformed data early on.

**Step-by-step example (manual):**

```bash
# Step 1: Extract positive press–release pairs
python Data_preproccer.py
# ✅ Then use the Viewer tool to inspect the generated press–release dataset(Optional)


# Step 2: Merge multiple .npz files
python Data_merge.py
# ✅ Inspect the merged file using the Viewer(Optional)


# Step 3: Augment positive samples
python Data_argumentation.py
# ✅Open the augmented output in the Viewer to confirm transformations(Optional)


# Step 4: Extract negative samples from video
python Data_Negativ_scrapping.py
# ✅ Check extracted negative samples visually with the Viewer(Optional)


# Step 5: Reduce excessive negative frames
python Data_reducier.py
# ✅ View the reduced dataset for balance verification(Optional)


# Step 6: Merge final dataset
python Data_merge_final.py
# ✅ Final sanity check using the Viewer before training (Optional)
```

---

## 🧠 Data Structure Explained

Each dataset entry is a tuple with the following structure:

```python
(
  image,          # np.ndarray – RGB frame of size (227, 227, 3)
  label,          # float – 1.0 for jump, 0.0 for no jump
  keys_raw,       # list/array – multi-hot key state (e.g., [1.] if 'w' pressed)
  hold_duration,  # list/array – key hold duration in seconds (e.g., [0.294])
  phase,          # str – phase of the action (usually "press")
  frame_index     # int – frame index used for video synchronization
)
```

This structure ensures that the model can learn *when* and *how long* to jump, based on visual input and frame timing.

---

## 🔗 Blog Reference

This project is part of a multi-stage blog series:

*Data Scrapping Tool:
** 🧹 Part 1: [Modular Snip Recorder – Data Collection](https://dev.to/ertugrulmutlu/modular-snip-recorder-a-data-collection-tool-for-behavior-cloning-12-5di8)
** 🧹 Part 2: [Advanced Viewer & Inspector](https://dev.to/ertugrulmutlu/modular-snip-recorder-a-data-collection-tool-for-behavior-cloning-22-1lgl)

---
*Data procces
** 🚀 Part 1: [Part 1: From Raw Gameplay to Labeled Intelligence — Building the Data Foundation for JumpNet](https://dev.to/ertugrulmutlu/jumpnet-part-1-from-raw-gameplay-to-labeled-intelligence-building-the-data-foundation-for-2e2f)
** 🚀 Part 2: [Part 2: From Pixels to Policy — Training JumpNet to Make the Right Move](https://dev.to/ertugrulmutlu/jumpnet-part-1-from-raw-gameplay-to-labeled-intelligence-building-the-data-foundation-for-3c49)
** 🚀 Part 2: [Part 3: Real-Time Inference — Watching JumpNet Come Alive](https://dev.to/ertugrulmutlu/jumpnet-part-3-real-time-inference-watching-jumpnet-come-alive-21b2)


---

## 🧑‍💻 Troubleshooting

### ❌ `ValueError: object too deep for desired array`

* This error typically occurs when saving complex objects without using `dtype=object`.
* ✅ Fix: Ensure arrays are saved as `np.array(data, dtype=object)`.

### ❌ `KeyError: 'data'`

* This means the `.npz` file is missing a required `data` key.
* ✅ Fix: Check the generation step and confirm that `'data'` is part of the saved file.

### ❌ `cv2.error` in `Data_Negativ_scrapping.py`

* Likely due to an invalid or missing video file.
* ✅ Fix: Confirm that `cv2.VideoCapture(video_path).isOpened()` returns `True`.

### ❌ Model overfits quickly due to lack of shuffling

* ✅ Fix: Always apply `np.random.shuffle(all_data)` in `Data_merge_final.py` before saving.

---
