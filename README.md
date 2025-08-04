# 🧠 JumpNet – Behavior Cloning Pipeline for Jump-Based Games

JumpNet is a full-scale machine learning pipeline that empowers an AI agent to play jump-based games such as *Geometry Dash* using **behavior cloning**. The system consists of three main stages: data collection, model training, and real-time inference. Every part is modular, extensible, and designed for transparency and reproducibility.

The complete pipeline is backed by a multi-part blog series and two GitHub repositories, making it easy for researchers, students, and hobbyists to dive into AI-powered gameplay.

---

## 🔗 Blog Series Reference

JumpNet is documented step-by-step through a 3-part technical blog series:

### 🧹 Data Collection Tools

* 📌 [Modular Snip Recorder – Data Collection](https://dev.to/ertugrulmutlu/modular-snip-recorder-a-data-collection-tool-for-behavior-cloning-12-5di8): Learn how to build a modular snipping tool that captures screen frames and keypresses in sync.
* 📌 [Advanced Viewer & Inspector](https://dev.to/ertugrulmutlu/modular-snip-recorder-a-data-collection-tool-for-behavior-cloning-22-1lgl): A Tkinter-based tool to visually inspect `.npz` datasets after each stage of processing.

### 🧠 Data Processing & Model Training

* 🚀 [Part 1 – From Raw Gameplay to Labeled Intelligence](https://dev.to/ertugrulmutlu/jumpnet-part-1-from-raw-gameplay-to-labeled-intelligence-building-the-data-foundation-for-2e2f): Convert raw video and key logs into usable labeled datasets.
* 🚀 [Part 2 – From Pixels to Policy](https://dev.to/ertugrulmutlu/jumpnet-part-1-from-raw-gameplay-to-labeled-intelligence-building-the-data-foundation-for-3c49): Train a MobileNetV2-based dual-head model to predict jump action and duration.

### 🎮 Real-Time Inference & Gameplay

* 📺 [Part 3 – Watching JumpNet Come Alive](https://dev.to/ertugrulmutlu/jumpnet-part-3-real-time-inference-watching-jumpnet-come-alive-21b2): Build a Tkinter GUI that connects your model to real-time gameplay using keyboard simulation.
* 📹 [Gameplay Demo Video](https://www.youtube.com/watch?v=FjLwtyjw5OY): A demonstration of the model playing the game in real-time.

---

# 1️⃣ Dataset Builder – From Raw to Labeled

This stage transforms raw gameplay and keylogging recordings into a clean, structured dataset ready for model training. It focuses on identifying "press-release" jump segments and balancing the dataset with non-jump frames.

### 🧰 Installation

```bash
pip install -r Data_Process_requirements.txt
```

### 🧱 Pipeline Components

Each processing step is handled by a modular Python script:

| Step | File                        | Purpose                                               |
| ---- | --------------------------- | ----------------------------------------------------- |
| 1️⃣  | `Data_preproccer.py`        | Extract valid press–release jump events               |
| 2️⃣  | `Data_merge.py`             | Merge multiple labeled `.npz` files                   |
| 3️⃣  | `Data_argumentation.py`     | Augment positive samples (flip, shift, noise)         |
| 4️⃣  | `Data_Negativ_scrapping.py` | Extract negative (non-jump) frames from raw gameplay  |
| 5️⃣  | `Data_reducier.py`          | Downsample negative samples to maintain class balance |
| 6️⃣  | `Data_merge_final.py`       | Merge the final positive and negative datasets        |

> ✅ After every stage, you can inspect intermediate results using the **Viewer Tool**.

### 📦 Example Output (Single Entry)

```
image.shape     : (227, 227, 3)
label           : 1.0
keys_raw        : [1.]
hold_duration   : [0.294]
phase           : press
frame_index     : 7720
```

### 🔍 Dataset Format

```python
(
  image: np.ndarray,         # Shape: (227, 227, 3)
  label: float,              # 1.0 for jump, 0.0 for no jump
  keys_raw: list[float],
  hold_duration: list[float],
  phase: str,                # Usually "press"
  frame_index: int           # Used for video synchronization
)
```

### 🐞 Troubleshooting

* `ValueError: object too deep for desired array` → Use `dtype=object` when saving arrays.
* `KeyError: 'data'` → Ensure the `data` key is properly written when saving.
* `cv2.error` → Check if video path is valid and readable.

---

# 2️⃣ Model Trainer – Learning to Jump

This section trains a CNN model (JumpNet) to make real-time jump decisions using screen pixels and timing features.

### 🧰 Installation

```bash
pip install -r Train_requirements.txt
```

### 🏗️ Components

| File             | Description                                                  |
| ---------------- | ------------------------------------------------------------ |
| `train.py`       | Main training loop using MobileNetV2                         |
| `test.py`        | Model evaluation script                                      |
| `dataset.py`     | Custom PyTorch Dataset class + train/test split              |
| `model.py`       | CNN architecture with two heads (classification, regression) |
| `utils.py`       | Loss functions, optimizer, save/load, metrics                |
| `data_contol.py` | Helper script for dataset visualization/stats                |

### 🧠 Model Architecture: JumpNet

* Base: MobileNetV2
* Head 1: `jump_head` → binary output (jump/no-jump)
* Head 2: `hold_head` → regression (hold duration in seconds)

```python
jump_prob, hold_duration = model(image_tensor)
```

### ▶️ Training Command

```bash
python train.py
```

* Input: `./datas/Geodashreelfinaldata.npz`
* Epochs: 20
* Batch size: 32
* Optimizer: Adam (LR=1e-4)
* Logging: TensorBoard (logs in `runs/`)

### 📊 Output Example

```
[Epoch 4/20 | Batch 11/56] Loss: 0.1810 (Cls: 0.1287, Reg: 0.0523)
✅ Epoch 4 completed | Total: 0.1924 | Cls: 0.1342 | Reg: 0.0582
```

### 🧪 Evaluation Metrics

```bash
python test.py
```

* Accuracy
* F1 Score
* Precision / Recall
* Hold Duration MSE

> 📌 For regression, only positive labels (jump = 1) are considered.

---

# 3️⃣ Real-Time Inference – GUI Simulation

Once trained, the model can be deployed using a live GUI to control the game via screen reading and key simulation.

### 🧰 Installation

```bash
pip install -r Simulation_requierments.txt
```

### ▶️ Launch

```bash
python main.py
```

### 🎛️ GUI Features

| Feature         | Description                              |
| --------------- | ---------------------------------------- |
| Load Model      | Load a `.pt` file with trained weights   |
| Snip Region     | Select game window region to capture     |
| Threshold       | Confidence threshold to trigger jump     |
| Interval        | Delay between each prediction in ms      |
| Debounce        | Min. time between two keypresses         |
| Hold Multiplier | Multiplies the predicted hold duration   |
| Key to Simulate | Define the jump key (e.g., "w", "space") |

### ⚙️ Under the Hood

1. GUI launches `CaptureThread`
2. Screen is captured via `mss`
3. Image is preprocessed and sent to model
4. If `jump_prob > threshold`, and enough time passed since last jump:

   * The model triggers keypress using `pynput`
5. Logs and status updates appear in the GUI

### 🪵 Sample Log

```
[INFO] Pressing key: 'w', planned duration: 0.315 s
[INFO] Released key: 'w', actual duration: 0.314 s
[INFO] Triggered: prob=0.786, hold=0.315s, threshold=0.50
FPS: 31.4
```

---

# 🎯 Final Notes & Hook-Up

JumpNet is a complete AI pipeline from data to deployment:

* 🧩 [Data Tool + Viewer (GitHub)](https://github.com/Ertugrulmutlu/-Data-Scrap-Tool-Advanced-Dataset-Viewer)
* 🤖 [Training + GUI Inference (GitHub)](https://github.com/Ertugrulmutlu/JumpNet)

Feel free to test with different screen sizes, thresholds, and retrained models to push the limits.

---

## ⭐ If You Liked It

Consider giving the project a star ⭐ to support open-source AI experiments:

👉 [Star Data Tool](https://github.com/Ertugrulmutlu/-Data-Scrap-Tool-Advanced-Dataset-Viewer)
👉 [Star JumpNet](https://github.com/Ertugrulmutlu/JumpNet)

Thanks for reading — and happy jumping! 🎮🚀
