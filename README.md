[README(3).md](https://github.com/user-attachments/files/28412385/README.3.md)
Models: https://drive.google.com/drive/folders/1XMOiqnan7a5JxifsDSUbqo4RdrTx7RDm?usp=sharing
# PPE Safety Detection System
A YOLO-based computer vision tool that detects whether workers on construction sites are wearing the required Personal Protective Equipment (PPE) — specifically a **safety vest** and a **hard hat**. It runs on images, videos, or benchmark datasets with different occlusion levels.

---

## File Structure

```
project/
│
├── Code.py                  # Main script
│
├── Output/                  # Auto-created on first run
│   ├── safety_output.jpg    # Annotated image output
│   └── safety_output.mp4    # Annotated video output
│
└── <your_model>.pt          # YOLO model weights (user-supplied)
```

When running in evaluation mode, datasets are read from the path defined by `DIR_DATASETS` inside the script (default: `/home/raul/Downloads/`). Expected dataset layout:

```
DIR_DATASETS/
├── construction_occlusion_none/data.yaml
├── Dataset_occlusion_low/data.yaml
├── Dataset_occlusion_medium/data.yaml
└── Dataset_occlusion_High/data.yaml
```

---

## Dependencies

Install Python dependencies before running:

```bash
pip install ultralytics opencv-python
```

| Package | Purpose |
|---|---|
| `os` | Standard library — file path and directory handling |
| `cv2` (OpenCV) | Image and video reading, annotation, and writing |
| `argparse` | Standard library — CLI argument parsing |
| `ultralytics` | YOLO model loading, inference, and validation |

---

## CLI Flags

| Flag | Required | Description |
|---|---|---|
| `--model <path>` | **Always** | Path to the YOLO `.pt` model weights file |
| `--source <path>` | In inference mode | Path to an input image or video file |
| `--eval` | Optional | Run dataset evaluation instead of inference |

### Supported input formats

- **Images:** `.jpg`, `.jpeg`, `.png`, `.bmp`
- **Videos:** `.mp4`, `.avi`, `.mov`, `.mkv`

---

## Usage Examples

**Run on an image:**
```bash
python Code.py --model best.pt --source site_photo.jpg
```

**Run on a video:**
```bash
python Code.py --model best.pt --source site_footage.mp4
```
Press `q` to stop video processing early.

**Run occlusion benchmark evaluation:**
```bash
python Code.py --model best.pt --eval
```
This iterates over all four occlusion levels (None, Low, Medium, High) and prints mAP50, mAP75, and mAP50-95 metrics for each.

---

## Detection Classes

| ID | Class | Role |
|---|---|---|
| 0 | `Vest` | Safety vest detected |
| 1 | `Person` | Full-body person bounding box |
| 2 | `head` | Head bounding box (used when full body isn't detected) |
| 3 | `No-Vest` | Explicit no-vest detection (logged but not currently scored) |
| 4 | `hat` | Hard hat detected |

---

## Scoring Logic

Each detected **Person** is evaluated for two items. Each missing item deducts 50 points:

| PPE Present | Score |
|---|---|
| Vest + Hard Hat | 100 — `SAFE` (green) |
| One item missing | 50 — `WARNING` (orange) |
| Both items missing | 0 — `DANGER` (red) |

If a **Person** box is not detected but a standalone **head** is visible, it is evaluated only for a hard hat (0 or 100). The **global safety score** shown on each frame is the average across all evaluated subjects.

---

## Configuration

The following constants at the top of `Code.py` can be adjusted:

| Constant | Default | Description |
|---|---|---|
| `CONF_THRESHOLD` | `0.30` | Minimum detection confidence to include a box |
| `DIR_DATASETS` | `/home/raul/Downloads/` | Root path for evaluation datasets |
