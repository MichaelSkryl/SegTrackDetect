# Extended SegTrackDetect — full user manual

This document complements the original `README.md` of
[SegTrackDetect](https://github.com/deepdrivepl/SegTrackDetect). The original
explains the architecture and how to register pre-trained models; this one
explains how to **install, run, and train** the extended version, including the
three proposed modifications and `run.py`, which processes ordinary photos and
videos without any COCO annotations.

### Contents

* [Part I — From clone to first result](#part-i--from-clone-to-first-result)
* [Part II — Running on your own photos and videos](#part-ii--running-on-your-own-photos-and-videos)
* [Part III — Running on an annotated dataset](#part-iii--running-on-an-annotated-dataset)
* [Part IV — Training your own models](#part-iv--training-your-own-models)
* [Part V — Command reference](#part-v--command-reference)
* [Part VI — Troubleshooting](#part-vi--troubleshooting)

---

## What the extension adds

| Module | Flag | Needs training? |
|---|---|---|
| **Temporal ConvGRU** — recurrent block inside the ROI encoder that accumulates evidence across frames, so the region of interest tracks the object more tightly. | `--use_temporal bottleneck` | **Yes** — a checkpoint trained on your domain. |
| **Detection significance heatmap** — decaying memory of recent detections, used as a third source of regions of interest. | `--use_heatmap` | No. |
| **Object-centric adaptive windowing** — extra detection windows centred on small tracked objects, sized so the crop is never downscaled. | `--use_adaptive_windowing` | No. |

All three are off by default. With none of them enabled the system reproduces
the original SegTrackDetect exactly.

**Recommended starting point** — this was the best configuration on the airport
dataset (+2.5 AP, +3.0 AR<sub>100</sub> over the baseline) and needs no extra
weights:

```
--allow_resize --use_adaptive_windowing
```

---

# Part I — From clone to first result

Follow these seven steps in order. Total time: about 15 minutes, most of it the
Docker build.

### Step 1 — Check prerequisites

* Docker 20.10+ with Compose v2 (the command is `docker compose`, with a space)
* For GPU: NVIDIA driver 525+ and
  [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
* CPU-only works, roughly 10–20× slower

Confirm Docker can see the GPU **before** building:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

A GPU table means you are ready. An error means the toolkit is missing — fix
that first, or plan to use the CPU service in step 4.

### Step 2 — Get the code

```bash
git clone <this-repository> SegTrackDetect
cd SegTrackDetect
```

### Step 3 — Create the working directories

These four are mounted into the container; everything you put in them is
visible inside, and everything the container writes appears on your machine.

```bash
mkdir -p weights input output data
```

| Directory | Holds |
|---|---|
| `weights/` | model files (`.pt`) |
| `input/` | your photos and videos |
| `output/` | annotated results |
| `data/` | COCO datasets (only for `inference.py` and training) |

### Step 4 — Build the image

```bash
docker compose build
```

This takes ~10 minutes. It installs PyTorch, OpenCV, ffmpeg and clones the SORT
tracker **over HTTPS**, so unlike the original `build_and_run.sh` it needs no
SSH key.

Check it worked:

```bash
docker compose run --rm segtrack python -c "import torch, cv2, kornia; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
```

If `cuda False` is printed but you expect a GPU, see
[Troubleshooting](#part-vi--troubleshooting). If you have no GPU at all, use
`segtrack-cpu` in place of `segtrack` in every command below, and add `--cpu`.

### Step 5 — Put the weights in place

Two models are needed: a **ROI estimator** (finds regions worth looking at) and
a **detector** (finds objects inside those regions).

For the airport configuration used throughout this manual:

```
weights/
├── yolov7t_airport_best.torchscript.pt                    ← detector
└── airport_baseline_batch_8/
    └── full_model_best.torchscript.pt                     ← ROI estimator
```

To download the public models instead (SeaDronesSee, DroneCrowd, MTSD, ZebraFish):

```bash
docker compose run --rm segtrack bash scripts/download_models.sh
```

These paths are not guessed by the program — they come from two registry files:

* `detector/configs/__init__.py` → the `DETECTION_MODELS` dictionary
* `rois/estimator/configs/__init__.py` → the `ESTIMATOR_MODELS` dictionary

The `--det_model` and `--roi_model` flags take **keys from those dictionaries**,
not file paths. Section [Register the result](#3-register-the-result) shows how
to add your own.

### Step 6 — Drop in a test photo

Copy any photo into `input/`. Then:

```bash
docker compose run --rm segtrack python run.py --source input/photo.jpg --out_dir output/first_test
```

### Step 7 — Look at the result

Open `output/first_test/frames/photo.jpg`. Boxes should be drawn around the
detected objects.

Nothing detected? That is usually correct behaviour, not a bug — the airport
models only know airport objects. Add `--draw_masks --draw_windows` to see what
the system was looking at; the [last troubleshooting
entry](#part-vi--troubleshooting) explains how to read that.

You now have a working installation. Part II covers real use.

---

# Part II — Running on your own photos and videos

`run.py` is the entry point for media without annotations. It accepts four kinds
of `--source`:

| `--source` | Treated as | Tracker & temporal modules |
|---|---|---|
| `input/photo.jpg` | one image | off (no temporal context) |
| `input/clip.mp4` | video | on |
| `input/frames/` | one ordered sequence, files sorted by name | on |
| `0` | webcam index | on |

### A single photo

```bash
docker compose run --rm segtrack python run.py --source input/photo.jpg --roi_model Airport_tiny_batch_8 --det_model AirportYolov7 --allow_resize --out_dir output/photo --save_json
```

Writes `output/photo/frames/photo.jpg` and `output/photo/detections.json`.

A lone image has no previous frame, so the tracker and all three temporal
modules are switched off automatically. `run.py` prints a note listing any flags
it ignored for that reason.

### A video

```bash
docker compose run --rm segtrack python run.py --source input/clip.mp4 --roi_model Airport_tiny_batch_8 --det_model AirportYolov7 --allow_resize --use_adaptive_windowing --out_dir output/clip --hud --save_json
```

Writes `output/clip/clip_annotated.mp4`.

Useful additions:

| Need | Flag |
|---|---|
| Try it on the first 100 frames only | `--max_frames 100` |
| Process every 3rd frame (3× faster) | `--stride 3` |
| Halve the output resolution (4K → 1080p) | `--vis_scale 0.5` |
| Show FPS and counts on the video | `--hud` |
| Raise the confidence needed to draw a box | `--vis_conf_th 0.5` |
| Boxes without text labels | `--no_labels` |

### A folder of frames

```bash
docker compose run --rm segtrack python run.py --source input/frames/ --roi_model Airport_tiny_batch_8 --det_model AirportYolov7 --allow_resize --use_adaptive_windowing --out_dir output/frames
```

A directory is treated as **one continuous sequence**. If your folder holds
unrelated stills, run them one at a time instead — otherwise the tracker carries
state from one scene into the next and invents regions of interest.

### Seeing how the system works (debug view)

```bash
docker compose run --rm segtrack python run.py --source input/clip.mp4 --roi_model Airport_tiny_batch_8 --det_model AirportYolov7 --allow_resize --use_adaptive_windowing --draw_masks --draw_windows --draw_oc_windows --out_dir output/debug
```

This reproduces the visualisation used for the paper figures:

| Colour | Meaning |
|---|---|
| blue overlay | segmentation mask (ROI estimation branch) |
| orange overlay | tracker mask (ROI prediction branch) |
| black rectangle | detection window sent to the detector |
| yellow rectangle | object-centric adaptive window |
| magenta box | the tracked object that produced a yellow window |

It is also the first thing to turn on when results look wrong — it tells you
immediately whether the problem is the ROI stage or the detector.

### Live window and webcam

```bash
xhost +local:docker          # Linux host, once per session
# then uncomment the DISPLAY / X11 lines in docker-compose.yml
docker compose run --rm segtrack python run.py --source 0 --show --hud --roi_model Airport_tiny_batch_8 --det_model AirportYolov7
```

Press `q` or `Esc` to stop. On Windows and macOS hosts this needs an X server;
the reliable path there is to drop `--show` and open the saved file.

## Using the temporal ConvGRU

The ConvGRU is a recurrent block inside the ROI encoder. On a single frame the
segmentation mask is produced from that frame alone, so it moves and changes
shape from one frame to the next even when the object does not. The ConvGRU
carries a hidden state forward, so each mask is informed by what the previous
frames showed. The practical effect is a steadier, more compact region of
interest, which in turn places the detection window more consistently on the
object.

### It cannot be used without trained weights

Unlike the heatmap and adaptive windowing, this module is a set of learned
parameters, and it ships **untrained**. It is deliberately initialised close to
the identity function — the blend gate starts near zero and the output
projection is zeroed — so that training begins from the baseline behaviour
rather than from noise. The consequence at inference time is that an untrained
ConvGRU passes features through almost unchanged and reproduces the baseline
result exactly.

`--use_temporal bottleneck` on its own therefore does nothing useful. It must
always be paired with `--temporal_weights`. If the file is missing or the path
is wrong, `run.py` stops with an error rather than falling back to baseline
behaviour, which would otherwise look like a successful run.

There are two ways to obtain weights:

**Option 1 — use the released checkpoint.** Trained on SeaDronesSee, suitable
for aerial and water scenes with very small objects.

<!-- TODO: replace with the release URL once the weights are uploaded -->
```bash
wget -nv <RELEASE_URL>/sds_convgru_phase2.pt -O weights/sds_convgru/full_model_best.pt
```

| Property | Value |
|---|---|
| Base ROI model | `SDS_large` |
| Insertion point | `2` |
| Hidden channels | `32` |
| Kernel size | `3` |
| Training | Phase 1 (30 epochs) + Phase 2 joint fine-tuning (15 epochs) |

**Option 2 — train your own** on your data, following
[Path B in Part IV](#path-b--training-the-temporal-convgru). This is the right
choice if your footage differs substantially from aerial water scenes.

### Three flags must match the checkpoint

`--insertion_point`, `--temporal_hidden` and `--temporal_ks` are not tuning
preferences at inference time — they define the shapes of the tensors being
loaded. The insertion point selects which encoder layer the block sits after
(`2` = after `layer1`, 64 channels; `5` = after `layer4`, 512 channels), and the
hidden size sets the recurrent width. If any of them differs from the values
used during training, loading fails with `RuntimeError: size mismatch`.

Every checkpoint is accompanied by a `train_args.json` recording exactly what
was used:

```bash
cat weights/sds_convgru/train_args.json
```

Read the three values from it, keeping in mind that the training and inference
flags have different names:

| In `train_args.json` | Pass at inference as |
|---|---|
| `insertion_point` | `--insertion_point` |
| `gru_hidden` | `--temporal_hidden` |
| `gru_kernel` | `--temporal_ks` |

> The defaults differ between training and inference (`--gru_hidden` defaults to
> 64, `--temporal_hidden` to 16). Always pass these values explicitly rather
> than relying on defaults.

### When it is worth enabling

The ConvGRU runs on any sequence — static camera, drone, hand-held. Whether it
improves anything depends on how stable the single-frame masks already are.

On a static camera the masks are typically stable from frame to frame, so a
module whose purpose is to stabilise them has little to correct; in our
experiments on static airport footage it produced no measurable gain over the
baseline. On moving-camera footage, where ego-motion makes the masks jitter, it
does help. It also costs a few frames per second.

Run your footage both ways and compare before committing to it.

### Command

```bash
docker compose run --rm segtrack python run.py --source input/drone.mp4 --roi_model SDS_large --det_model SDS --use_temporal bottleneck --temporal_weights weights/sds_convgru/full_model_best.pt --insertion_point 2 --temporal_hidden 32 --use_heatmap --out_dir output/drone --hud
```

`--roi_model` is still required: it supplies the input resolution and the
post-processing settings, even though the network weights come from
`--temporal_weights`. Use the same value the checkpoint was trained with.

---

# Part III — Running on an annotated dataset

This is the original workflow, used to reproduce the published numbers. It needs
a COCO-annotated dataset and gives you accuracy metrics; `run.py` needs neither
and gives you pictures.

### Dataset layout

```
data/YourDataset/
├── images/
│   ├── seq1/          ← video sequences, one directory each
│   │   ├── 000001.jpg
│   │   └── ...
│   └── seq2/
├── train.json         ← COCO format
├── val.json
└── test.json
```

For still images with no temporal order, put the files directly in `images/`
with no sub-directories.

> **Critical.** The `file_name` entries inside the JSON must be **absolute paths
> as seen inside the container**, e.g. `/SegTrackDetect/data/YourDataset/images/seq1/000001.jpg`.
> The loader filters the list with `os.path.isfile()`, so wrong paths do not
> raise an error — they silently produce an empty dataset, and the next line
> fails with a confusing `IndexError`. This is the single most common setup
> mistake.

### Inference

```bash
docker compose run --rm segtrack python inference.py --data_root data/YourDataset --split test --roi_model Airport_tiny_batch_8 --det_model AirportYolov7 --tracker sort --bbox_type sorted --allow_resize --use_adaptive_windowing --out_dir detections/my_run
```

Writes one text file of detections per sequence into `detections/my_run/`.

### Metrics

```bash
docker compose run --rm segtrack python metrics.py --dir detections/my_run --gt_path data/YourDataset/test.json --csv detections/my_run/metrics.csv
```

Prints the standard COCO table (AP, AP50, AP75, AP<sub>S/M/L</sub>, AR) and
appends a row to the CSV, so running several configurations into the same CSV
gives you a comparison table.

For DroneCrowd-style data (very dense, very small objects) add `--dc`, which
switches to `maxDets=500` and `iouThr=0.5`.

---

# Part IV — Training your own models

## What can and cannot be trained here

| Component | Trained here? |
|---|---|
| **ROI estimator** (UNet-ResNet18) | **Yes** — `train_bottleneck_temporal.py` |
| **Temporal ConvGRU** | **Yes** — same script, `--mode bottleneck` |
| **Detector** (YOLOv7-tiny) | **No.** Train it with the official [YOLOv7](https://github.com/WongKinYiu/yolov7) repository, export to TorchScript, and register the result. SegTrackDetect only consumes a finished detector. |

So "training your own model" here means training the part that decides **where
to look**. The part that decides **what is there** comes from outside.

## Which of the two paths do you need?

```
Do you want temporal memory in the ROI stage?
├── No  → Path A: fine-tune the baseline UNet on your data.
│         Simpler, faster, and what most users want.
└── Yes → Path A first (you need a good single-frame model),
          then Path B: two-phase ConvGRU training on top of it.
```

## Preparing the data

Training uses the same layout as Part III:

```
data/YourDataset/
├── images/seq1/... seq2/...
├── train.json
└── val.json
```

Only the training split is read during training. Keep a held-out split as well —
you will need it to evaluate the result with `inference.py` + `metrics.py`, which
is how you find out whether the training actually helped.

The ground-truth masks are generated automatically from the COCO bounding boxes,
so no segmentation annotation is required. Sequences matter: the script samples
consecutive runs of `--seq_len` frames from within a single sequence, which is
what makes temporal training possible.

---

## Path A — Fine-tuning the baseline ROI estimator

Four steps: train, export, register, use.

### 1. Train

```bash
docker compose run --rm segtrack python train_bottleneck_temporal.py --data_root data/YourDataset --split train --roi_model SDS_large --mode baseline --epochs 30 --lr 1e-3 --baseline_batch_size 8 --out_dir weights/my_baseline
```

`--roi_model` here selects the **starting weights** — training begins from that
pre-trained model rather than from scratch, which is why 30 epochs suffice.
Choose the public model closest to your domain (`SDS_large` for aerial/water
scenes, `DC_medium` for crowds, `MTSD` for road signs).

Produces in `weights/my_baseline/`:

| File | What it is |
|---|---|
| `full_model_best.pt` | lowest training loss — **use this one** |
| `full_model_final.pt` | last epoch |
| `train_args.json` | every argument used, for reproducibility |

Tips:

* `--baseline_batch_size 8` matters. BatchNorm behaves poorly at batch size 1;
  8 was what worked in our experiments.
* Checkpoint selection is by **training** loss. There is no validation loop, so
  a falling loss curve is not by itself evidence of a better model — evaluate
  the exported result on a held-out split with `metrics.py` before trusting it.
* `--bn_eval` freezes BatchNorm statistics. Worth trying if training is unstable
  or your dataset is small.
* `--seed 42` is the default; change it to check that a result is not luck.

### 2. Export to TorchScript

The training script saves a PyTorch `state_dict`, but the standard (non-temporal)
estimator loads **TorchScript**. Convert it:

```bash
docker compose run --rm segtrack python export_baseline.py
```

> `export_baseline.py` has hard-coded paths — open it and set the input path on
> line 8 and the output path on line 21 to your own before running. It strips
> the ConvGRU parameters and traces `forward_without_gru`, producing a clean
> single-frame model.

### 3. Register the result

Add an entry to `rois/estimator/configs/__init__.py`:

```python
MyModel = dict(
    weights      = 'weights/my_baseline/full_model_best.torchscript.pt',
    in_size      = (448, 768),      # must match your training resolution
    transform    = estimator_preprocess(448, 768),
    postprocess  = unet_postprocess,
    postprocess_args = dict(
        sigmoid_included = True,
        thresh           = 0.5,     # binarisation threshold
        dilate           = True,
        k_size           = 7,       # dilation kernel
    ),
)

ESTIMATOR_MODELS = {
    ...,
    "MyModel": MyModel,
}
```

Use a **relative** path. Two existing entries use absolute
`/SegTrackDetect/weights/...` paths, which break outside the container.

### 4. Use it

```bash
docker compose run --rm segtrack python run.py --source input/clip.mp4 --roi_model MyModel --det_model AirportYolov7 --allow_resize --use_adaptive_windowing --out_dir output/my_model
```

---

## Path B — Training the temporal ConvGRU

Two phases. The reason for splitting them: if you unfroze everything at once,
the randomly-initialised ConvGRU would send noisy gradients into a well-trained
UNet and damage it. Phase 1 lets the ConvGRU become useful while the UNet is
protected; Phase 2 then adapts them to each other at a low learning rate.

### Phase 1 — ConvGRU only, UNet frozen

```bash
docker compose run --rm segtrack python train_bottleneck_temporal.py --data_root data/YourDataset --split train --roi_model SDS_large --mode bottleneck --phase 1 --insertion_point 2 --gru_hidden 32 --gru_kernel 3 --alpha_init -3.0 --epochs 30 --lr 1e-3 --seq_len 16 --bptt_steps 4 --out_dir weights/my_gru_phase1
```

Produces `weights/my_gru_phase1/bottleneck_gru_best.pt` — the ConvGRU parameters
only, a small file.

Choosing the parameters:

| Parameter | Guidance |
|---|---|
| `--insertion_point` | `2` = after `layer1`, 64 channels, high spatial resolution — better for very small objects. `5` = after `layer4`, 512 channels, coarse — better for large objects. `2` was the better choice in our experiments on small targets. |
| `--gru_hidden` | `32` at insertion point 2; `64` at point 5. Larger means more capacity and more memory. |
| `--seq_len` | Consecutive frames per training subsequence. `16` is a good default; longer gives more temporal context and costs more memory. |
| `--bptt_steps` | How many frames the gradient flows back through. `4` is safe; `8` gives longer memory but risks vanishing gradients and needs more VRAM. Must be ≤ `seq_len`. |
| `--alpha_init` | Pre-sigmoid value of the blend gate. `-3.0` → the ConvGRU starts contributing ~5%, so the model begins near the baseline and cannot collapse at step 1. |

The loss is a weighted sum, controlled by three flags:

```
loss = 0.7 · BCE(prediction, ground truth)     ← --gt_weight
     + 0.2 · BCE(prediction, frozen teacher)   ← --distill_weight
     + 0.1 · temporal consistency              ← --tc_weight
```

The distillation term is what keeps the model anchored to the pre-trained
behaviour while the ground-truth term lets it improve on it; the consistency
term penalises frame-to-frame jitter. The defaults are what we used. Raise
`--tc_weight` if masks still flicker; raise `--gt_weight` if the model is too
conservative.

Watch the printed **alpha gate** each epoch. It is the fraction of the signal
coming from the ConvGRU. If it stays near its initial value, the ConvGRU is
learning nothing useful and something upstream is wrong — usually the data.

### Phase 2 — joint fine-tuning

```bash
docker compose run --rm segtrack python train_bottleneck_temporal.py --data_root data/YourDataset --split train --roi_model SDS_large --mode bottleneck --phase 2 --gru_weights weights/my_gru_phase1/bottleneck_gru_best.pt --insertion_point 2 --gru_hidden 32 --epochs 15 --lr 1e-4 --seq_len 16 --bptt_steps 4 --unet_lr_scale 0.01 --out_dir weights/my_gru_phase2
```

Produces `weights/my_gru_phase2/full_model_best.pt` — the complete model.

The architecture flags **must be identical to Phase 1**. Note the three
deliberate changes from Phase 1:

* `--phase 2` unfreezes the UNet
* `--lr 1e-4` — ten times lower, and half the epochs
* `--unet_lr_scale 0.01` — the UNet learns at 1% of that rate, i.e. 1e-6. It is
  being nudged, not retrained.

### Using the result

No export step — the temporal estimator loads the `state_dict` directly:

```bash
docker compose run --rm segtrack python run.py --source input/clip.mp4 --roi_model SDS_large --det_model SDS --use_temporal bottleneck --temporal_weights weights/my_gru_phase2/full_model_best.pt --insertion_point 2 --temporal_hidden 32 --out_dir output/temporal
```

`--roi_model` still matters: it supplies the input resolution and the
post-processing settings even though the weights come from `--temporal_weights`.
Keep it the same as during training.

Phase-1 weights can also be used on their own (`--temporal_weights .../bottleneck_gru_best.pt`);
the loader detects the format automatically. Note that this path currently hits
the `state` / `peek_state` bug described in the audit — apply that one-line fix
first.

### Ablations (optional)

To show that a gain comes from *temporal memory* rather than from merely
perturbing the features at that layer, `--perturbation_type` replaces the
ConvGRU with a control:

| Value | Control |
|---|---|
| `gru` | the real ConvGRU (default) |
| `identity` | pass-through — isolates the effect of fine-tuning alone |
| `gaussian` | additive noise (`--perturb_std`) |
| `dropout` | channel dropout (`--perturb_p`) |
| `frozen_gru` | same architecture, frozen random weights (`--perturb_alpha_init`) |

Everything else stays the same. If the ConvGRU beats all four, the gain is
attributable to what it learned.

---

# Part V — Command reference

## `run.py` — photos, videos, webcam

```bash
python run.py --source <path> [options]
```

**Input / output**

| Flag | Default | Meaning |
|---|---|---|
| `--source` | *required* | Image, video, directory, or camera index |
| `--out_dir` | `output/run` | Where results are written |
| `--stride` | `1` | Process every Nth frame |
| `--max_frames` | *none* | Stop after N processed frames |
| `--vis_scale` | `1.0` | Scale of the rendered output |
| `--show` | off | Live window (needs a display) |
| `--no_save` | off | Do not write media |
| `--save_json` | off | Also write `detections.json` |

**Models**

| Flag | Default | Meaning |
|---|---|---|
| `--roi_model` | `Airport_tiny_batch_8` | Key in `ESTIMATOR_MODELS` |
| `--det_model` | `AirportYolov7` | Key in `DETECTION_MODELS` |
| `--tracker` | `sort` | Key in `PREDICTOR_MODELS` |
| `--cpu` | off | Force CPU |

**Detection windows**

| Flag | Default | Meaning |
|---|---|---|
| `--bbox_type` | `sorted` | `all` / `naive` / `sorted` — window filtering strategy |
| `--allow_resize` | off | Downscale oversized windows instead of tiling them. Recommended on for scenes with mixed object sizes |
| `--obs_iou_th` | `0.7` | Overlapping Box Suppression threshold |

**Modification 1 — ConvGRU**

| Flag | Default | Meaning |
|---|---|---|
| `--use_temporal` | `none` | `bottleneck` enables the ConvGRU |
| `--temporal_weights` | *none* | Checkpoint path — **required** when temporal is on |
| `--insertion_point` | `5` | Encoder layer 2–5 — must match the checkpoint |
| `--temporal_hidden` | `16` | Hidden channels — must match the checkpoint's `gru_hidden` |
| `--temporal_ks` | `3` | Kernel size — must match the checkpoint |

**Modification 2 — heatmap**

| Flag | Default | Meaning |
|---|---|---|
| `--use_heatmap` | off | Enable the detection significance heatmap |
| `--heatmap_decay` | `0.85` | Per-frame decay. Higher = longer memory |
| `--heatmap_tiny_threshold` | `0.01` | Relative size below which detections are boosted |
| `--heatmap_activation_threshold` | `0.3` | Value above which a pixel becomes a region of interest |

**Modification 3 — adaptive windowing**

| Flag | Default | Meaning |
|---|---|---|
| `--use_adaptive_windowing` | off | Enable object-centric windows |
| `--aw_tiny_threshold` | `0.01` | Relative size below which an object gets its own window |
| `--aw_min_window_ratio` | `0.5` | Minimum window size as a fraction of the detector input |
| `--aw_padding_factor` | `2.0` | Padding multiplier around the tracked object |
| `--aw_max_extra_windows` | `5` | Cap per frame — this is the FPS/recall trade-off dial |

**Visualisation**

| Flag | Default | Meaning |
|---|---|---|
| `--vis_conf_th` | `0.3` | Only draw detections above this confidence |
| `--draw_masks` | off | Overlay ROI masks (blue = segmentation, orange = tracker) |
| `--draw_windows` | off | Draw detection windows in black |
| `--draw_oc_windows` | off | Highlight object-centric windows in yellow |
| `--no_labels` | off | Boxes without text |
| `--hud` | off | FPS and per-frame counts |

## `inference.py` — annotated datasets

```bash
python inference.py --data_root <dir> --split <name> [options]
```

Shares the model, window, and all three modification groups with `run.py`
(identical names and defaults). Differences:

| Flag | Default | Meaning |
|---|---|---|
| `--data_root` | *required* | Dataset root |
| `--split` | `test` | Split name → `<data_root>/<split>.json` |
| `--flist` | *none* | Text file listing specific images instead of a split |
| `--name` | *none* | Name for the custom split created from `--flist` |
| `--out_dir` | `detections` | Output directory |
| `--debug` | off | Save visualisations alongside the detections |
| `--use_temporal` | `none` | Also accepts the experimental `dual`, `postdecoder`, `multiscale` modes |

## `metrics.py` — COCO evaluation

```bash
python metrics.py --dir <detections_dir> --gt_path <ground_truth.json> [options]
```

| Flag | Default | Meaning |
|---|---|---|
| `--dir` | *required* | Directory of detection files from `inference.py` |
| `--gt_path` | `data/SeaDronesSee/test_dev.json` | Ground-truth JSON |
| `--th` | `0.01` | Score threshold applied before evaluation |
| `--csv` | `checks/metrics.csv` | Results CSV; appended to if it exists |
| `--dc` | off | DroneCrowd mode: `maxDets=500`, `iouThr=0.5` |

## `train_bottleneck_temporal.py` — training

```bash
python train_bottleneck_temporal.py --data_root <dir> --mode <mode> [options]
```

**Data**

| Flag | Default | Meaning |
|---|---|---|
| `--data_root` | *required* | Dataset root |
| `--split` | `train` | Training split |
| `--roi_model` | `SDS_tiny` | Starting weights, a key in `ESTIMATOR_MODELS` |

**Mode**

| Flag | Default | Meaning |
|---|---|---|
| `--mode` | `bottleneck` | `baseline` = UNet only (Path A); `bottleneck` = ConvGRU in the encoder (Path B); `post_unet` = refiner after the UNet |
| `--phase` | `1` | `1` = ConvGRU only, UNet frozen; `2` = joint fine-tuning |

**Optimisation**

| Flag | Default | Meaning |
|---|---|---|
| `--epochs` | `30` | Use ~15 for Phase 2 |
| `--lr` | `1e-3` | Use `1e-4` for Phase 2 |
| `--unet_lr_scale` | `0.01` | Phase 2 only: UNet learning-rate multiplier |
| `--seq_len` | `16` | Consecutive frames per training subsequence |
| `--bptt_steps` | `4` | Truncated-BPTT depth. Must be ≤ `seq_len` |
| `--baseline_batch_size` | `8` | `--mode baseline` only |
| `--bn_eval` | off | Freeze BatchNorm statistics |
| `--seed` | `42` | Random seed |
| `--cpu` | off | Force CPU (very slow for training) |

**ConvGRU architecture** — record these; you need them again at inference

| Flag | Default | Inference equivalent |
|---|---|---|
| `--insertion_point` | `5` | `--insertion_point` |
| `--gru_hidden` | `64` | `--temporal_hidden` (default `16` — always pass explicitly) |
| `--gru_kernel` | `3` | `--temporal_ks` |
| `--alpha_init` | `-3.0` | — (stored in the checkpoint) |
| `--gru_weights` | *none* | Phase-1 checkpoint to start Phase 2 from |

**Loss weights**

| Flag | Default | Meaning |
|---|---|---|
| `--gt_weight` | `0.7` | Ground-truth supervision |
| `--distill_weight` | `0.2` | Self-distillation from the frozen teacher |
| `--tc_weight` | `0.1` | Temporal consistency between frames |

**Checkpointing**

| Flag | Default | Meaning |
|---|---|---|
| `--out_dir` | `weights/temporal_v4` | Where checkpoints and `train_args.json` go |

The best checkpoint is the epoch with the lowest **training** loss; there is no
validation loop. Evaluate the result on a held-out split with `inference.py` +
`metrics.py` before drawing conclusions from it.

**Ablation controls**

| Flag | Default | Meaning |
|---|---|---|
| `--perturbation_type` | `gru` | `gru` / `identity` / `gaussian` / `dropout` / `frozen_gru` |
| `--perturb_std` | `0.05` | `gaussian` only |
| `--perturb_p` | `0.05` | `dropout` only |
| `--perturb_alpha_init` | `-5.0` | `frozen_gru` only |

**Which file you get**

| Mode | Phase | Best checkpoint |
|---|---|---|
| `baseline` | — | `full_model_best.pt` (then run `export_baseline.py`) |
| `bottleneck` | 1 | `bottleneck_gru_best.pt` |
| `bottleneck` | 2 | `full_model_best.pt` |
| `post_unet` | — | `temporal_refiner_best.pt` |

Each also has a `*_final.pt` twin from the last epoch. Prefer `*_best.pt`.

---

# Part VI — Troubleshooting

**`exec format error` when running a script under `scripts/`.** Invoke it
through bash rather than directly — `bash scripts/download_models.sh`, not
`./scripts/download_models.sh`. The upstream scripts carry no `#!` line, so
Docker has no way to know what should execute them.

**`$'\r': command not found`, or a directory literally named `weights?`.** The
scripts were checked out on Windows with CRLF line endings and copied into the
image that way. Rebuilding fixes it — the Dockerfile now strips the carriage
returns after `COPY`:

```bash
docker compose build --no-cache
```

To stop it recurring in the working tree itself, a `.gitattributes` pins `*.sh`
to LF. It applies to files as they are checked out, so existing files need one
renormalisation pass:

```bash
git add --renormalize .
```

**`cuda False` / everything runs on CPU.** The container did not get the GPU.
Run the `nvidia-smi` check from Step 1; if that fails, install
nvidia-container-toolkit and restart the Docker daemon.

**`<name> not in ESTIMATOR_MODELS.keys()`.** `--roi_model` takes a dictionary
key, not a file path. Valid keys are at the bottom of
`rois/estimator/configs/__init__.py`; likewise `--det_model` and
`detector/configs/__init__.py`.

**`FileNotFoundError` on a `.pt` file.** The registry entry points somewhere the
file is not. Check that `./weights` on the host really contains it, and that the
entry uses a path relative to `/SegTrackDetect`. `docker compose config` shows
the resolved mounts.

**`RuntimeError: size mismatch` when loading temporal weights.** Almost always
`--temporal_hidden`: training defaults to 64, inference to 16. Read the correct
values from `train_args.json` next to the checkpoint and pass them explicitly.

**Temporal weights load but results equal the baseline.** Either the checkpoint
path was wrong (older versions skipped it silently — `run.py` now refuses to
start), or the alpha gate never rose during training. Check the alpha values
printed at startup and during training.

**`IndexError` immediately after "Found 0 images".** The `file_name` entries in
your COCO JSON do not resolve. They must be absolute paths as seen **inside the
container** — `/SegTrackDetect/data/...`, not a host path and not a relative one.

**`Could not open video writer`.** ffmpeg missing — rebuild with
`docker compose build --no-cache`.

**`ModuleNotFoundError: rois.predictor.SORT`.** The clone failed during the
build, usually a network issue. Rebuild, or clone it manually into
`rois/predictor/SORT`.

**`cannot connect to X server` with `--show`.** Expected without a display. Drop
`--show` and open the saved file.

**Out of memory.** During inference: add `--stride 2`, lower
`--aw_max_extra_windows`, or use `--cpu`. Memory scales with the number of
detection windows per frame, not directly with frame size. During training:
lower `--seq_len`, then `--bptt_steps`, then `--baseline_batch_size`.

**Runs fine, detects nothing.** Usually correct behaviour — check the ROI model
and the detector belong to the same domain (an airport detector will not find
swimmers). To find out which stage is at fault, add `--draw_masks
--draw_windows`: an empty blue overlay means the ROI estimator found nothing, so
that is what needs retraining; a populated overlay with no boxes means the ROI
stage works and the detector is the problem.
