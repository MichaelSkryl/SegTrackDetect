# =============================================================================
# Extended SegTrackDetect — container image
#
# Differences from the original SegTrackDetect Dockerfile:
#   * SORT is cloned over HTTPS during the build, so no SSH key is needed on
#     the host (the original build_and_run.sh used git@github.com:...).
#   * ffmpeg + video codecs are installed so run.py can read and write video.
#   * The application code is COPIED into the image, so the container is
#     self-contained: only weights and media need to be mounted.
#   * Extra Python packages used by the modified pipeline are pinned.
#
# Build:
#   docker build -t segtrackdetect:extended .
#
# See README_EXTENDED.md for run instructions.
# =============================================================================
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# --- system packages ---------------------------------------------------------
# libgl1-mesa-glx + libglib2.0-0 : OpenCV runtime
# ffmpeg + libsm6 + libxext6      : video decode/encode for run.py
# libxrender1                     : needed when using cv2.imshow via X11
# unzip: every scripts/download_*.sh extracts a .zip. The upstream image
# installs only `zip`, which does not provide it, so the dataset downloads
# failed at the extraction step.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git gcc g++ wget zip unzip htop screen \
        libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# --- python packages ---------------------------------------------------------
RUN pip install --no-cache-dir \
        seaborn==0.13.2 \
        cython==3.0.11 \
        thop==0.1.1.post2209072238 \
        opencv-python==4.10.0.84 \
        gdown==5.2.0 \
        Pillow==10.2.0 \
        scikit-image==0.24.0 \
        filterpy==1.4.5 \
        lap==0.4.0 \
        kornia==0.7.3 \
        pandas==2.2.2 \
        tqdm==4.66.4

RUN pip install --no-cache-dir \
        git+"https://github.com/Cufix/tinycocoapi.git#egg=pycocotools&subdirectory=PythonAPI"

WORKDIR /SegTrackDetect

# --- SORT tracker ------------------------------------------------------------
# Cloned over HTTPS so the build needs no credentials. It lands exactly where
# rois/predictor/configs/sort.py expects it (module 'rois.predictor.SORT').
RUN git clone --depth 1 https://github.com/deepdrivepl/SORT.git \
        /SegTrackDetect/rois/predictor/SORT

# --- application code --------------------------------------------------------
# Copied last so that code edits do not invalidate the dependency layers.
# Anything listed in .dockerignore (weights, data, .git, outputs) is skipped.
COPY . /SegTrackDetect

# --- normalise shell scripts -------------------------------------------------
# A checkout on Windows with core.autocrlf=true rewrites every .sh file with
# CRLF line endings, which a Linux shell cannot run: `mkdir -p $OUT_DIR` would
# create a directory whose name ends in a carriage return, and every command
# reports `$'\r': command not found`. Strip the CRs and set the executable bit,
# which git does not preserve on Windows either.
RUN find /SegTrackDetect/scripts -type f -name '*.sh' -print0 \
        | xargs -0 -r sed -i 's/\r$//' \
    && find /SegTrackDetect/scripts -type f -name '*.sh' -print0 \
        | xargs -0 -r chmod +x

# Mount points for things that stay outside the image
RUN mkdir -p /SegTrackDetect/weights \
             /SegTrackDetect/data \
             /SegTrackDetect/input \
             /SegTrackDetect/output \
             /SegTrackDetect/detections

CMD ["/bin/bash"]
