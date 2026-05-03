"""TensorBoard web endpoint backed by the shared `hexhex-runs` Modal volume.

Usage:
    uv run modal deploy hexhex/modal/tb.py

Modal prints a stable `https://...modal.run` URL on deploy. The container scales
to zero when idle (you only pay volume storage), and cold-starts in seconds.

Why we mirror the volume into a local dir:
    TensorBoard's data server keeps events files open with persistent file handles.
    Modal's `volume.reload()` refuses to refresh the mount while files on it are
    open ("there are open files preventing the operation"), so pointing TB
    directly at the volume mount means new runs never become visible. We rsync
    the volume (read-only mount) into a local path that TB watches, and refresh
    on a timer. Trade-off: a few seconds of staleness and some disk churn for
    actually-working live updates.

    rsync runs with --inplace so growing events files keep the same inode —
    without it, rsync writes a temp file and renames it, giving the destination
    a new inode each sync. TB's data server caches an open handle to the old
    inode and never sees the appended bytes, which manifested as "first reload
    shows one datapoint, then it never updates again."
"""
import modal

PROJECT_ROOT = "/workspace"
VOLUME_MOUNT = f"{PROJECT_ROOT}/runs_volume"  # read-only mount of the volume
LOG_DIR = f"{PROJECT_ROOT}/runs"              # what TB watches; rsynced from VOLUME_MOUNT
VOLUME_NAME = "hexhex-runs"

runs_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

tb_image = modal.Image.debian_slim(python_version="3.11").apt_install("rsync").pip_install("tensorboard")

app = modal.App("hexhex-tb")


@app.function(
    image=tb_image,
    volumes={VOLUME_MOUNT: runs_volume},
    # Keep idle containers alive long enough to be useful, then let them shut
    # down so storage is the only thing you keep paying for.
    scaledown_window=60 * 30,
    timeout=60 * 60 * 12,
    min_containers=0,
    max_containers=1,
)
@modal.web_server(6006, startup_timeout=60)
def tensorboard():
    import os
    import subprocess
    import threading
    import time

    os.makedirs(LOG_DIR, exist_ok=True)

    # Initial sync before TB starts so it sees existing runs immediately.
    subprocess.run(["rsync", "-a", "--inplace", "--delete", f"{VOLUME_MOUNT}/", f"{LOG_DIR}/"], check=False)

    subprocess.Popen(
        [
            "tensorboard",
            "--logdir", LOG_DIR,
            "--host", "0.0.0.0",
            "--port", "6006",
            "--reload_multifile=true",
            "--reload_interval=5",
        ]
    )

    # Refresh the volume view, then mirror it into LOG_DIR so TB picks up new
    # runs without holding file handles on the volume itself.
    def syncer():
        while True:
            time.sleep(10)
            try:
                runs_volume.reload()
            except Exception as e:
                print("reload failed:", e)
                continue
            subprocess.run(
                ["rsync", "-a", "--inplace", "--delete", f"{VOLUME_MOUNT}/", f"{LOG_DIR}/"],
                check=False,
            )

    threading.Thread(target=syncer, daemon=True).start()
