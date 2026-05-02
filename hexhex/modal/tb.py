"""TensorBoard web endpoint backed by the shared `hexhex-runs` Modal volume.

Usage:
    uv run modal deploy hexhex/modal/tb.py

Modal prints a stable `https://...modal.run` URL on deploy. The container scales
to zero when idle (you only pay volume storage), and cold-starts in seconds.
"""
import modal

PROJECT_ROOT = "/workspace"
RUNS_DIR = f"{PROJECT_ROOT}/runs"
VOLUME_NAME = "hexhex-runs"

runs_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

tb_image = modal.Image.debian_slim(python_version="3.11").pip_install("tensorboard")

app = modal.App("hexhex-tb")


@app.function(
    image=tb_image,
    volumes={RUNS_DIR: runs_volume},
    # Keep idle containers alive long enough to be useful, then let them shut
    # down so storage is the only thing you keep paying for.
    scaledown_window=60 * 30,
    timeout=60 * 60 * 12,
    min_containers=0,
)
@modal.web_server(6006, startup_timeout=60)
def tensorboard():
    import os
    import subprocess
    import threading
    import time

    os.makedirs(RUNS_DIR, exist_ok=True)
    subprocess.Popen(
        [
            "tensorboard",
            "--logdir", RUNS_DIR,
            "--host", "0.0.0.0",
            "--port", "6006",
            "--reload_multifile=true",
        ]
    )

    # Refresh the container's view of the volume so newly-committed events
    # files from the trainer (or `modal volume put` from your laptop) become
    # visible without restarting the TB container.
    def reloader():
        while True:
            time.sleep(30)
            try:
                runs_volume.reload()
            except Exception as e:
                print("reload failed:", e)

    threading.Thread(target=reloader, daemon=True).start()
