"""Run repeated_self_training on a Modal GPU.

Usage:
    uv run modal run hexhex/modal/train.py
    uv run modal run hexhex/modal/train.py --preset 5x5
    uv run modal run hexhex/modal/train.py --preset 5x5 --overrides "rst.num_iterations=200"
    uv run modal run hexhex/modal/train.py --preset 5x5 --exp-name a10g_run1

    # push a locally-trained run into the same volume so it shows up in TensorBoard
    uv run modal volume put hexhex-runs ./runs/<exp_id> /runs/

    # download / list / clean up
    uv run modal volume ls  hexhex-runs
    uv run modal volume get hexhex-runs runs/<exp_id> ./runs/
    uv run modal volume rm  hexhex-runs runs/<exp_id> -r
"""
import modal

PROJECT_ROOT = "/workspace"
RUNS_DIR = f"{PROJECT_ROOT}/runs"
VOLUME_NAME = "hexhex-runs"

runs_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "numpy",
        "tensorboard",
        "hydra-core>=1.3.0",
        "matplotlib",
        "scikit-learn",
        "onnx",
    )
    .add_local_dir(
        local_path=".",
        remote_path=PROJECT_ROOT,
        ignore=[
            "runs/**",
            "outputs/**",
            ".venv/**",
            "node_modules/**",
            "app/**",
            "tests/**",
            ".git/**",
            "tables/**",
            "**/__pycache__/**",
        ],
    )
)

app = modal.App("hexhex-train")


@app.function(
    image=train_image,
    gpu="L4",
    volumes={RUNS_DIR: runs_volume},
    timeout=60 * 60 * 6,
)
def train(preset: str, overrides: str, exp_name: str | None, tz: str):
    import os
    import subprocess
    import sys
    import threading
    import time

    os.chdir(PROJECT_ROOT)

    # Match the trainer's clock to the laptop so run-dir timestamps line up.
    if tz:
        os.environ["TZ"] = tz
        time.tzset()

    args = [
        sys.executable,
        "-m",
        "hexhex.training.repeated_self_training",
        f"preset={preset}",
    ]
    if exp_name:
        args.append(f"exp_name={exp_name}")
    if overrides:
        args.extend(overrides.split())

    print("nvidia-smi:")
    subprocess.run(["nvidia-smi"], check=False)
    print("running:", " ".join(args))

    # Commit every 20s so the TB container can see partial progress.
    stop_committer = threading.Event()

    def committer():
        while not stop_committer.wait(20):
            try:
                runs_volume.commit()
            except Exception as e:
                print("commit failed:", e)

    threading.Thread(target=committer, daemon=True).start()

    try:
        subprocess.run(args, check=True)
    finally:
        stop_committer.set()
        runs_volume.commit()


def _detect_local_tz() -> str:
    import os
    tz = os.environ.get("TZ")
    if tz:
        return tz
    try:
        link = os.readlink("/etc/localtime")
    except OSError:
        return ""
    marker = "zoneinfo/"
    return link.split(marker, 1)[1] if marker in link else ""


@app.local_entrypoint()
def main(preset: str = "5x5", overrides: str = "", exp_name: str = "", tz: str = ""):
    train.remote(
        preset=preset,
        overrides=overrides,
        exp_name=exp_name or None,
        tz=tz or _detect_local_tz(),
    )
