from pathlib import Path
from typing import Literal
from pprint import pprint

import modal

CURRENT_DIR = Path(__file__).parent
REMOTE_DIR = Path("/app")
DEFAULT_ARTIFACT_GLOBS = ",".join(
    [
        "plots/vector-add-performance.png",
        "*.chrome_trace",
        "*.hatchet",
    ]
)

IGNORE_PATTERNS = [
    ".venv",
    "__pycache__",
    ".git",
    "*.pyc",
    ".codex",
    ".claude",
    ".claude-plugin",
    "benchmark_results",
    "references",
    "templates",
    "compiled_resources",
]

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git", "build-essential", "cmake")
    .uv_pip_install("torch==2.11.0", index_url="https://download.pytorch.org/whl/cu128")
    .uv_pip_install("pytest", "numpy", "ninja")
    .apt_install("zlib1g-dev")
    .run_commands(
        "git clone --depth 1 https://github.com/facebookexperimental/triton.git /opt/triton",
        "cd /opt/triton && /.uv/uv pip install --python $(command -v python) --compile-bytecode -r python/requirements.txt",
        "cd /opt/triton && /.uv/uv pip install --python $(command -v python) --compile-bytecode -e . ",
    )
    .uv_pip_install("matplotlib", "pandas")
    .add_local_dir(
        CURRENT_DIR,
        remote_path=REMOTE_DIR,
        ignore=IGNORE_PATTERNS
    )
)
app = modal.App("TLX-learning", image=image)

@app.function(gpu="H100")
def run_script(script: str, artifact_globs: list[str], max_artifact_bytes: int):
    """
    Run a Python script on H100 GPU.
    """
    import subprocess
    import os
    import shlex

    os.chdir(REMOTE_DIR)

    cmd = ["python", *shlex.split(script)]
    print(f"Running: {' '.join(cmd)}")
    print(f"-"*60)

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"STDERR: {result.stderr}")

    load = {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0,
        "artifacts": {},
        "artifact_errors": [],
    }

    seen_artifacts = set()
    for pattern in artifact_globs:
        for artifact_path in sorted(REMOTE_DIR.glob(pattern)):
            if not artifact_path.is_file():
                continue

            relative_path = artifact_path.relative_to(REMOTE_DIR).as_posix()
            if relative_path in seen_artifacts:
                continue
            seen_artifacts.add(relative_path)

            artifact_size = artifact_path.stat().st_size
            if artifact_size > max_artifact_bytes:
                load["artifact_errors"].append(
                    f"{relative_path} is {artifact_size} bytes, above --max-artifact-bytes={max_artifact_bytes}"
                )
                continue

            load["artifacts"][relative_path] = artifact_path.read_bytes()

    return load

@app.function(gpu="H100")
def test_image():
    import os
    import subprocess
    import torch # type: ignore
    import triton  # type: ignore
    import triton.language as tl # type: ignore
    import triton.language.extra as tlx # type: ignore

    print("Python OK")
    print(f"Torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}")
    print(f"Triton: {triton.__file__}")
    print(f"TLX: {tlx.__file__}")

    subprocess.run(["python", "-m", "pip", "show", "triton"], check=False)
    subprocess.run(["nvidia-smi"], check=False)

@app.function(gpu="H100")
def interactive_session():
    """
    Start an interactive session for debugging.
    Use with: modal shell run_modal.py::interactive_session
    """
    import os
    import subprocess

    os.chdir(REMOTE_DIR)

    print("Interactive session on H100 started!")
    print(f"Working directory: {os.getcwd()}")

    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print(result.stdout)

    # Show PyTorch + CUDA info
    import torch # type: ignore

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"SM capability: {torch.cuda.get_device_capability(0)}")

    return {"status": "ready", "cwd": os.getcwd()}




@app.local_entrypoint()
def main(
    action: Literal["run", "test_image"],
    script: str = None,
    artifact_globs: str = DEFAULT_ARTIFACT_GLOBS,
    artifact_dir: str = ".",
    max_artifact_bytes: int = 100_000_000,
    verbose: bool = True,
):
    if action == "test_image":
        print(f"="*60)
        print(f"Testing The Image on Modal H100")
        print(f"="*60)

        test_image.remote()
    elif action == "run":
        if not script:
            print(f"Error: --script required for action =run")
            print(f"Usage: modal run run_modal.py --action run --script examples/my_kernel.py")
            return
        
        print(f"="*60)
        print(f"Running {script} on Modal H100")
        print(f"="*60)

        artifact_patterns = [pattern.strip() for pattern in artifact_globs.split(",") if pattern.strip()]
        result = run_script.remote(script, artifact_patterns, max_artifact_bytes)

        print("\n" + "="*60)
        if result["success"]:
            print("Completed successfully!")
        else:
            print(f"Failed! (return code: {result['returncode']})")
        
        display_result = dict(result)
        display_result["artifacts"] = {
            path: f"<{len(contents)} bytes>"
            for path, contents in display_result.get("artifacts", {}).items()
        }
        pprint(display_result)

        artifacts = result.get("artifacts", {})
        if artifacts:
            artifact_root = Path(artifact_dir)
            for relative_path, contents in artifacts.items():
                local_path = artifact_root / relative_path
                local_path.parent.mkdir(parents=True, exist_ok=True)
                local_path.write_bytes(contents)
                print(f"Saved artifact to {local_path}")
        elif result["success"]:
            print("No matching artifacts were returned.")
