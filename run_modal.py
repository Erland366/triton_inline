from pathlib import Path
import shlex
from typing import Literal, Optional
from pprint import pprint

import modal

CURRENT_DIR = Path(__file__).parent
REMOTE_DIR = Path("/app")
DEFAULT_ARTIFACT_GLOBS = ",".join(
    [
        "**/*.png",
        "**/*.chrome_trace",
        "**/*.hatchet",
    ]
)
MACHINE_ARCH = "H100:1"

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
    .uv_pip_install("matplotlib", "pandas", "expecttest")
    .add_local_dir(
        CURRENT_DIR,
        remote_path=REMOTE_DIR,
        ignore=IGNORE_PATTERNS
    )
)
app = modal.App("TLX-learning", image=image)


def normalize_script_command(script: str) -> str:
    """Map local absolute script paths into the remote /app mount."""
    command_parts = shlex.split(script)
    if not command_parts:
        raise ValueError("--script must not be empty")

    script_path = Path(command_parts[0]).expanduser()
    if script_path.is_absolute():
        try:
            script_path = script_path.resolve().relative_to(CURRENT_DIR.resolve())
        except ValueError as exc:
            raise ValueError(
                f"--script path must be inside {CURRENT_DIR} when using an absolute path: {script_path}"
            ) from exc
        command_parts[0] = script_path.as_posix()

    return shlex.join(command_parts)

@app.function(gpu=MACHINE_ARCH)
def run_script(script: str, artifact_globs: list[str], max_artifact_bytes: int, 
               env_vars: dict[str, str], action: Literal["run", "pytest", "distributed"]):
    f"""
    Run a Python script on {MACHINE_ARCH} GPU.
    """
    import subprocess
    import os
    import shlex

    os.chdir(REMOTE_DIR)

    def iter_artifact_paths():
        seen_paths = set()
        for pattern in artifact_globs:
            for artifact_path in sorted(REMOTE_DIR.glob(pattern)):
                if not artifact_path.is_file():
                    continue

                relative_path = artifact_path.relative_to(REMOTE_DIR).as_posix()
                if relative_path in seen_paths:
                    continue

                seen_paths.add(relative_path)
                yield relative_path, artifact_path

    initial_artifacts = {
        relative_path: (artifact_path.stat().st_mtime_ns, artifact_path.stat().st_size)
        for relative_path, artifact_path in iter_artifact_paths()
    }

    if action == "distributed":
        machine_arch_parts = MACHINE_ARCH.split(":")
        if len(machine_arch_parts) > 0:
            num_gpus = machine_arch_parts[1]

    env = os.environ.copy()
    env.update(env_vars)
    if action == "run":
        cmd = ["python", *shlex.split(script)]
    elif action == "distributed":
        cmd = ["torchrun", "--standalone", "--nnodes=1", f"--nproc-per-node={num_gpus}", *shlex.split(script)]
    elif action == "pytest":
        cmd = ["python", "-m", "pytest", *shlex.split(script)]
    else:
        raise ValueError(f"Action={action} is not recognizable!")

    print(f"Running: {shlex.join(cmd)}")
    print(f"-"*60)

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
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
        "artifact_globs": artifact_globs,
    }

    for relative_path, artifact_path in iter_artifact_paths():
        artifact_stat = artifact_path.stat()
        artifact_state = (artifact_stat.st_mtime_ns, artifact_stat.st_size)
        if initial_artifacts.get(relative_path) == artifact_state:
            continue

        artifact_size = artifact_stat.st_size
        if artifact_size > max_artifact_bytes:
            load["artifact_errors"].append(
                f"{relative_path} is {artifact_size} bytes, above --max-artifact-bytes={max_artifact_bytes}"
            )
            continue

        load["artifacts"][relative_path] = artifact_path.read_bytes()

    return load


# @app.function(gpu=MACHINE_ARCH) # Uncomment if you actually want to use it
def debug_script(script: str, env_vars: dict[str, str]):
    f"""
    Run a Python script in-process on {MACHINE_ARCH} GPU for breakpoint debugging.
    """
    import os
    import runpy
    import shlex
    import sys

    os.chdir(REMOTE_DIR)
    os.environ.update(env_vars)

    command_parts = shlex.split(script)
    if not command_parts:
        raise ValueError("--script must not be empty")

    script_path = Path(command_parts[0])
    script_args = command_parts[1:]
    remote_script_path = script_path if script_path.is_absolute() else REMOTE_DIR / script_path
    script_dir = str(remote_script_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    print(f"Debugging in-process: python {shlex.join(command_parts)}")
    print("Use `modal run -i ... --action debug ...`; breakpoint() will attach to this terminal.")
    modal.interact()

    sys.argv = [str(script_path), *script_args]
    runpy.run_path(str(remote_script_path), run_name="__main__")


# @app.function(gpu="H100") # Uncomment if you actually want to use it
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

# @app.function(gpu="H100") # Uncomment if you actually want to use it
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


def parse_env_vars(env_vars: Optional[str]) -> dict[str, str]:
    if not env_vars:
        return {}

    parsed = {}
    for entry in env_vars.split(","):
        entry = entry.strip()
        if not entry:
            continue

        key, sep, value = entry.partition("=")
        if not sep or not key:
            raise ValueError(f"--env-vars must be KEY=VALUE, got: {entry}")
        parsed[key] = value

    return parsed

@app.local_entrypoint()
def main(
    action: Literal["run", "test", "pytest", "debug", "test_image", "distributed"],
    script: str = None,
    artifact_globs: str = DEFAULT_ARTIFACT_GLOBS,
    artifact_dir: str = ".",
    max_artifact_bytes: int = 100_000_000,
    env_vars: Optional[str] = None,
):
    if not script:
        print(f"Error: --script required for action ={action}")
        print(f"Usage: modal run run_modal.py --action {action} --script examples/my_kernel.py")
        return

    normalized_script = normalize_script_command(script)
    env_vars = parse_env_vars(env_vars)

    print(f"="*60)
    print(f"Running {normalized_script} on Modal {MACHINE_ARCH}")
    print(f"="*60)

    if action == "test_image":
        print(f"="*60)
        print(f"Testing The Image on Modal H100")
        print(f"="*60)

        test_image.remote()
    elif action in ["run", "pytest", "distributed", "test"]:
        if action == "test":
            action = "pytest"
        machine_arch_parts = MACHINE_ARCH.split(":")

        if action == "run" and len(machine_arch_parts) > 0:
            if int(machine_arch_parts[1]) > 1:
                print(f"Error: GPU usage more than 1 is detected for action ={action}")
                print(f"You might want to use action =distributed instead")
                print(f"Usage: modal run run_modal.py --action distributed --script examples/my_kernel.py")
                return

        artifact_patterns = [pattern.strip() for pattern in artifact_globs.split(",") if pattern.strip()]
        result = run_script.remote(normalized_script, artifact_patterns, max_artifact_bytes, env_vars, action)

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
    elif action == "debug":
        print(f"Make sure to use -i! (eg. modal run -i run_modal.py --action debug --script examples/my_kernel.py)")

        debug_script.remote(normalized_script, env_vars)