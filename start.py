"""One-command setup and demo runner. First run downloads packages from PyPI."""
from pathlib import Path
import subprocess
import sys
import venv


def main():
    root = Path(__file__).resolve().parent
    env = root / ".venv"
    python = env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if sys.version_info < (3, 10):
        raise SystemExit("Python 3.10 or newer is required; Python 3.12 is recommended.")
    if not python.exists():
        print("Creating a local Python environment...", flush=True)
        venv.EnvBuilder(with_pip=True).create(env)
    requirements = root / "requirements.txt"
    marker = env / "neurotwin-requirements.txt"
    if not marker.exists() or marker.read_bytes() != requirements.read_bytes():
        subprocess.run([str(python), "-m", "pip", "install", "-r", str(requirements)], check=True)
        marker.write_bytes(requirements.read_bytes())
    subprocess.run([str(python), str(root / "run.py"), *sys.argv[1:]], cwd=root, check=True)


if __name__ == "__main__":
    main()
