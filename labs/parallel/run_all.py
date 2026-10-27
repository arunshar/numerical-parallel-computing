"""Run the deterministic Python experiments without launching native programs."""
from pathlib import Path
import subprocess
import sys

if __name__ == "__main__":
    for script in sorted(Path(__file__).parent.glob("[0-9][0-9]-*.py")):
        subprocess.run([sys.executable, str(script)], check=True)
    print("All 14 parallel-track Python experiments passed.")
