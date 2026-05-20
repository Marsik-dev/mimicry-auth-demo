"""CLI entry point: launches Streamlit via subprocess so relative imports work."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run() -> None:
    app = Path(__file__).parent.parent.parent / "streamlit_app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app), "--server.headless", "true"],
        check=True,
    )
