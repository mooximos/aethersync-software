"""The collector's tests need its packages (OpenCV, supervision, Flask, astral, smbus2).
On a laptop with only requirements-analysis.txt they are skipped (the root conftest says so)."""
import importlib.util

if any(importlib.util.find_spec(n) is None for n in ("cv2", "supervision", "flask", "astral", "smbus2")):
    collect_ignore_glob = ["test_*.py"]
