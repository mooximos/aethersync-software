"""Shared pytest setup: say clearly when the collector's tests are skipped."""
import importlib.util

COLLECTOR_PACKAGES = ("cv2", "supervision", "flask", "astral", "smbus2")


def missing_collector_packages() -> list[str]:
    return [name for name in COLLECTOR_PACKAGES if importlib.util.find_spec(name) is None]


def pytest_terminal_summary(terminalreporter):
    missing = missing_collector_packages()
    if missing:
        terminalreporter.write_line(
            f"NOTE: the collector tests were SKIPPED (missing {', '.join(missing)}). To run them: "
            "pip install -r requirements-collector.txt opencv-python-headless==5.0.0.93")
