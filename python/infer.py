import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "external" / "yolov5"))

from yolov5 import detect # type ignore

if __name__ == "__main__":
    detect.run(
        weights="yolov5s.pt",      # auto-downloads first run
        source="images/",          # put a sample image in images/
        device="cpu",              # CPU-only
        project="runs/detect",
        name="cpu_test",
        exist_ok=True,
        save_txt=True,
        save_conf=True,
    )
