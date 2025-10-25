#!/usr/bin/env python3
"""
Enhanced YOLO Detection Module
Provides standalone YOLO detection capabilities for benchmarking
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("Warning: Ultralytics YOLO not found. Detection functionality will be limited.")
    YOLO_AVAILABLE = False


class YOLODetector:
    """
    Standalone YOLO detector for benchmarking and analysis
    """
    
    def __init__(self, weights: str = "yolov5s.pt", device: str = "cpu", conf_thresh: float = 0.25, iou_thresh: float = 0.45):
        self.weights = weights
        self.device = device
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.model = None
        self.img_size = 640
        
        if YOLO_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """Load YOLO model"""
        try:
            self.model = YOLO(self.weights)
            self.model.to(self.device)
            
            print(f"YOLO model loaded: {self.weights}")
            print(f"Device: {self.device}")
            
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            self.model = None
    
    
    def detect_objects(self, image_path: Union[str, Path], save_results: bool = False, output_dir: Optional[str] = None) -> Dict:
        """
        Run YOLO detection on a single image
        
        Returns:
            Dictionary with detection results and timing information
        """
        if not YOLO_AVAILABLE or self.model is None:
            return {"error": "YOLO model not available"}
        
        image_path = Path(image_path)
        
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            return {"error": f"Could not load image: {image_path}"}
        
        start_time = time.time()
        
        # Use ultralytics YOLO model directly (simplified)
        inference_start = time.time()
        
        # Run inference
        results = self.model(str(image_path), conf=self.conf_thresh, device=self.device, verbose=False)
        
        inference_time = time.time() - inference_start
        
        # Process results
        postprocess_start = time.time()
        detections = []
        
        if results and len(results) > 0:
            result = results[0]  # First (and only) image result
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                
                for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                    x1, y1, x2, y2 = [int(x) for x in box]
                    detections.append({
                        "class": int(cls),
                        "class_name": result.names[cls],
                        "confidence": float(conf),
                        "bbox": [x1, y1, x2, y2],
                        "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                        "area": int((x2 - x1) * (y2 - y1))
                    })
        
        postprocess_time = time.time() - postprocess_start
        total_time = time.time() - start_time
        
        # Save results if requested
        if save_results and output_dir:
            self._save_detection_results(img, detections, image_path, output_dir)
        
        return {
            "image_path": str(image_path),
            "detections": detections,
            "detection_count": len(detections),
            "timing": {
                "preprocess": 0.0,  # Handled by ultralytics internally
                "inference": inference_time * 1000,    # ms
                "postprocess": postprocess_time * 1000, # ms
                "total": total_time * 1000             # ms
            },
            "model_info": {
                "weights": self.weights,
                "device": str(self.device),
                "conf_threshold": self.conf_thresh,
                "iou_threshold": self.iou_thresh
            }
        }
    
    def _save_detection_results(self, img: np.ndarray, detections: List[Dict], image_path: Path, output_dir: str):
        """Save detection results with bounding boxes"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Draw bounding boxes
        result_img = img.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            conf = detection["confidence"]
            class_name = detection["class_name"]
            
            # Draw bounding box
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_img, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(result_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Save image
        output_path = output_dir / f"detected_{image_path.name}"
        cv2.imwrite(str(output_path), result_img)
        
        # Save detection data as text
        txt_path = output_dir / f"{image_path.stem}.txt"
        with open(txt_path, 'w') as f:
            for detection in detections:
                bbox = detection["bbox"]
                # Convert to YOLO format (normalized xywh)
                img_h, img_w = img.shape[:2]
                x_center = (bbox[0] + bbox[2]) / 2 / img_w
                y_center = (bbox[1] + bbox[3]) / 2 / img_h
                width = (bbox[2] - bbox[0]) / img_w
                height = (bbox[3] - bbox[1]) / img_h
                
                f.write(f"{detection['class']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {detection['confidence']:.6f}\n")
        
        print(f"Detection results saved to: {output_dir}")
    
    def benchmark_detection(self, image_path: Union[str, Path], num_runs: int = 10) -> Dict:
        """
        Benchmark YOLO detection performance
        
        Args:
            image_path: Path to test image
            num_runs: Number of detection runs for averaging
            
        Returns:
            Benchmark results with timing statistics
        """
        if not YOLO_AVAILABLE or self.model is None:
            return {"error": "YOLO model not available"}
        
        print(f"Benchmarking YOLO detection with {num_runs} runs...")
        
        # Warm up run
        self.detect_objects(image_path)
        
        # Benchmark runs
        times = []
        detection_counts = []
        
        for i in range(num_runs):
            result = self.detect_objects(image_path)
            if "error" in result:
                return result
            
            times.append(result["timing"])
            detection_counts.append(result["detection_count"])
            
            if (i + 1) % 5 == 0:
                print(f"  Completed {i + 1}/{num_runs} runs")
        
        # Calculate statistics
        timing_stats = {}
        for timing_key in ["preprocess", "inference", "postprocess", "total"]:
            values = [t[timing_key] for t in times]
            timing_stats[timing_key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "median": np.median(values)
            }
        
        return {
            "image_path": str(image_path),
            "num_runs": num_runs,
            "detection_count": {
                "mean": np.mean(detection_counts),
                "std": np.std(detection_counts),
                "all_runs": detection_counts
            },
            "timing_stats": timing_stats,
            "model_info": {
                "weights": self.weights,
                "device": str(self.device),
                "conf_threshold": self.conf_thresh,
                "iou_threshold": self.iou_thresh
            }
        }


def main():
    """Demo and testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("-w", "--weights", default="yolov5s.pt", help="Model weights")
    parser.add_argument("-d", "--device", default="cpu", help="Device (cpu, cuda, mps)")
    parser.add_argument("-c", "--confidence", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("-o", "--output", help="Output directory for results")
    parser.add_argument("-b", "--benchmark", type=int, help="Run benchmark with N iterations")
    parser.add_argument("--save", action="store_true", help="Save detection results")
    
    args = parser.parse_args()
    
    detector = YOLODetector(
        weights=args.weights,
        device=args.device,
        conf_thresh=args.confidence
    )
    
    if args.benchmark:
        results = detector.benchmark_detection(args.image, args.benchmark)
        print("\n=== Benchmark Results ===")
        print(f"Image: {results['image_path']}")
        print(f"Runs: {results['num_runs']}")
        print(f"Average detections: {results['detection_count']['mean']:.1f}")
        print("\nTiming (ms):")
        for phase, stats in results['timing_stats'].items():
            print(f"  {phase:12}: {stats['mean']:6.1f} ± {stats['std']:5.1f} (min: {stats['min']:5.1f}, max: {stats['max']:5.1f})")
    else:
        results = detector.detect_objects(
            args.image,
            save_results=args.save,
            output_dir=args.output or "runs/detect/standalone"
        )

        # Check for errors
        if "error" in results:
            print(f"\n=== Detection Failed ===")
            print(f"Error: {results['error']}")
            sys.exit(1)

        print(f"\n=== Detection Results ===")
        print(f"Image: {results['image_path']}")
        print(f"Detections: {results['detection_count']}")
        print(f"Total time: {results['timing']['total']:.1f} ms")

        if results['detections']:
            print("\nDetected objects:")
            for i, det in enumerate(results['detections']):
                print(f"  {i+1}. {det['class_name']} ({det['confidence']:.2f}) at {det['bbox']}")


if __name__ == "__main__":
    main()