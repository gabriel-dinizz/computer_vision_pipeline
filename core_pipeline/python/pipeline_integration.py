#!/usr/bin/env python3
"""
Streamlined Computer Vision Pipeline Integration
Coordinates C++ preprocessing with YOLO detection
"""

import subprocess
import time
import json
from pathlib import Path
from typing import Dict, Optional
import argparse

from yolo_detector import YOLODetector


class CVPipeline:
    """
    Streamlined computer vision pipeline integrating:
    1. C++ OpenMP preprocessing 
    2. Python YOLO detection
    """
    
    def __init__(self, preprocess_bin: str = "bin/preprocess_optimized"):
        self.preprocess_bin = Path(preprocess_bin)
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)
        
    def run_full_pipeline(self, image_path: str, filter_type: str = "auto",
                         confidence: float = 0.25, device: str = "cpu",
                         save_results: bool = True, output_dir: str = None) -> Dict:
        """
        Run complete pipeline: preprocessing + YOLO detection
        
        Returns:
            Dictionary with pipeline results and timing
        """
        image_path = Path(image_path)
        if not image_path.exists():
            return {"error": f"Image not found: {image_path}"}
            
        results = {
            "input_image": str(image_path),
            "filter_type": filter_type,
            "timing": {},
            "preprocessing": {},
            "detection": {}
        }
        
        # Step 1: C++ Preprocessing
        print(f"🔄 Step 1: Preprocessing with {filter_type} filter...")
        preprocess_start = time.time()
        
        preprocessed_path = self.temp_dir / f"preprocessed_{image_path.name}"
        
        try:
            cmd = [
                str(self.preprocess_bin),
                str(image_path),
                str(preprocessed_path),
                filter_type
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                return {"error": f"Preprocessing failed: {result.stderr}"}
                
            preprocess_time = time.time() - preprocess_start
            results["timing"]["preprocessing"] = preprocess_time * 1000  # ms
            results["preprocessing"]["success"] = True
            results["preprocessing"]["output_path"] = str(preprocessed_path)
            
        except subprocess.TimeoutExpired:
            return {"error": "Preprocessing timeout"}
        except Exception as e:
            return {"error": f"Preprocessing error: {e}"}
        
        # Step 2: YOLO Detection
        print("🔄 Step 2: YOLO object detection...")
        detection_start = time.time()
        
        detector = YOLODetector(device=device, conf_thresh=confidence)
        
        if save_results:
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                output_dir = self.temp_dir / "detection_results"
        else:
            output_dir = None
        detection_results = detector.detect_objects(
            preprocessed_path, 
            save_results=save_results, 
            output_dir=output_dir
        )
        
        if "error" in detection_results:
            return {"error": f"Detection failed: {detection_results['error']}"}
        
        detection_time = time.time() - detection_start
        results["timing"]["detection"] = detection_time * 1000  # ms
        results["timing"]["total"] = (preprocess_time + detection_time) * 1000  # ms
        
        results["detection"] = detection_results
        
        # Summary
        print(f"✅ Pipeline completed:")
        print(f"   Preprocessing: {results['timing']['preprocessing']:.1f} ms")
        print(f"   Detection: {results['timing']['detection']:.1f} ms")
        print(f"   Total: {results['timing']['total']:.1f} ms")
        print(f"   Objects detected: {detection_results['detection_count']}")
        
        if save_results and output_dir:
            print(f"   Results saved to: {output_dir}")
        
        return results
    
    def benchmark_pipeline(self, image_path: str, num_runs: int = 5) -> Dict:
        """
        Benchmark the complete pipeline performance
        """
        print(f"🔬 Benchmarking pipeline with {num_runs} runs...")
        
        times = []
        detection_counts = []
        
        # Warm-up run
        self.run_full_pipeline(image_path, save_results=False)
        
        for i in range(num_runs):
            result = self.run_full_pipeline(image_path, save_results=False)
            
            if "error" in result:
                return result
                
            times.append(result["timing"])
            detection_counts.append(result["detection"]["detection_count"])
            
            print(f"   Run {i+1}/{num_runs}: {result['timing']['total']:.1f} ms")
        
        # Calculate statistics
        import numpy as np
        
        timing_stats = {}
        for phase in ["preprocessing", "detection", "total"]:
            values = [t[phase] for t in times]
            timing_stats[phase] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values)
            }
        
        return {
            "benchmark_results": {
                "num_runs": num_runs,
                "timing_stats": timing_stats,
                "detection_count": {
                    "mean": np.mean(detection_counts),
                    "std": np.std(detection_counts)
                }
            }
        }


def main():
    parser = argparse.ArgumentParser(description="Streamlined CV Pipeline")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("-f", "--filter", default="auto", help="Preprocessing filter")
    parser.add_argument("-c", "--confidence", type=float, default=0.25, help="YOLO confidence")
    parser.add_argument("-d", "--device", default="cpu", help="Device (cpu, cuda, mps)")
    parser.add_argument("-b", "--benchmark", type=int, help="Run benchmark with N runs")
    parser.add_argument("--no-save", action="store_true", help="Don't save detection results")
    
    args = parser.parse_args()
    
    pipeline = CVPipeline()
    
    if args.benchmark:
        results = pipeline.benchmark_pipeline(args.image, args.benchmark)
        print("\n📊 Benchmark Results:")
        stats = results["benchmark_results"]["timing_stats"]
        for phase, values in stats.items():
            print(f"  {phase:12}: {values['mean']:6.1f} ± {values['std']:5.1f} ms")
    else:
        results = pipeline.run_full_pipeline(
            args.image,
            filter_type=args.filter,
            confidence=args.confidence,
            device=args.device,
            save_results=not args.no_save
        )
        
        if "error" in results:
            print(f"❌ Error: {results['error']}")
        else:
            # Save results
            results_file = Path("temp/pipeline_results.json")
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"📄 Results saved to: {results_file}")


if __name__ == "__main__":
    main()