#!/usr/bin/env python3
"""
Enhanced Computer Vision Pipeline
Integrates C++ preprocessing with YOLO object detection
"""

import subprocess
import sys
import os
import time
import argparse
from pathlib import Path
import cv2
import numpy as np

# Add YOLOv5 to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "external" / "yolov5"))

try:
    from yolov5 import detect
except ImportError:
    print("Warning: YOLOv5 not found. Detection functionality will be limited.")
    detect = None

class VisionPipeline:
    def __init__(self, project_root=None):
        if project_root is None:
            self.project_root = Path(__file__).resolve().parents[1]
        else:
            self.project_root = Path(project_root)
        
        self.preprocess_bin = self.project_root / "bin" / "preprocess"
        self.temp_dir = self.project_root / "temp"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Check if preprocessing binary exists
        if not self.preprocess_bin.exists():
            print(f"Warning: Preprocessing binary not found at {self.preprocess_bin}")
            print("Run 'make' in the project root to build it.")
    
    def assess_image_quality(self, image_path):
        """
        Assess image quality and suggest preprocessing strategy
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate blur (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate brightness
        brightness = np.mean(gray)
        
        # Calculate contrast (standard deviation)
        contrast = np.std(gray)
        
        # Calculate noise level
        blur_kernel = cv2.getGaussianKernel(5, 1.0)
        blurred = cv2.filter2D(gray, -1, blur_kernel)
        noise_level = np.std(gray - blurred)
        
        metrics = {
            'blur_variance': laplacian_var,
            'brightness': brightness,
            'contrast': contrast,
            'noise_level': noise_level,
            'resolution': (img.shape[1], img.shape[0])
        }
        
        # Determine quality assessment
        quality_issues = []
        recommended_filter = "auto"
        
        if laplacian_var < 100:
            quality_issues.append("blurry")
            recommended_filter = "sharpen"
        if noise_level > 15:
            quality_issues.append("noisy")
            recommended_filter = "denoise"
        if brightness < 50:
            quality_issues.append("too_dark")
            recommended_filter = "clahe"
        elif brightness > 200:
            quality_issues.append("too_bright")
            recommended_filter = "clahe"
        if contrast < 30:
            quality_issues.append("low_contrast")
            recommended_filter = "clahe"
        
        assessment = {
            'metrics': metrics,
            'quality_issues': quality_issues,
            'recommended_filter': recommended_filter,
            'overall_quality': 'good' if not quality_issues else 'needs_enhancement'
        }
        
        return assessment
    
    def preprocess_image(self, input_path, output_path=None, filter_type="auto", verbose=True):
        """
        Apply C++ preprocessing to image
        """
        if output_path is None:
            output_path = self.temp_dir / f"preprocessed_{Path(input_path).name}"
        
        if not self.preprocess_bin.exists():
            # Fallback to Python preprocessing if C++ binary not available
            return self._python_fallback_preprocess(input_path, output_path, filter_type)
        
        cmd = [str(self.preprocess_bin), str(input_path), str(output_path)]
        
        if filter_type != "auto":
            cmd.append(filter_type)
        else:
            cmd.append("auto")
        
        if verbose:
            print(f"Running preprocessing: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                print(f"Preprocessing failed: {result.stderr}")
                return None
            
            if verbose and result.stdout:
                print("Preprocessing output:")
                print(result.stdout)
            
            return output_path if Path(output_path).exists() else None
            
        except subprocess.TimeoutExpired:
            print("Preprocessing timed out")
            return None
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None
    
    def _python_fallback_preprocess(self, input_path, output_path, filter_type):
        """
        Fallback Python preprocessing if C++ binary not available
        """
        print("Using Python fallback preprocessing...")
        
        img = cv2.imread(str(input_path))
        if img is None:
            return None
        
        if filter_type == "sharpen" or filter_type == "auto":
            # Unsharp masking
            blurred = cv2.GaussianBlur(img, (0, 0), 1.0)
            sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
            processed = sharpened
        elif filter_type == "denoise":
            processed = cv2.bilateralFilter(img, 9, 75, 75)
        elif filter_type == "clahe":
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(lab[:,:,0])
            processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:  # blur or default
            processed = cv2.GaussianBlur(img, (5, 5), 1.0)
        
        cv2.imwrite(str(output_path), processed)
        return output_path
    
    def run_yolo_detection(self, image_path, output_dir=None, confidence=0.25, device="cpu"):
        """
        Run YOLO object detection on preprocessed image
        """
        if detect is None:
            print("YOLO detection not available. Please install YOLOv5 dependencies.")
            return None
        
        if output_dir is None:
            output_dir = self.project_root / "runs" / "detect" / "pipeline_results"
        
        weights_path = self.project_root / "yolov5s.pt"
        
        try:
            print(f"Running YOLO detection on {image_path}")
            
            results = detect.run(
                weights=str(weights_path),
                source=str(image_path),
                device=device,
                project=str(output_dir.parent),
                name=output_dir.name,
                exist_ok=True,
                save_txt=True,
                save_conf=True,
                conf_thres=confidence,
                verbose=False
            )
            
            return output_dir
            
        except Exception as e:
            print(f"YOLO detection error: {e}")
            return None
    
    def run_full_pipeline(self, input_path, output_dir=None, filter_type="auto", 
                         confidence=0.25, device="cpu", verbose=True):
        """
        Run the complete pipeline: assess -> preprocess -> detect
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input image not found: {input_path}")
        
        print("=== Computer Vision Pipeline ===")
        print(f"Input: {input_path}")
        
        start_time = time.time()
        
        # Step 1: Assess image quality
        if verbose:
            print("\n1. Assessing image quality...")
        
        try:
            assessment = self.assess_image_quality(input_path)
            
            if verbose:
                print(f"   Resolution: {assessment['metrics']['resolution']}")
                print(f"   Blur variance: {assessment['metrics']['blur_variance']:.1f}")
                print(f"   Brightness: {assessment['metrics']['brightness']:.1f}")
                print(f"   Contrast: {assessment['metrics']['contrast']:.1f}")
                print(f"   Noise level: {assessment['metrics']['noise_level']:.1f}")
                print(f"   Quality: {assessment['overall_quality']}")
                if assessment['quality_issues']:
                    print(f"   Issues: {', '.join(assessment['quality_issues'])}")
                print(f"   Recommended filter: {assessment['recommended_filter']}")
            
            # Use recommended filter if auto mode
            if filter_type == "auto":
                filter_type = assessment['recommended_filter']
                
        except Exception as e:
            print(f"Quality assessment failed: {e}")
            assessment = None
        
        # Step 2: Preprocess image
        if verbose:
            print(f"\n2. Preprocessing with filter: {filter_type}")
        
        preprocessed_path = self.preprocess_image(
            input_path, 
            filter_type=filter_type, 
            verbose=verbose
        )
        
        if preprocessed_path is None:
            print("Preprocessing failed!")
            return None
        
        preprocess_time = time.time() - start_time
        
        # Step 3: YOLO detection
        if verbose:
            print(f"\n3. Running YOLO object detection...")
        
        detection_results = self.run_yolo_detection(
            preprocessed_path,
            output_dir=output_dir,
            confidence=confidence,
            device=device
        )
        
        total_time = time.time() - start_time
        
        # Summary
        print(f"\n=== Pipeline Complete ===")
        print(f"Preprocessing time: {preprocess_time:.2f}s")
        print(f"Total time: {total_time:.2f}s")
        
        if detection_results:
            print(f"Results saved to: {detection_results}")
        
        return {
            'assessment': assessment,
            'preprocessed_image': preprocessed_path,
            'detection_results': detection_results,
            'processing_time': {
                'preprocessing': preprocess_time,
                'total': total_time
            }
        }

def main():
    parser = argparse.ArgumentParser(description="Computer Vision Pipeline with Preprocessing and YOLO Detection")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("-o", "--output", help="Output directory for results")
    parser.add_argument("-f", "--filter", default="auto", 
                       choices=["auto", "blur", "sharpen", "denoise", "clahe", "edge"],
                       help="Preprocessing filter type")
    parser.add_argument("-c", "--confidence", type=float, default=0.25,
                       help="YOLO confidence threshold")
    parser.add_argument("-d", "--device", default="cpu",
                       help="Device for YOLO inference (cpu, cuda, mps)")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Verbose output")
    parser.add_argument("--assess-only", action="store_true",
                       help="Only assess image quality, don't run full pipeline")
    
    args = parser.parse_args()
    
    pipeline = VisionPipeline()
    
    if args.assess_only:
        assessment = pipeline.assess_image_quality(args.input)
        print("\n=== Image Quality Assessment ===")
        for key, value in assessment['metrics'].items():
            print(f"{key}: {value}")
        print(f"Overall quality: {assessment['overall_quality']}")
        print(f"Quality issues: {assessment['quality_issues']}")
        print(f"Recommended filter: {assessment['recommended_filter']}")
    else:
        results = pipeline.run_full_pipeline(
            args.input,
            output_dir=args.output,
            filter_type=args.filter,
            confidence=args.confidence,
            device=args.device,
            verbose=args.verbose
        )

if __name__ == "__main__":
    main()
