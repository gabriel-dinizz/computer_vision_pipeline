#!/usr/bin/env python3
"""
Full Pipeline Wrapper - calls pipeline_integration.py
This provides compatibility for commands expecting full_pipeline.py
"""

import sys
import subprocess
from pathlib import Path

def main():
    # Get the directory of this script
    script_dir = Path(__file__).parent
    
    # Path to the actual pipeline integration script
    pipeline_script = script_dir / "python" / "pipeline_integration.py"
    
    # Forward all arguments to the real pipeline
    cmd = [sys.executable, str(pipeline_script)] + sys.argv[1:]
    
    # Execute the real pipeline
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
