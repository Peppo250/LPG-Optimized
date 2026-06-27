"""
LPG Catering Project Runner
==========================
Utility script to orchestrate project phases: data pipeline, model training,
test suites, and hosting the REST API.

Usage:
    python run_project.py --pipeline    # Run data pipeline
    python run_project.py --train       # Train ML models
    python run_project.py --test        # Run test suite
    python run_project.py --api         # Start API server
    python run_project.py --all         # Run pipeline, train, and test in sequence
"""

import argparse
import subprocess
import sys
import os

def run_cmd(args_list, description):
    print(f"\n>>> Running: {description} ({' '.join(args_list)})")
    # Use python from the current virtual env if executing Python files
    result = subprocess.run(args_list)
    if result.returncode != 0:
        print(f"\nERROR: {description} failed with return code {result.returncode}")
        sys.exit(result.returncode)
    print(f"--- Completed: {description} successfully.")

def main():
    parser = argparse.ArgumentParser(description="LPG Catering Project Runner")
    parser.add_argument("--pipeline", action="store_true", help="Run the dataset pipeline")
    parser.add_argument("--train", action="store_true", help="Run model training")
    parser.add_argument("--test", action="store_true", help="Run test suite")
    parser.add_argument("--api", action="store_true", help="Start the FastAPI uvicorn server")
    parser.add_argument("--all", action="store_true", help="Run pipeline, train, and test in sequence")

    args = parser.parse_args()

    # Determine Python executable
    python_exe = sys.executable
    
    # If no flags are provided, show help
    if not (args.pipeline or args.train or args.test or args.api or args.all):
        parser.print_help()
        sys.exit(0)

    # 1. Run Pipeline
    if args.pipeline or args.all:
        run_cmd([python_exe, "data_pipeline.py"], "Data Processing Pipeline")

    # 2. Run Training
    if args.train or args.all:
        run_cmd([python_exe, "train_final.py"], "Model Training Pipeline")

    # 3. Run Tests
    if args.test or args.all:
        run_cmd([python_exe, "-m", "unittest", "discover", "tests"], "Test Suite Discovery")

    # 4. Start API Server
    if args.api:
        print("\n>>> Launching FastAPI REST Server...")
        cmd = [python_exe, "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
        if os.path.exists(".env"):
            cmd.extend(["--env-file", ".env"])
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\nAPI Server stopped by user.")

if __name__ == "__main__":
    main()
