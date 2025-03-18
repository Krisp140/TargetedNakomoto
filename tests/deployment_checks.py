import os
import sys
import subprocess
import platform
import json

def check_environment():
    """Check Python version and environment"""
    print(f"Python version: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    
    # Check for virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    print(f"Running in virtual environment: {'Yes' if in_venv else 'No'}")
    if not in_venv:
        print("⚠️ Warning: Not running in a virtual environment")

def check_requirements():
    """Check requirements.txt for deployment issues"""
    with open("../requirements.txt", "r") as f:
        requirements = f.readlines()
    
    for req in requirements:
        req = req.strip()
        if req and not req.startswith("#"):
            if "==" not in req and ">=" not in req:
                print(f"⚠️ Warning: {req} has no version specified, may cause issues in deployment")

def check_files_exist():
    """Check if all critical files exist"""
    critical_files = [
        "app.py",
        "api.py", 
        "requirements.txt",
        "data/merged_data.csv",
        "src/models/blockchain.py",
        "src/simulation/engine.py"
    ]
    
    for file in critical_files:
        if not os.path.exists(file):
            print(f"❌ Error: Critical file {file} not found")
        else:
            print(f"✅ Found critical file: {file}")

def check_streamlit_config():
    """Check Streamlit configuration"""
    config_path = os.path.expanduser("~/.streamlit/config.toml")
    if os.path.exists(config_path):
        print(f"Found Streamlit config at {config_path}")
    else:
        print("⚠️ No Streamlit config found. Default settings will be used.")

def main():
    print("Running deployment readiness checks...\n")
    
    print("\n=== Environment Check ===")
    check_environment()
    
    print("\n=== Requirements Check ===")
    check_requirements()
    
    print("\n=== Critical Files Check ===")
    check_files_exist()
    
    print("\n=== Streamlit Config Check ===")
    check_streamlit_config()
    
    print("\nDeployment readiness check complete.")

if __name__ == "__main__":
    main()