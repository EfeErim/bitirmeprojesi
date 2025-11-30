import os
import sys
from pathlib import Path

def mount_drive(mount_path='/content/drive'):
    """
    Mounts Google Drive in Google Colab.
    
    Args:
        mount_path (str): The path where Google Drive will be mounted.
    """
    try:
        from google.colab import drive
        drive.mount(mount_path)
        print(f"Google Drive mounted at {mount_path}")
    except ImportError:
        print("Not running in Google Colab. Skipping Drive mount.")

def setup_project(repo_name='bitirmeprojesi', branch='main'):
    """
    Sets up the project environment in Colab:
    1. Mounts Drive
    2. Installs requirements
    """
    mount_drive()
    
    # Install requirements if requirements.txt exists
    req_path = Path('requirements.txt')
    if req_path.exists():
        print("Installing requirements...")
        os.system('pip install -r requirements.txt')
        print("Requirements installed.")
    else:
        print("requirements.txt not found.")

if __name__ == "__main__":
    setup_project()
