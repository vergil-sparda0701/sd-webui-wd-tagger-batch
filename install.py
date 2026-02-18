"""
install.py
Executed by SD WebUI during extension installation to install Python dependencies.
"""

import subprocess
import sys
import os

REQUIREMENTS = [
    "onnxruntime>=1.16.0",
    "huggingface-hub>=0.19.0",
]

def install_package(pkg: str):
    print(f"[WD Tagger Batch] Installing {pkg} …")
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])


def check_and_install():
    for pkg in REQUIREMENTS:
        pkg_name = pkg.split(">=")[0].split("==")[0]
        try:
            __import__(pkg_name.replace("-", "_"))
        except ImportError:
            try:
                install_package(pkg)
            except Exception as e:
                print(f"[WD Tagger Batch] Warning: could not install {pkg}: {e}")


check_and_install()
print("[WD Tagger Batch] Dependencies OK.")
