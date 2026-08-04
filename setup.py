import subprocess
import sys


# Packages your project needs
REQUIRED_PACKAGES = [
    "pandas",

]


def install_packages():
    for package in REQUIRED_PACKAGES:
        try:
            __import__(package)
            print(f"{package} is already installed")

        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package]
            )


if __name__ == "__main__":
    install_packages()
    print("All packages installed successfully")