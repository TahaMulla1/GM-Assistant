from setuptools import setup, find_packages
import platform
import subprocess
import sys

def install_optimal_dependencies():
    """Install the optimal PyTorch and dependencies for the user's hardware"""
    
    # Detect hardware type
    system = platform.system()
    machine = platform.machine()
    
    # Mac with Apple Silicon
    if system == "Darwin" and machine == "arm64":
        print("Detected Apple Silicon Mac, installing compatible PyTorch...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio"
        ])
        return
    
    # Check for NVIDIA GPU on Linux/Windows
    try:
        nvidia_present = False
        
        if system == "Linux":
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            nvidia_present = result.returncode == 0
        elif system == "Windows":
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, shell=True)
            nvidia_present = result.returncode == 0
            
        if nvidia_present:
            print("NVIDIA GPU detected, installing CUDA PyTorch...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio"
            ])
            return
    except Exception:
        pass
    
    # Check for AMD GPU on Linux
    try:
        if system == "Linux":
            result = subprocess.run(["rocm-smi"], capture_output=True, text=True)
            if result.returncode == 0:
                print("AMD GPU with ROCm detected, attempting to install ROCm PyTorch...")
                # Try multiple ROCm versions, starting with latest
                rocm_versions = ["5.7", "5.6", "5.5", "5.4.2", "5.3", "5.2", "5.1.1", "5.0"]
                
                for version in rocm_versions:
                    try:
                        print(f"Trying ROCm version {version}...")
                        subprocess.check_call([
                            sys.executable, "-m", "pip", "install", 
                            "torch", "torchvision", "torchaudio",
                            "--index-url", f"https://download.pytorch.org/whl/rocm{version}"
                        ])
                        print(f"Successfully installed PyTorch with ROCm {version}")
                        return
                    except subprocess.CalledProcessError:
                        print(f"ROCm {version} installation failed, trying next version...")
                        continue
                
                # If all ROCm versions fail, try the default PyPI version
                # which might work with ROCm if compatible version exists
                try:
                    print("Trying PyTorch from PyPI which may work with ROCm...")
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", 
                        "torch", "torchvision", "torchaudio"
                    ])
                    print("Installed PyTorch from PyPI. Please verify ROCm compatibility.")
                    return
                except subprocess.CalledProcessError:
                    print("Could not install PyTorch from PyPI")
                    
                print("Could not find a compatible ROCm version, falling back to CPU version...")
    except Exception:
        pass
    
    # Default to CPU version
    print("No compatible GPU detected or GPU drivers not found, installing CPU PyTorch...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cpu"
    ])

if __name__ == "__main__":
    install_optimal_dependencies()

setup(
    name="universal_gpu_rag",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A universal GPU-compatible RAG system",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/universal_gpu_rag",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "transformers>=4.30.0",
        "sentence-transformers>=2.2.0",
        "langchain>=0.0.267",
        "chromadb>=0.4.6",
        "pypdf",
        "markdown",
        "bitsandbytes>=0.39.0",  # For 8-bit quantization
        "accelerate>=0.20.0",    # For device mapping
        "einops",                # Common dependency for many models
        "safetensors",           # For safer model loading
    ],
    entry_points={
        "console_scripts": [
            "install_optimal_dependencies=setup:install_optimal_dependencies",
        ],
    },
)