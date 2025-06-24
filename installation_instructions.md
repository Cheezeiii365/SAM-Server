# SAM Server Installation Guide

This guide will help you set up the SAM (Style-based Age Manipulation) server on a Linux GPU instance.

## Prerequisites

- Linux system (Ubuntu 20.04+ recommended)
- NVIDIA GPU with CUDA support
- Python 3.10
- At least 8GB GPU memory (16GB+ recommended)
- Internet connection for downloading models

## Step 1: System Dependencies

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential build tools
sudo apt install -y cmake build-essential git wget curl

# Install Python development headers
sudo apt install -y python3.10-dev python3.10-venv python3.10-distutils libpython3.10-dev

# Install additional dependencies for computer vision
sudo apt install -y libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev pkg-config

# Verify NVIDIA drivers (should show GPU info)
nvidia-smi
```

## Step 2: Install Miniconda (for dlib)

```bash
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

# Add conda to PATH
export PATH="$HOME/miniconda3/bin:$PATH"
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Clean up installer
rm Miniconda3-latest-Linux-x86_64.sh
```

## Step 3: Create Python Environment

```bash
# Clone the SAM server repository
git clone https://github.com/Cheezeiii365/SAM-Server.git
cd SAM-Server

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

## Step 4: Install dlib via Conda

```bash
# Install dlib using conda (avoids build issues)
conda install -c conda-forge dlib

# Verify dlib installation
python -c "import dlib; print('dlib version:', dlib.version)"
```

## Step 5: Install Python Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://cuda.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt

# Verify PyTorch CUDA setup
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Step 6: Download Pretrained Models

```bash
# Create models directory
mkdir -p pretrained_models

# Download SAM pretrained model
gdown "https://drive.google.com/u/0/uc?id=1XyumF6_fdAxFmxpFcmPf-q84LU_22EMC&export=download" -O pretrained_models/sam_ffhq_aging.pt

# Download face landmarks model
wget "https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat" -O shape_predictor_68_face_landmarks.dat

# Verify model downloads
ls -la pretrained_models/
ls -la shape_predictor_68_face_landmarks.dat
```

## Step 7: Test Installation

```bash
# Test basic imports
python -c "
import torch
import torchvision
import numpy as np
from PIL import Image
import dlib
import cv2
print('✓ All core dependencies imported successfully!')
print('✓ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✓ GPU:', torch.cuda.get_device_name(0))
    print('✓ GPU Memory:', f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
"

# Test SAM model loading (this will take a moment)
python -c "
import torch
from models.psp import pSp
from argparse import Namespace

# Basic test configuration
opts = Namespace(
    checkpoint_path='pretrained_models/sam_ffhq_aging.pt',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

try:
    ckpt = torch.load('pretrained_models/sam_ffhq_aging.pt', map_location='cpu')
    print('✓ SAM model checkpoint loaded successfully!')
    print('✓ Model size:', f'{len(str(ckpt))/1e6:.1f}MB')
except Exception as e:
    print('✗ Error loading SAM model:', e)
"
```

## Step 8: Environment Variables (Optional)

```bash
# Create .env file for configuration
cat > .env << EOF
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_KEY=your-secret-api-key-here

# Model Configuration
MODEL_PATH=pretrained_models/sam_ffhq_aging.pt
LANDMARKS_PATH=shape_predictor_68_face_landmarks.dat

# Processing Configuration
MAX_IMAGE_SIZE=1024
TARGET_AGE=25
GPU_MEMORY_FRACTION=0.8
EOF
```

## Troubleshooting

### CUDA Issues
- **"CUDA not available"**: Check `nvidia-smi` and reinstall PyTorch with correct CUDA version
- **Out of memory**: Reduce batch size or image resolution

### dlib Issues
- **Build failures**: Always install dlib via conda, not pip
- **Import errors**: Make sure conda environment is activated

### Model Download Issues
- **gdown failures**: Try downloading manually from Google Drive
- **Slow downloads**: Use wget with resume: `wget -c [url]`

### Permission Issues
- **Access denied**: Check file permissions with `ls -la`
- **Port binding**: Use `sudo` or change port in configuration

## Next Steps

After successful installation:

1. **Test inference**: Run the provided test scripts
2. **Start API server**: Launch the FastAPI server
3. **Test API endpoints**: Send test images to verify functionality
4. **Monitor performance**: Check GPU usage and response times

## System Requirements Met ✓

- [x] Linux system with GPU support
- [x] Python 3.10 environment
- [x] CUDA-enabled PyTorch
- [x] dlib for face processing
- [x] SAM model downloaded
- [x] API dependencies installed

Your SAM server should now be ready for age regression inference!