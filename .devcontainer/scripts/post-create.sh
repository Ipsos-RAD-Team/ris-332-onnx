#!/bin/bash
pip install -r requirements.txt
sudo usermod -aG docker $USER
aws ecr get-login-password --region $REGION | sudo docker login --username AWS --password-stdin $AWSID.dkr.ecr.$REGION.amazonaws.com
pip install ipykernel --trusted-host=pypi.python.org --trusted-host=pypi.org --trusted-host=files.pythonhosted.org

# # Install CUDA Toolkit
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
# sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
# wget https://developer.download.nvidia.com/compute/cuda/11.4.1/local_installers/cuda-repo-ubuntu2004-11-4-local_11.4.1-470.57.02-1_amd64.deb
# sudo dpkg -i cuda-repo-ubuntu2004-11-4-local_11.4.1-470.57.02-1_amd64.deb
# sudo apt-key add /var/cuda-repo-ubuntu2004-11-4-local/7fa2af80.pub
# sudo apt-get update
# sudo apt-get -y install cuda

# # Install cuDNN
# wget https://developer.download.nvidia.com/compute/cuda/11.4.1/local_installers/cuda_11.4.1_470.57.02_linux.run
# sudo sh cuda_11.4.1_470.57.02_linux.run --silent --toolkit --override

# # Set environment variables
# echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
# echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc


