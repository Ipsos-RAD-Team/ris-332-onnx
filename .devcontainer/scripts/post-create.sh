#!/bin/bash
pip install -r requirements.txt
sudo usermod -aG docker $USER
aws ecr get-login-password --region $REGION | sudo docker login --username AWS --password-stdin $AWSID.dkr.ecr.$REGION.amazonaws.com
pip install ipykernel --trusted-host=pypi.python.org --trusted-host=pypi.org --trusted-host=files.pythonhosted.org


