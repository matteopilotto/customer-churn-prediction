#!/bin/bash

# update package lists
sudo apt-get update

# install python and venv
sudo apt-get install -y python3-pip python3-venv

# create virtual environment
python3 -m venv myvenv

# activate virtual environment
source myvenv/bin/activate

# install python libraries
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
else
    echo "[INFO] requirements.txt not found. Skipping package installation."
fi

echo "[INFO] Setup complete. Virtual environment is activated."