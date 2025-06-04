## Use a Python image with CUDA (if GPU support is needed)
#FROM nvidia/cuda:12.1.1-base-ubuntu22.04
#
## Install system dependencies
#RUN apt-get update && apt-get install -y \
#    python3 python3-pip git ffmpeg libgl1 \
#    && rm -rf /var/lib/apt/lists/*
#
## Set working directory
#WORKDIR /app
#
## Copy everything into the container
#COPY . .
##COPY models/ ./models/
#
## Install Python dependencies
#RUN pip3 install --upgrade pip
#RUN pip3 install -r requirements.txt
#
## Expose the port your Flask app runs on
#EXPOSE 6000
#
## Run your Flask server
#CMD ["python3", "server.py"]


# Use a base image with Python and CUDA support
FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git ffmpeg libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy all files into the container (excluding those ignored by .dockerignore if present)
COPY . .
RUN test -f /app/models/model-5_14000_vpw.keras && echo "✅ Model file exists." || echo "❌ Model file is missing!"
RUN test -f /app/models/label_encoder_model-5_14000_vpw.pkl && echo "✅ Label encoder exists." || echo "❌ Label encoder is missing!"

# Ensure pip is up to date
RUN pip3 install --upgrade pip

# Install dependencies
RUN pip3 install -r requirements.txt

# Run your server with RunPod serverless
CMD ["python3", "server.py"]