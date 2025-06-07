FROM python:3.11.1-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY builder/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ .

# Set entrypoint
CMD ["python", "-u", "handler.py"]