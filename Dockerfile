# Use a lightweight Python base image optimized for ARM64 edge devices
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install OS-level dependencies required by Whisper (ffmpeg)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the Whisper base model so it never requires internet at runtime
RUN python -c "import whisper; whisper.load_model('base')"

# Copy the rest of the application code and the pre-computed ChromaDB
COPY . .

# Expose Gradio's port (Hugging Face Spaces default)
EXPOSE 7860

# Command to boot the application
CMD ["python", "app.py"]