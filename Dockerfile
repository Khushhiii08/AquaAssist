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
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code and the pre-computed ChromaDB
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Command to boot the application
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]