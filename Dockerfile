# Use a lightweight Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (awscli, curl, and compilation tools for ML libraries)
RUN apt-get update -y && apt-get install -y \
    awscli \
    curl \
    build-essential \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# COPY THE ENTIRE PROJECT FIRST to ensure setup.py is present
COPY . .

# 1. Upgrade core Python build tools to prevent compilation errors
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 2. Install requirements safely
RUN pip install --no-cache-dir -r requirements.txt

# 3. Cleanly upgrade the specific ML libraries
RUN pip install --no-cache-dir -U transformers accelerate

# Expose Streamlit's default port
EXPOSE 8501

# Streamlit-specific environment variables for headless environments
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true

# Add a healthcheck so AWS knows if the app crashes
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Command to run the application
ENTRYPOINT ["streamlit", "run", "app.py"]