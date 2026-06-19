# Use a lightweight Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# CRITICAL FIX: Added 'curl' for the healthcheck and 'build-essential' for ML C-compilers
RUN apt update -y && apt install -y awscli curl build-essential && rm -rf /var/lib/apt/lists/*

# COPY THE ENTIRE PROJECT FIRST
COPY . .

# Install requirements, then forcefully upgrade/reinstall transformers and accelerate in ONE layer
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --upgrade --force-reinstall transformers accelerate

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