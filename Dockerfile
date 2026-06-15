# Use a lightweight Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system-level dependencies required for compiling ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python dependencies
# We do this before copying the rest of the code to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
# NOTE: Ensure your fine-tuned model inside 'artifacts/model_trainer/' is copied!
COPY . .

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

RUN pip install -r requirements.txt
RUN pip install --upgrade accelerate
RUN pip uninstall -y transformers accelerate
RUN pip install transformers accelerate

# Command to run the application
ENTRYPOINT ["streamlit", "run", "app.py"]