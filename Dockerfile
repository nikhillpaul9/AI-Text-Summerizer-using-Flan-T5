# Use a lightweight Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system-level dependencies required for compiling ML libraries
RUN apt update -y && apt install awscli -y

# COPY THE ENTIRE PROJECT FIRST
# This ensures setup.py and all source files are present before pip runs
COPY . .

# Install all requirements (including the local -e . package)
RUN pip install -r requirements.txt
RUN pip install --upgrade accelerate
RUN pip uninstall -y transformers accelerate
RUN pip install transformers accelerate

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