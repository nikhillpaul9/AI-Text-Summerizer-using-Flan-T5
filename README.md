# AI-Powered Text Summarization Tool using FLAN-T5

An end-to-end, production-grade Natural Language Processing (NLP) pipeline designed to ingest, process, and summarize long-form technical documents, research papers, and academic content.

The project leverages a fine-tuned **FLAN-T5-Small** sequence-to-sequence transformer model and implements a custom PyTorch inference pipeline for efficient memory utilization, deterministic generation, and scalable deployment through a responsive Streamlit interface.

---

## 🚀 Features

- Fine-tuned FLAN-T5 model for domain-specific summarization
- End-to-end MLOps pipeline architecture
- Custom PyTorch inference engine
- Streamlit-based interactive web application
- Dockerized deployment
- AWS-ready infrastructure
- CI/CD automation using GitHub Actions
- Git LFS support for large model artifacts
- Configuration-driven architecture
- Optimized for CPU environments

---

## 📂 Project Structure

```text
AI-Text-Summarizer-Project/
│
├── artifacts/
│   ├── data_ingestion/
│   ├── data_validation/
│   ├── data_transformation/
│   └── model_trainer/
│
├── config/
│   └── config.yaml
│
├── research/
│   └── trials.ipynb
│
├── src/
│   └── textSummarizer/
│       ├── components/
│       ├── config/
│       ├── entity/
│       ├── pipeline/
│       │   ├── stage_01_data_ingestion.py
│       │   ├── stage_02_data_validation.py
│       │   ├── stage_03_data_transformation.py
│       │   ├── stage_04_model_trainer.py
│       │   ├── stage_05_model_evaluation.py
│       │   └── prediction.py
│       │
│       └── utils/
│
├── app.py
├── main.py
├── Dockerfile
├── params.yaml
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Information

The model is trained using a curated subset of the **ArXiv Summarization Dataset**.

### Dataset Details

| Attribute | Value |
|------------|---------|
| Dataset | `ccdv/arxiv-summarization` |
| Domain | Scientific Research Papers |
| Samples Used | ~4,000 |
| Input Type | Full Research Papers |
| Target Type | Author-written Abstracts |
| Max Input Length | 1024 Tokens |
| Max Target Length | 256 Tokens |

### Prompt Engineering

Each input document is prepended with:

```text
summarize:
```

This aligns with the instruction-tuned nature of FLAN-T5 and improves generation quality.

---

## 🧠 Model Architecture

### Base Model

- FLAN-T5-Small
- Encoder-Decoder Transformer Architecture
- Instruction-Tuned Language Model

### Why Custom Inference?

Instead of relying on Hugging Face's high-level `pipeline()` abstraction, inference is performed directly through:

```python
AutoModelForSeq2SeqLM
AutoTokenizer
model.generate()
```

### Benefits

- Lower memory consumption
- Better control over generation parameters
- Faster inference
- Deterministic outputs
- Elimination of common pipeline-related runtime errors

---

## ⚙️ Training Configuration

### Optimized Hyperparameters

```yaml
TrainingArguments:

  # Memory Management
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  gradient_checkpointing: false
  save_total_limit: 2

  # Training Dynamics
  num_train_epochs: 4
  learning_rate: 0.0003
  lr_scheduler_type: linear
  weight_decay: 0.01
  warmup_steps: 40
  logging_steps: 10

  # Evaluation & Saving
  evaluation_strategy: steps
  eval_steps: 200
  save_strategy: steps
  save_steps: 200
  load_best_model_at_end: true

  # CPU Optimization
  use_cpu: true
  fp16: false
  bf16: false
  max_grad_norm: 1.0
```

### Effective Batch Size

```text
Batch Size = 4 × Gradient Accumulation = 4

Effective Batch Size = 16
```

---

## 🎨 Streamlit Application

The application provides an intuitive interface for generating summaries from long-form text.

### Key Features

#### Resource Caching

```python
@st.cache_resource
```

Loads the model only once into memory, reducing inference latency.

#### Session State Management

Uses:

```python
st.session_state
```

to prevent unnecessary reruns and preserve generated summaries.

#### Dynamic Controls

Users can adjust:

- Beam Search Width
- Maximum Summary Length
- Length Penalty
- Generation Parameters

in real time.

## 🐳 Docker Deployment

### Dockerfile Highlights

- Python 3.11 Slim Base Image
- CPU-optimized PyTorch Wheels
- Streamlit Headless Configuration
- Health Checks
- Production-ready Container

### Build Image

```bash
docker build -t text-summarizer .
```

### Run Container

```bash
docker run -p 8501:8501 text-summarizer
```

### Access Application

```text
http://localhost:8501
```

---

## 📦 .dockerignore

The following files are excluded to reduce image size:

```text
.git
venv/
__pycache__/
*.pyc

artifacts/data_ingestion/
artifacts/data_validation/
artifacts/data_transformation/

research/
*.ipynb
```

---

## 🔄 CI/CD Pipeline

The project uses GitHub Actions for automated build and deployment.

### Continuous Integration

- Code checkout
- Dependency validation
- Linting
- Unit testing

### Continuous Delivery

- Docker image build
- AWS authentication
- Push image to Amazon ECR

### Continuous Deployment

- Self-hosted GitHub runner on EC2
- Pull latest image
- Remove old container
- Deploy updated version

### Git LFS Support

Large model files (~300MB) are managed through Git LFS:

```bash
git lfs track "*.safetensors"
```

---

## ☁️ AWS Deployment Architecture

### Infrastructure Components

| Service | Purpose |
|-----------|------------|
| Amazon EC2 | Application Hosting |
| Amazon ECR | Container Registry |
| GitHub Actions | CI/CD |
| Git LFS | Large Model Storage |

### Recommended EC2 Instance

```text
t3.medium or larger
```

### Storage

```text
16GB - 30GB EBS Volume
```

### Security Group

Allow inbound access:

```text
Port: 8501
Protocol: TCP
Source: 0.0.0.0/0
```

---

## 🏃 Running Locally

### Clone Repository

```bash
git clone https://github.com/yourusername/AI-Text-Summarizer-Project.git

cd AI-Text-Summarizer-Project
```

### Create Virtual Environment

#### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

#### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔬 Train the Model

Run the complete MLOps pipeline:

```bash
python main.py
```

This performs:

1. Data Ingestion
2. Data Validation
3. Data Transformation
4. Model Training
5. Model Evaluation

---

## ▶️ Launch the Application

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## 🛠️ Tech Stack

### Machine Learning

- PyTorch
- Transformers
- FLAN-T5
- Hugging Face Datasets

### Backend

- Python
- YAML Configuration

### Frontend

- Streamlit

### DevOps

- Docker
- GitHub Actions
- AWS EC2
- Amazon ECR
- Git LFS

### MLOps

- Modular Pipeline Design
- Configuration Management
- Artifact Tracking
- Automated Deployment

---

## 👨‍💻 Author

**Nikhil Paul**

---
⭐ If you found this project useful, consider giving it a star on GitHub.
