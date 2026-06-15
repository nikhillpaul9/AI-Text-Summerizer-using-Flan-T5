# AI powered Text Summarization Tool using Flat-T5

An end-to-end, production-grade Natural Language Processing (NLP) pipeline designed to ingest, process, and summarize dense, long-form technical documentation and academic prose. 

This system features a fine-tuned **FLAN-T5-Small** sequence-to-sequence transformer model. It bypasses abstract high-level pipeline wrappers in favor of a native PyTorch inference engine to achieve maximum memory efficiency and deterministic decoding. The application is completely containerized and optimized for deployment via a high-performance, responsive Streamlit dashboard.

---

## 📂 Project Directory Structure

The project strictly follows an industrial, object-oriented MLOps design pattern. Configuration management is centralized, decoupling code execution from hyperparameter values.

```text
AI-Text-Summarizer-Project/
│
├── artifacts/                         # Generated dynamically during pipeline runs
│   ├── data_ingestion/                # Raw downloaded dataset splits
│   ├── data_validation/               # Schema validation status logs
│   ├── data_transformation/           # Tokenized tensor arrays
│   └── model_trainer/                 # Fine-tuned model weights and tokenizers
│
├── config/
│   └── config.yaml                    # Local disk pathways and file routing properties
│
├── research/
│   └── trials.ipynb                   # Experimental notebooks for parsing and training
│
├── src/
│   └── textSummarizer/
│       ├── __init__.py
│       ├── components/                # Core processing worker blocks
│       ├── config/                    # Configuration management layers
│       ├── entity/                    # Strongly typed dataclass definitions
│       ├── pipeline/                  # Execution orchestrators
│       │   ├── stage_01_data_ingestion.py
│       │   ├── stage_02_data_validation.py
│       │   ├── stage_03_data_transformation.py
│       │   ├── stage_04_model_trainer.py
│       │   ├── stage_05_model_evaluation.py
│       │   └── prediction.py          # High-performance native PyTorch inference worker
│       │
│       └── utils/                     # Global shared utility functions
│
├── .dockerignore                      # Limits Docker build context memory footprint
├── app.py                             # Interactive Streamlit Monolithic Dashboard UI
├── Dockerfile                         # Production containerization blueprint
├── main.py                            # Central MLOps pipeline orchestration file
├── params.yaml                        # Hyperparameters for model training & evaluation
├── README.md                          # System documentation
└── requirements.txt                   # Project dependencies

```

---

## 📊 Core Dataset Characteristics

The model is fine-tuned using a high-density academic scientific data pool derived from the **arXiv Summarization Dataset** (`ccdv/arxiv-summarization`).

* **Text Source:** Long-form, multi-page scientific publications covering artificial intelligence, machine learning, and deep neural networks.
* **Target Schema:** The target summaries consist of the author-verified abstracts.
* **Volume Footprint:** A curated, highly cleaned operational subset of **4,000 data points** optimized specifically for domain-specific fine-tuning on local compute.
* **Instruction Prefix Injection:** To prompt the instruction-tuned FLAN-T5 encoder appropriately, the tokenization wrapper systematically prepends the string token `"summarize: "` to every document entry.
* **Context Length Truncation:** Inputs are tokenized to a maximum sequence length of **1024 tokens**. Target abstracts are capped at **256 tokens** to prevent memory degradation while capturing necessary technical nuances.

---

## 🧠 Architectural Decisions & Troubleshooting

During development, the core architecture was heavily optimized to resolve critical production faults associated with standard library wrappers:

1. **Bypassing the Pipeline API (`Unknown task` Error):** Standard Hugging Face `pipeline()` calls rely on explicit task definitions string-mapped to local configuration keys. When loading fine-tuned local T5 checkpoints, registry drops can raise `Unknown task` exceptions.
* **Solution:** The inference layer (`prediction.py`) explicitly instantiates `AutoModelForSeq2SeqLM` and manages manual tokenization mappings and `.generate()` blocks directly.


2. **Eliminating Object Mutation Faults (`object has no attribute 'kwargs'` Error):** Attempting to dynamically adjust generation arguments on-the-fly by altering a pipeline's internal dictionary attributes triggers execution crashes.
* **Solution:** The implementation cleanly decouples presentation logic from tensor steps. Streamlit controls pass raw values into the prediction method as standard Python keyword arguments (`kwargs`), unpacking them safely into the PyTorch generation config.



---

## 🎨 Streamlit Application Architecture (`app.py`)

The presentation layer is designed as a standalone interface that communicates directly with the local PyTorch model instance cached in system memory.

* **Resource Optimization:** Utilizes `@st.cache_resource` to load the deep learning pipeline into RAM exactly once at startup, preventing latency bottlenecks on subsequent user interactions.
* **Session State Safeguards:** Text areas and calculated summary states are managed via `st.session_state` to prevent unnecessary UI model re-runs when hyperparameter sliders are toggled.
* **Dynamic Generation Sliders:** Exposes real-time parameters directly to the UI, including **Beam Search Width** (multi-path hypothesis exploration) and **Length Penalty** (exponential constraints balancing summary lengths).

---

## ⚙️ Optimized Hyperparameters (`params.yaml`)

The training parameters have been mathematically optimized for the 4,000-sample dataset, balancing per-device batch sizes and gradient accumulation to achieve an **Effective Batch Size of 16** while preventing local CPU thread-thrashing.

```yaml
TrainingArguments:
  # Memory Management
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  gradient_checkpointing: False
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
  load_best_model_at_end: True
  
  # CPU Specifics
  use_cpu: True
  fp16: False                  
  bf16: False
  max_grad_norm: 1.0      

```

---

## 🐳 Docker Container Blueprint

The compilation profile is explicitly configured to minimize spatial footprints while setting strict, headless parameters for Streamlit runtime tasks.

### `Dockerfile`

```dockerfile
# Use a lightweight Python base image
FROM python:3.11-slim

WORKDIR /app

# Install system-level dependencies required for compiling ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Leverage Docker layer caching for dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project (including artifacts/model_trainer weights)
COPY . .

EXPOSE 8501

# Enforce headless environment rules for Streamlit
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py"]

```

### `.dockerignore`

Ensures massive data checkpoints do not bloat container builds or increase registry upload latencies.

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

## 🛠️ Local Setup & Execution

**1. Environment Initialization**

```bash
git clone [https://github.com/yourusername/AI-Text-Summarizer-Project.git](https://github.com/yourusername/AI-Text-Summarizer-Project.git)
cd AI-Text-Summarizer-Project
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

```

**2. Execute the MLOps Pipeline**
Run the master coordinator to extract data, process tensors, and train the model.

```bash
python main.py

```

**3. Launch the Application**

```bash
streamlit run app.py

```

---
