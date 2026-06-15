import time
import streamlit as st
from src.textSummarizer.pipeline.prediction import PredictionPipeline

# Page Configurations for an enterprise aesthetic
st.set_page_config(
    page_title="Text Summarization Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# 1. RESOURCE OPTIMIZATION (Caching the Deep Learning Model)
# ----------------------------------------------------
@st.cache_resource(show_spinner="Initializing Model & Tokenizer Configuration...")
def load_prediction_pipeline():
    """
    Instantiates the prediction pipeline once and keeps it cached globally in memory.
    Prevents expensive disk-to-RAM reading cycles on every single UI rerun.
    """
    return PredictionPipeline()

# Initialize the pipeline cleanly
try:
    pipeline_instance = load_prediction_pipeline()
except Exception as e:
    st.error(f"Critical Failure loading the pipeline framework: {str(e)}")
    st.stop()

# ----------------------------------------------------
# 2. SIDEBAR - CONTROL PANEL & HYPERPARAMETER TUNING
# ----------------------------------------------------
st.sidebar.title("🎛️ Control Center")
st.sidebar.markdown("Fine-tune generation parameters dynamically without modifying core code.")

st.sidebar.subheader("Generation Strategy")
num_beams = st.sidebar.slider("Beam Search Width (num_beams)", min_value=1, max_value=8, value=4, step=1)
length_penalty = st.sidebar.slider("Length Penalty", min_value=0.5, max_value=2.5, value=1.2, step=0.1)

with st.sidebar.expander("Advanced Token Boundaries"):
    min_length = st.slider("Min Length Tokens", min_value=10, max_value=100, value=30, step=5)
    max_length = st.slider("Max Length Tokens", min_value=64, max_value=512, value=256, step=16)
    no_repeat_ngram_size = st.slider("No Repeat N-Gram", min_value=1, max_value=5, value=3, step=1)

st.sidebar.markdown("---")
st.sidebar.info("**Pipeline Mode:** Sequence-to-Sequence (FLAN-T5)\n\n**Environment:** Execution optimized via custom `ConfigurationManager` layouts.")

# ----------------------------------------------------
# 3. MAIN DASHBOARD - MODULAR TAB INTERFACE
# ----------------------------------------------------
st.title("🤖 Text Summarization Studio")
st.markdown("An advanced, enterprise-grade interface for processing dense academic and technical literature.")

# Initialize tabs for modular functional grouping
tab_direct, tab_file = st.tabs(["📝 Direct Text Processing", "📂 Batch File Processing"])

# --- TAB 1: DIRECT TEXT SUMMARIZATION ---
with tab_direct:
    st.subheader("Input Passage Execution")
    
    # Text area with word metrics
    input_text = st.text_area(
        "Paste technical abstract, paper segment, or long-form documentation here:",
        height=300,
        placeholder="Enter text to analyze..."
    )
    
    # Live stats tracking
    word_count = len(input_text.split()) if input_text.strip() else 0
    st.caption(f"Metrics Counter: {word_count} words | {len(input_text)} characters")

    # Action layout container
    col_trigger, col_metrics, _ = st.columns([2, 3, 5])
    
    with col_trigger:
        submit_btn = st.button("Generate Summary", type="primary", use_container_width=True)
        
    if submit_btn:
        if not input_text.strip():
            st.warning("Validation Failed: Prompt area cannot be empty.")
        else:
            with st.spinner("Running deep sequence transduction..."):
                start_time = time.time()
                
                # Bundle the UI configuration settings safely
                custom_kwargs = {
                    "max_length": max_length,
                    "min_length": min_length,
                    "length_penalty": length_penalty,
                    "num_beams": num_beams,
                    "no_repeat_ngram_size": no_repeat_ngram_size,
                    "early_stopping": True
                }
                
                try:
                    # RECTIFIED: Pass parameters safely as arguments into predict()
                    summary_output = pipeline_instance.predict(input_text, **custom_kwargs)
                    latency = time.time() - start_time
                    
                    st.session_state["latest_summary"] = summary_output
                    st.session_state["latest_latency"] = latency
                    st.session_state["summary_word_count"] = len(summary_output.split())
                    
                except Exception as eval_err:
                    st.error(f"Inference Loop Execution Exception: {str(eval_err)}")
    # Summary Display Panel
    if "latest_summary" in st.session_state:
        st.markdown("---")
        with col_metrics:
            st.metric("Inference Latency", f"{st.session_state['latest_latency']:.2f}s")
            st.metric("Compression Efficiency", f"{((word_count - st.session_state['summary_word_count']) / max(1, word_count)) * 100:.1f}% Reduction")
        
        st.subheader("Output Abstractive Summary")
        st.info(st.session_state["latest_summary"])

# --- TAB 2: FILE BATCH SUMMARIZATION ---
with tab_file:
    st.subheader("Document Extraction Utilities")
    uploaded_file = st.file_uploader("Upload raw text files for processing:", type=["txt"])
    
    if uploaded_file is not None:
        try:
            raw_bytes = uploaded_file.read()
            decoded_text = raw_bytes.decode("ust-8")
            
            st.success(f"Successfully staged file: {uploaded_file.name}")
            
            with st.expander("Preview Extracted Text Buffer"):
                st.text(decoded_text[:1500] + "\n\n[... truncated preview ...]")
                
            file_submit = st.button("Execute File Summary", type="secondary")
            if file_submit:
                with st.spinner("Analyzing document structure..."):
                    file_summary = pipeline_instance.predict(decoded_text)
                    st.markdown("### Document Summary Output")
                    st.success(file_summary)
        except Exception as file_err:
            st.error(f"File handling failure: {str(file_err)}")