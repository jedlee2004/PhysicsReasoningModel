
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import time
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Sliding Window Attention is enabled but not implemented")
warnings.filterwarnings("ignore", message="The attention mask and the pad token id were not set")

# Set page configuration
st.set_page_config(
    page_title="Physics Chat PEFT Model",
    page_icon="🧠",
    layout="wide"
)

# App title and description
st.title("Physics Chat PEFT Model Demo")
st.markdown("""
This application loads a Parameter-Efficient Fine-Tuned (PEFT) model for physics-related conversations.
Enter your physics question below to get a response from the model.
""")

# Cache the model loading to avoid reloading on each interaction
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    start_time = time.time()
    st.info("Loading model, please wait...")
    
    # Check for CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.write(f"Using device: {device}")
    
    # Load base model
    base_loading_start = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    base_loading_time = time.time() - base_loading_start
    st.write(f"Base model loaded in {base_loading_time:.2f} seconds")
    
    # Load PEFT adapter
    adapter_loading_start = time.time()
    model = PeftModel.from_pretrained(base_model, "jedlee2004/physics-chat")
    adapter_loading_time = time.time() - adapter_loading_start
    st.write(f"PEFT adapter loaded in {adapter_loading_time:.2f} seconds")
    
    # Load tokenizer
    tokenizer_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    
    # Ensure pad token is properly set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer_time = time.time() - tokenizer_start
    st.write(f"Tokenizer loaded in {tokenizer_time:.2f} seconds")
    
    total_time = time.time() - start_time
    st.success(f"Total loading time: {total_time:.2f} seconds")
    
    return model, tokenizer, device

# Sidebar for model configuration and system info
st.sidebar.header("Model Configuration")
st.sidebar.text("Base Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
st.sidebar.text("PEFT Model: jedlee2004/physics-chat")

# Generation parameters
st.sidebar.header("Generation Parameters")
max_new_tokens = st.sidebar.slider("Maximum Length", 64, 1024, 512)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
top_p = st.sidebar.slider("Top P", 0.0, 1.0, 0.9)

# Load model button
if st.sidebar.button("Load Model"):
    try:
        model, tokenizer, device = load_model_and_tokenizer()
        st.session_state.model_loaded = True
        st.session_state.model = model
        st.session_state.tokenizer = tokenizer 
        st.session_state.device = device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.session_state.model_loaded = False

# Check if model is loaded
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function for model inference
def generate_response(prompt):
    model = st.session_state.model
    tokenizer = st.session_state.tokenizer
    
    # Prepare the prompt with explicit attention mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the response
    response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response

# User input
if prompt := st.chat_input("Ask a physics question..."):
    if not st.session_state.model_loaded:
        st.error("Please load the model first using the button in the sidebar.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Generating response...")
            try:
                response = generate_response(prompt)
                message_placeholder.markdown(response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                message_placeholder.error(f"Error generating response: {str(e)}")

# Display model status
st.sidebar.subheader("Model Status")
if st.session_state.model_loaded:
    st.sidebar.success("Model loaded successfully! You can now ask questions.")
else:
    st.sidebar.warning("Model not loaded. Please load the model to start the conversation.")

# Add system information
st.sidebar.header("System Info")
if torch.cuda.is_available():
    st.sidebar.success(f"GPU available: {torch.cuda.get_device_name(0)}")
    st.sidebar.text(f"CUDA Version: {torch.version.cuda}")
    st.sidebar.text(f"Memory Allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
    st.sidebar.text(f"Memory Reserved: {torch.cuda.memory_reserved(0)/1e9:.2f} GB")
else:
    st.sidebar.warning("No GPU detected. Model will run on CPU which may be slow.")

# Add some helpful information
st.sidebar.header("About")
st.sidebar.info("""
This app demonstrates how to load and use a PEFT model with the 
DeepSeek-R1-Distill-Qwen-1.5B base model for physics-related questions.
The adapter is loaded from jedlee2004/physics-chat.
""")

# Disable file watcher warning
os.environ["STREAMLIT_WATCH_FILE_POLLING"] = "true"