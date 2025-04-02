import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Title and instructions
st.title("Physics Chat Model")
st.write("Chat with the fine-tuned Physics Chat model!")

# Use st.cache_resource to load the model only once
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    st.info("Loading model, please wait...")
    base_model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    model = PeftModel.from_pretrained(base_model, "jedlee2004/physics-chat")
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Function to generate a response
def generate_response(prompt, max_length=1024, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt")
    # Ensure inputs are on the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        temperature=temperature,
        do_sample=True,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input area
user_input = st.text_input("You:", key="user_input")

if st.button("Send") and user_input:
    with st.spinner("Generating response..."):
        response = generate_response(user_input)
        # Append user and model messages to the chat history
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Model", response))
        # Clear the input field
        st.session_state.user_input = ""

# Display the chat history
for speaker, message in st.session_state.chat_history:
    if speaker == "You":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Model:** {message}")
