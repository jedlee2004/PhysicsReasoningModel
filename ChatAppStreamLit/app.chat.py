import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

@st.cache_resource
def load_model():
    model = AutoModelForCausalLM.from_pretrained("./physics_chatbot_model")
    tokenizer = AutoTokenizer.from_pretrained("./physics_chatbot_model")
    return model, tokenizer

model, tokenizer = load_model()

def generate_response(prompt, max_length=100, temperature=0.7):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,
        temperature=temperature,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Initialize session state for conversation history.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("Physics Chatbot")

# Display conversation history
for entry in st.session_state.chat_history:
    if entry["role"] == "user":
        st.markdown(f"**User:** {entry['text']}")
    else:
        st.markdown(f"**Chatbot:** {entry['text']}")

user_input = st.text_input("Enter your physics question:")

if st.button("Send") and user_input:
    # Append the user's input to the history.
    st.session_state.chat_history.append({"role": "user", "text": user_input})
    
    # For context, you might include previous exchanges.
    # Here, we simply use the latest question to generate a response.
    prompt = f"Q: {user_input}\nA:"
    response = generate_response(prompt)
    
    # Append the chatbot's response to the history.
    st.session_state.chat_history.append({"role": "chatbot", "text": response})
    
    # Rerun to update the conversation display.
    st.experimental_rerun()
