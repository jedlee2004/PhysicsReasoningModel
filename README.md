export STREAMLIT_SERVER_RUN_ON_SAVE=false
streamlit run your_app.py --server.runOnSave=false


# Physics Chat Model Demo

This project demonstrates a fine-tuned Parameter-Efficient Fine-Tuned (PEFT) model for physics-related conversations. The application is built using Streamlit and allows users to interact with the model in a conversational manner.

---

## Prerequisites

Before running the application, ensure you have the following installed:
- Python 3.8 or higher
- pip (Python package manager)

---

## Installation

1. **Clone the Repository**:
```bash
git clone <repository-url>
cd <repository-folder>
```

2. Setup a virtual environment 
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```

3. Install dependencies via requirements.txt
```bash
pip install -r requirements.txt
```

## Running the Streamlit App
Run the Application: Use the following command to start the Streamlit app:

1. Run the application:
```bash
streamlit run streamlit.app.py
```

2. Access the app through local host: `http://localhost:8501`

3. Prompt the model

### Troubleshooting
CUDA Not Available: If the app falls back to CPU, ensure that:

Your GPU drivers are installed.
CUDA is properly configured for PyTorch.
Missing Dependencies: If you encounter missing dependencies, ensure you installed all packages from requirements.txt:
