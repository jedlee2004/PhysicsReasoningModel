# Create a virtual environment with Python 3.9
python -m venv venv --prompt "Portfolio-Milestone"

# Activate the virtual environment
.\venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install the required packages
pip install -r requirements.txt

# Check Python version
python --version

# List installed packages
pip list