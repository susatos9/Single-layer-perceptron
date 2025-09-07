# Deep Learning - Single Layer Perceptron Setup Instructions

## Setting up Virtual Environment

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone or download this project**
   ```bash
   # If using git
   git clone <repository-url>
   cd "Deep Learning"
   
   # Or extract the downloaded files to a folder
   ```

2. **Create a virtual environment**
   ```bash
   # Create virtual environment
   python -m venv slp_env
   
   # Activate virtual environment
   # On Linux/macOS:
   source slp_env/bin/activate
   
   # On Windows:
   slp_env\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import numpy, pandas, matplotlib, seaborn, sklearn; print('All packages installed successfully!')"
   ```

### Running the Code

#### Option 1: Run Python scripts directly
```bash
# Run the complete implementation
python iris_slp_complete.py

# Run the detailed implementation
python iris_slp_detailed.py

# Run the basic implementation
python iris_slp.py
```

#### Option 2: Run Jupyter Notebook
```bash
# Start Jupyter Notebook
jupyter notebook

# Then open iris_slp_notebook.ipynb in the browser
```

### Package Versions Used
- Python: 3.8+
- NumPy: 2.3.2
- Pandas: 2.3.2
- Matplotlib: Latest
- Seaborn: Latest
- Scikit-learn: Latest

### Deactivating Environment
When you're done working:
```bash
deactivate
```

### Troubleshooting

**If you get permission errors on Windows:**
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**If pip install fails:**
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Then try installing requirements again
pip install -r requirements.txt
```

**If you're using conda instead of pip:**
```bash
# Create conda environment
conda create -n slp_env python=3.9

# Activate environment
conda activate slp_env

# Install packages
conda install numpy pandas matplotlib seaborn scikit-learn jupyter
```
# Single-layer-perceptron
