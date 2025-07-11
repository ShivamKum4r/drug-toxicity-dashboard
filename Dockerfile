FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install system dependencies required by RDKit and Streamlit
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libxrender1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libx11-6 \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Streamlit environment variables to avoid permission issues
ENV STREAMLIT_HOME="/app/.streamlit"
ENV STREAMLIT_BROWSER_GATHERUSAGESTATS=false
RUN mkdir -p /app/.streamlit

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch first to avoid build issues with torch-scatter
RUN pip install torch==2.1.2

# Install torch-scatter and torch-sparse with PyTorch compatibility
RUN pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

# Install RDKit and other requirements
RUN pip install rdkit-pypi==2022.9.5
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files and dataset
COPY tox21.csv ./
COPY gcn_model.pt ./
COPY tox_model.pt ./
COPY gcn_best_threshold.npy ./

# Copy your Streamlit app file (renamed here for clarity)
COPY app.py ./

# Expose port and define entrypoint
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
