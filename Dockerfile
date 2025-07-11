# Use a slim Python 3.9 base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install dependencies for RDKit and Streamlit
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

# Set environment variables for Streamlit
ENV STREAMLIT_HOME="/app/.streamlit"
ENV STREAMLIT_BROWSER_GATHERUSAGESTATS=false
RUN mkdir -p /app/.streamlit

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch first to match torch-scatter compatibility
RUN pip install torch==2.1.2

# Install torch-scatter and torch-sparse using PyG wheels
RUN pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

# Install RDKit and remaining requirements
RUN pip install rdkit-pypi==2022.9.5

# Copy requirements and install all
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy models and dataset
COPY tox21.csv ./
COPY gcn_model.pt ./
COPY tox_model.pt ./
COPY gcn_best_threshold.npy ./

# Copy app source code
COPY src/ ./src/

# Expose Streamlit default port
EXPOSE 8501

# Define health check (optional)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the Streamlit app
ENTRYPOINT ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
