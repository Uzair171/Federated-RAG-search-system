# Privacy RAG System

This is a Privacy-Enhanced Federated RAG System. It's a Flask-based web application and REST API server that uses local LLMs via Ollama to generate answers, ensuring no data leaves the machine.

## Prerequisites

- **Python 3.8+**
- **Ollama**: Must be installed and running locally. The system defaults to using `llama3.2` but will fall back to other models if it's not available.

## Installation

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone https://github.com/Uzair171/Federated-RAG-search-system.git
   cd privacy_rag_system
   ```

2. **Create a virtual environment (recommended)**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install the required libraries**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

To start the system, run the `app.py` file. On its first run, it will automatically build the ChromaDB index, set up the federated shards, and download the necessary embedding and reranking models.

```bash
python app.py
```

After initialization, open your browser and go to:
- **Main UI:** http://localhost:5000
- **Evaluation Dashboard:** http://localhost:5000/evaluation

**Note:** The first launch may take some time depending on your hardware, as it downloads transformer models and indexes the documents. These will be cached for subsequent runs!
