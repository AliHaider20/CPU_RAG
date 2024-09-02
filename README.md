# LocalRAG
Local RAG pipeline leveraging open-source SOTA LLM (Llama 3.1), Embedding (Mixed bread Embedding Large), and Reranker (BGE rernaker base) models.

# Requirements

Download Ollama and install from https://ollama.com/download

```
pip install -r requirements.txt
ollama pull mxbai-embed-large
ollama pull llama3.1
ollama serve # To start Ollama server
```

# Usage

```
python3 RAG.py
```

