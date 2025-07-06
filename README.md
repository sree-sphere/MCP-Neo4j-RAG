# RAG Graph Intelligence MCP

This project integrates a **RAG (Retrieval-Augmented Generation)** pipeline with both **vector-based** and **graph-based** retrieval using:

- **Milvus** for vector similarity search
- **Neo4j** for graph query execution
- **Groq (LLaMA3)** for Cypher query generation from natural language
- **LaBSE** and **CrossEncoder** for multilingual embedding and reranking

---

## Features

- Vector search over multilingual documents
- CrossEncoder reranking of search results
- Natural language to Cypher query generation
- Neo4j execution and visualization-ready outputs
- Designed as LangChain-compatible `FastMCP` toolchain

---

## Setup

### 1. Install Dependencies

Using `pip`:

```bash
pip install -r requirements.txt
```

Using `uv`:

```bash
uv pip install -r requirements.txt
```

---

### 2. Environment Variables

Create a `.env` file in the same directory as `main.py`:

```env
# Milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_TOKEN=your_milvus_token
COLLECTION_NAME=ycollection_name

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# Groq
GROQ_API_KEY=groq_api_key
GROQ_MODEL=llama3-70b-8192
```

---

## Files

```
.
├── main.py          # Main logic for ingestion, search, and graph querying
├── .env             # Environment variables for Milvus, Neo4j, Groq
├── requirements.txt # Python dependencies
└── README.md        # You're here
```

---

## Usage

Run the toolchain with:

```bash
python main.py
```

Under the hood, this runs:

```python
mcp.run(transport="stdio")
```

This makes the tools compatible with LangChain or other agentic systems using FastMCP protocol.

---

## Tools Defined in `main.py`

### Ingest Documents

```python
@mcp.tool()
async def ingest_documents(documents: List[Dict[str, Any]])
```

- Embeds and stores documents in Milvus
- Accepts `List[Dict[str, Any]]` with flexible schema

---

### Search Documents

```python
@mcp.tool()
async def search_documents(query: str, limit: int = 10, filter: Optional[Dict[str, Any]] = None)
```

- Embeds user query
- Performs vector search
- Applies CrossEncoder reranking

---

### Graph Search

```python
@mcp.tool()
async def graph_search(query: str, limit: int = 1)
```

- Embeds query
- Searches Milvus
- Uses Groq LLM to generate Cypher
- Executes Cypher on Neo4j
- Returns structured results

---

## Models Used

- **Embedding:** [LaBSE](https://huggingface.co/sentence-transformers/LaBSE)
- **Reranking:** [CrossEncoder (MiniLMv2)](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)
- **LLM:** Groq's `llama3-70b-8192`
