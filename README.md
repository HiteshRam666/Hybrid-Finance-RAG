# Hybrid Finance RAG

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Technology Stack](#technology-stack)
4. [Project Structure](#project-structure)
5. [Core Components](#core-components)
6. [Installation & Setup](#installation--setup)
7. [Configuration](#configuration)
8. [API Documentation](#api-documentation)
9. [Retrieval System](#retrieval-system)
---

## Project Overview

The **Hybrid Finance RAG** is a sophisticated Retrieval-Augmented Generation (RAG) system designed to answer finance-related questions using advanced hybrid retrieval techniques. The system combines semantic search, BM25 keyword search, and cross-encoder reranking to provide accurate, context-aware responses from a knowledge base of financial documents.

### Key Features

- **Hybrid Retrieval**: Combines semantic similarity search with BM25 keyword-based search
- **Advanced Reranking**: Uses cross-encoder models for improved relevance scoring
- **Vector Database Integration**: Leverages Pinecone for efficient vector storage and retrieval
- **Modern Web Interface**: Beautiful, responsive chat interface for user interaction
- **RESTful API**: FastAPI-based backend with comprehensive API documentation
- **Docker Support**: Containerized deployment for easy setup and scaling

### Use Cases

- Financial education and learning
- Investment research and analysis
- Personal finance guidance
- Banking and loan information
- Market analysis and insights

---

## Architecture

### System Architecture Diagram

```
┌─────────────────┐
│   Frontend      │
│  (HTML/JS)      │
└────────┬────────┘
         │ HTTP/REST
         ▼
┌─────────────────┐
│   FastAPI        │
│   Backend        │
└────────┬────────┘
         │
         ├──► Pinecone Vector Store
         │    (Document Embeddings)
         │
         ├──► OpenAI GPT-4o-mini
         │    (Answer Generation)
         │
         ├──► OpenAI Embeddings
         │    (text-embedding-3-large)
         │
         └──► Hybrid Retriever
              ├── Semantic Search
              ├── BM25 Search
              └── Cross-Encoder Reranking
```

### Data Flow

1. **Document Processing Pipeline**:
   - PDF documents are loaded from the `data/` directory
   - Documents are filtered and cleaned
   - Text is split into chunks (800 tokens with 100 token overlap)
   - Chunks are embedded using OpenAI embeddings
   - Embeddings are stored in Pinecone vector database

2. **Query Processing Pipeline**:
   - User query is received via API
   - Hybrid retrieval combines semantic and BM25 search
   - Retrieved documents are reranked using cross-encoder
   - Top-k documents are passed to LLM with context
   - LLM generates answer based on retrieved context

---

## Technology Stack

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **LangChain**: Framework for building LLM applications
- **OpenAI**: GPT-4o-mini for answer generation, text-embedding-3-large for embeddings
- **Pinecone**: Managed vector database for similarity search
- **Sentence Transformers**: For semantic search and reranking
- **BM25**: Keyword-based retrieval algorithm
- **PyPDF**: PDF document processing

### Frontend
- **HTML5/CSS3**: Modern web standards
- **Vanilla JavaScript**: No framework dependencies
- **Font Awesome**: Icon library

### Infrastructure
- **Docker**: Containerization
- **Python 3.10**: Runtime environment
- **Uvicorn**: ASGI server

---

## Project Structure

```
Bot/
├── backend/
│   ├── app.py                 # FastAPI application and routes
│   ├── store_index.py         # Vector store initialization and management
│   └── src/
│       ├── helper.py          # Document processing and hybrid retriever
│       └── prompt.py          # LLM prompt templates
├── frontend/
│   ├── index.html            # Main UI
│   └── app.js                # Frontend JavaScript logic
├── data/
│   └── Encyclopedia_finance.pdf  # Knowledge base document
├── .cache/                    # Cached processed documents
├── Dockerfile                 # Docker configuration
├── .dockerignore             # Docker ignore patterns
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
└── README.md                 # Basic project info
```

---

## Core Components

### 1. Document Processing (`backend/src/helper.py`)

#### `load_pdf_files(data)`
- **Purpose**: Loads PDF files from a directory
- **Input**: Directory path containing PDF files
- **Output**: List of LangChain Document objects
- **Technology**: PyPDFLoader, DirectoryLoader

#### `filter_docs(docs)`
- **Purpose**: Filters and cleans documents, preserving metadata
- **Features**:
  - Extracts source and page information
  - Cleans text content
  - Adds content statistics (length, word count)
- **Output**: Filtered Document objects with enhanced metadata

#### `text_split(filtered_docs, chunk_size=800, chunk_overlap=100)`
- **Purpose**: Splits documents into manageable chunks
- **Parameters**:
  - `chunk_size`: Maximum tokens per chunk (default: 800)
  - `chunk_overlap`: Overlapping tokens between chunks (default: 100)
- **Strategy**: Recursive character splitting with intelligent separators
- **Output**: List of text chunks with metadata

#### `clean_text(text)`
- **Purpose**: Normalizes and cleans text content
- **Operations**:
  - Removes excessive whitespace
  - Filters special characters
  - Preserves important punctuation

### 2. Hybrid Retrieval System (`backend/src/helper.py`)

#### `HybridRetrievar` Class

A sophisticated retrieval system combining multiple search strategies:

**Initialization**:
- Vector store (Pinecone)
- Embedding model (OpenAI)
- Sentence transformer for semantic search
- BM25 index for keyword search
- Cross-encoder for reranking

**Methods**:

1. **`build_bm25_index(documents)`**
   - Builds BM25 index from document corpus
   - Tokenizes documents for keyword matching

2. **`semantic_search(query, k=10)`**
   - Performs vector similarity search
   - Uses cosine similarity in embedding space
   - Returns top-k most semantically similar documents

3. **`bm25_search(query, k=10)`**
   - Keyword-based search using BM25 algorithm
   - Scores documents based on term frequency
   - Returns top-k documents with matching keywords

4. **`hybrid_search(query, k=10, alpha=0.7)`**
   - Combines semantic and BM25 search results
   - **Alpha parameter**: Weight for semantic search (0.7 = 70% semantic, 30% BM25)
   - Deduplicates results
   - Returns ranked documents by combined score

5. **`rerank_documents(query, documents, top_k=5)`**
   - Uses cross-encoder model for fine-grained relevance scoring
   - Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
   - Re-ranks documents based on query-document pairs
   - Returns top-k most relevant documents

6. **`retrieve(query, k=5, use_reranking=True)`**
   - Main retrieval method
   - Orchestrates hybrid search and optional reranking
   - Retrieves more documents initially if reranking is enabled
   - Returns final top-k documents

### 3. Vector Store Management (`backend/store_index.py`)

#### `initialize_store()`
- **Purpose**: Initializes Pinecone vector store and retriever
- **Features**:
  - Lazy initialization (only runs when needed)
  - Automatic index creation if it doesn't exist
  - Checks if index is empty before storing documents
  - Caches processed documents
- **Index Configuration**:
  - Name: `finance-bot`
  - Dimension: 3072 (for text-embedding-3-large)
  - Metric: Cosine similarity
  - Cloud: AWS (us-east-1)

#### `get_retriever()`
- **Purpose**: Returns initialized retriever (singleton pattern)
- **Returns**: `HybridRetrievar` instance

**Models**:

1. **`QueryRequest`**
   - `query`: User's question (required)
   - `use_reranking`: Enable/disable reranking (default: True)
   - `max_results`: Maximum documents to return (1-10, default: 5)

2. **`DocumentResponse`**
   - `content`: Document text snippet
   - `source`: Source file path
   - `page`: Page number
   - `relevance_score`: Relevance score (0-1)

3. **`QueryResponse`**
   - `status`: "success" or "error"
   - `answer`: Generated answer
   - `documents`: List of retrieved documents
   - `retrieval_time`: Time taken for retrieval
   - `error`: Error message (if any)

**Endpoints**:

1. **`GET /`**: Welcome endpoint with API information
2. **`GET /health`**: Health check endpoint
3. **`POST /query`**: Main query endpoint
4. **`GET /app`**: Serves frontend HTML
5. **`GET /app.js`**: Serves frontend JavaScript

### 5. Prompt Engineering (`backend/src/prompt.py`)

#### System Prompt
- **Role**: Expert Financial Assistant
- **Instructions**:
  - Base answers only on provided context
  - Never fabricate information
  - Cite specific details when relevant
  - Keep responses concise (2-4 sentences)
- **Tone**: Professional, clear, confident

---

## Installation & Setup

### Prerequisites

- Python 3.10 or higher
- Docker (optional, for containerized deployment)
- OpenAI API key
- Pinecone API key

### Local Development Setup

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd Bot
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment**:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   ```

6. **Run the application**:
   ```bash
   uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
   ```

7. **Access the application**:
   - Frontend: http://localhost:8000/app
   - API Docs: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

---

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for embeddings and chat | Yes |
| `PINECONE_API_KEY` | Pinecone API key for vector database | Yes |

### Model Configuration

**Embedding Model**:
- Model: `text-embedding-3-large`
- Dimensions: 3072
- Provider: OpenAI

**Chat Model**:
- Model: `gpt-4o-mini`
- Temperature: 0.1 (low for consistent answers)
- Max Tokens: 1000

**Reranking Model**:
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Purpose: Document reranking for improved relevance

### Retrieval Parameters

**Chunking**:
- Chunk Size: 800 tokens
- Chunk Overlap: 100 tokens

**Hybrid Search**:
- Alpha: 0.7 (70% semantic, 30% BM25)
- Initial Retrieval: k * 3 (when reranking enabled)

**Reranking**:
- Top-k after reranking: User-specified (default: 5)

---

## Retrieval System

### Hybrid Retrieval Architecture

The system uses a three-stage retrieval pipeline:

1. **Semantic Search**: Vector similarity search using OpenAI embeddings
2. **BM25 Search**: Keyword-based search using BM25 algorithm
3. **Reranking**: Cross-encoder model for fine-grained relevance scoring

### How It Works

1. **Initial Retrieval**:
   - Semantic search retrieves 2k documents
   - BM25 search retrieves 2k documents
   - Results are combined and deduplicated

2. **Score Combination**:
   - Semantic scores: Normalized by rank position
   - BM25 scores: Normalized by rank position
   - Combined score: `alpha * semantic_score + (1 - alpha) * bm25_score`
   - Default alpha: 0.7 (70% semantic, 30% keyword)

3. **Reranking** (Optional):
   - Top candidates are reranked using cross-encoder
   - Cross-encoder evaluates query-document pairs
   - Final top-k documents are selected

### Advantages

- **Semantic Search**: Captures meaning and context
- **BM25 Search**: Handles exact keyword matches
- **Reranking**: Improves precision of final results
- **Hybrid Approach**: Best of both worlds

---

## Future Enhancements

Potential improvements for the system:

1. **Multi-document Support**: Support for multiple PDF sources
2. **Conversation History**: Maintain context across queries
3. **User Authentication**: Add user management
4. **Analytics**: Track query patterns and performance
5. **Caching**: Cache frequent queries for faster responses
6. **Streaming Responses**: Real-time answer generation
7. **Multi-language Support**: Support for multiple languages
8. **Advanced Filtering**: Filter by document source, date, etc.

---

## Author

Hitesh Ram - hiteshram321@gmail.com

---


