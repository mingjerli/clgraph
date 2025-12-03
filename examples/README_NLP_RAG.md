# Natural Language Queries with Vector Database and RAG

This example demonstrates how to enable natural language queries over column lineage graphs and metadata using vector databases and Retrieval-Augmented Generation (RAG).

## Overview

The example combines three powerful technologies:

1. **Vector Databases (ChromaDB)** - For efficient semantic search over column metadata
2. **Embeddings (sentence-transformers)** - For capturing semantic relationships between text
3. **RAG (Retrieval-Augmented Generation)** - For intelligent question answering

## Architecture

```
Pipeline → Extract Metadata → Embed → Vector DB → Semantic Search → RAG → Answer
```

### Key Components

#### 1. **ColumnDocument**
Represents a column as a searchable document with:
- Column identity (name, table, full_name)
- Metadata (description, PII status, owner, tags)
- Lineage context (upstream/downstream columns)
- Transformation information (expressions, operations)

#### 2. **LineageVectorStore**
Vector database wrapper that:
- Extracts column documents from pipelines
- Generates embeddings using sentence-transformers
- Stores in ChromaDB with metadata filters
- Provides semantic search capabilities

#### 3. **LineageRAG**
RAG system that:
- Retrieves relevant columns using semantic search
- Builds context from retrieved documents
- Generates answers using LLM (optional)
- Falls back to formatted search results

## Installation

```bash
# Install base clgraph
pip install clgraph

# Install NLP dependencies
pip install clgraph[nlp]

# Or install manually
pip install chromadb sentence-transformers
```

## Quick Start

```python
from clgraph.pipeline import Pipeline
from examples.nlp_vector_database_rag import LineageVectorStore, LineageRAG

# Create your pipeline
queries = [
    ("table1", "CREATE TABLE table1 AS SELECT ..."),
    ("table2", "CREATE TABLE table2 AS SELECT ..."),
]
pipeline = Pipeline(queries, dialect="bigquery")
pipeline.propagate_all_metadata()

# Initialize vector store
vector_store = LineageVectorStore()
vector_store.add_pipeline(pipeline)

# Query using natural language
rag = LineageRAG(vector_store)
result = rag.query("Which columns contain PII data?")
print(result['answer'])
```

## Example Queries

### 1. Find PII Columns

```python
result = rag.query(
    "Which columns contain PII data?",
    where={"pii": "True"}
)
```

### 2. Find Revenue-Related Columns

```python
result = rag.query("Show me all revenue-related columns")
```

### 3. Find Columns by Owner

```python
result = rag.query("What columns are owned by the finance team?")
```

### 4. Semantic Search for User Data

```python
result = rag.query("Find columns related to user identification")
```

### 5. Find Metrics and KPIs

```python
result = rag.query("Which columns have KPI tags?")
```

## Advanced Usage

### With LLM Integration

```python
from langchain_ollama import ChatOllama

# Initialize LLM
llm = ChatOllama(model="llama2", temperature=0.3)

# Create RAG with LLM
rag = LineageRAG(vector_store, llm=llm)

# Get intelligent answers
result = rag.query("Explain how revenue metrics are calculated")
print(result['answer'])
```

### Metadata Filtering

ChromaDB supports powerful metadata filters:

```python
# Find PII columns
results = vector_store.search(
    query="email addresses",
    where={"pii": "True"}
)

# Find columns with specific owner
results = vector_store.search(
    query="financial data",
    where={"owner": "finance-team"}
)

# Find columns with descriptions
results = vector_store.search(
    query="user metrics",
    where={"has_description": "True"}
)
```

### Combine NLP Search with Lineage Tracing

```python
# Find revenue columns using NLP
revenue_results = vector_store.search("revenue columns", n_results=3)

# Trace lineage for each result
for result in revenue_results:
    column_name = result['metadata']['full_name']
    table_name, col_name = column_name.rsplit('.', 1)

    # Get source columns
    sources = pipeline.trace_column_backward(table_name, col_name)

    # Get downstream impacts
    impacts = pipeline.trace_column_forward(table_name, col_name)

    print(f"Column: {column_name}")
    print(f"Sources: {[s.full_name for s in sources]}")
    print(f"Impacts: {[i.full_name for i in impacts]}")
```

### Persistent Storage

```python
# Save vector database to disk
vector_store = LineageVectorStore(
    persist_directory="./lineage_vectors"
)

# Add data
vector_store.add_pipeline(pipeline)

# Data persists across sessions
# Next time:
vector_store = LineageVectorStore(
    persist_directory="./lineage_vectors"
)
# Existing data is loaded automatically
```

### Custom Embedding Models

```python
# Use different embedding model
vector_store = LineageVectorStore(
    embedding_model="all-mpnet-base-v2"  # Higher quality, slower
)

# Or use smaller/faster model
vector_store = LineageVectorStore(
    embedding_model="all-MiniLM-L6-v2"  # Default, good balance
)
```

## Use Cases

### 1. Data Discovery
Enable data analysts to find relevant columns using natural language:
- "Find all customer contact information"
- "Show me revenue metrics from the sales pipeline"
- "What columns track user behavior?"

### 2. Data Governance
Support compliance and governance queries:
- "Which columns contain PII?"
- "What data is owned by the privacy team?"
- "Find all columns with financial tags"

### 3. Impact Analysis
Understand data dependencies through natural language:
- "What downstream tables use customer email?"
- "Show me all metrics derived from raw events"
- "What columns are affected by the orders table?"

### 4. Documentation
Help teams understand their data:
- "Explain the user engagement metrics"
- "What is the lifetime revenue calculation?"
- "How is daily active users computed?"

### 5. Data Quality
Identify data quality issues:
- "Which columns lack descriptions?"
- "Find columns without owners"
- "Show me all undocumented transformations"

## How It Works

### 1. Document Extraction

Each column is converted to a rich text document:

```
Column: user_summary.lifetime_revenue
Table: user_summary
Name: lifetime_revenue
Description: Total revenue from all purchases
Type: output
Expression: SUM(total_revenue)
Operation: aggregate
Tags: financial, ltv
Derived from: daily_metrics.total_revenue
Transformation: aggregate
```

### 2. Embedding Generation

Text is converted to vector embeddings using sentence-transformers:
- Captures semantic meaning
- Similar concepts have similar embeddings
- Enables semantic search

### 3. Vector Storage

Embeddings stored in ChromaDB with:
- Full text for retrieval
- Metadata for filtering
- Efficient similarity search

### 4. Semantic Search

Query converted to embedding and compared to stored embeddings:
- Find most similar columns
- Can combine with metadata filters
- Returns ranked results

### 5. RAG (Optional)

Retrieved columns used as context for LLM:
- LLM generates natural language answer
- Grounded in actual column metadata
- More informative than raw search results

## Performance Tips

### 1. Choose Right Embedding Model

```python
# Fast but lower quality (good for testing)
embedding_model="all-MiniLM-L6-v2"  # 384 dimensions

# Balanced (recommended for production)
embedding_model="all-mpnet-base-v2"  # 768 dimensions

# High quality but slower
embedding_model="sentence-t5-xl"  # 768 dimensions
```

### 2. Use Metadata Filters

Reduce search space with filters:

```python
# Only search within specific table
results = vector_store.search(
    query="revenue",
    where={"table_name": "user_summary"}
)
```

### 3. Adjust Retrieval Size

```python
# Fewer results = faster
rag = LineageRAG(vector_store, retrieval_k=3)

# More results = better context but slower
rag = LineageRAG(vector_store, retrieval_k=10)
```

### 4. Batch Processing

```python
# Add multiple pipelines efficiently
for pipeline in pipelines:
    vector_store.add_pipeline(pipeline)
```

## Troubleshooting

### Import Errors

If you get import errors, install dependencies:

```bash
pip install chromadb sentence-transformers
```

### Model Download Issues

First run downloads embedding model from HuggingFace:
- Requires internet connection
- Models cached in `~/.cache/huggingface/`
- Can take a few minutes

### Memory Issues

For large pipelines:
- Use smaller embedding model
- Process in batches
- Increase system memory

### Performance Issues

If search is slow:
- Use smaller embedding model
- Reduce retrieval_k
- Add metadata filters
- Use persistent storage

## Integration Examples

### With Streamlit Dashboard

```python
import streamlit as st
from examples.nlp_vector_database_rag import LineageRAG, LineageVectorStore

st.title("Data Lineage Search")

# Initialize (cached)
@st.cache_resource
def get_rag():
    vector_store = LineageVectorStore()
    vector_store.add_pipeline(pipeline)
    return LineageRAG(vector_store)

rag = get_rag()

# Search interface
query = st.text_input("Ask about your data:")
if query:
    result = rag.query(query)
    st.write(result['answer'])

    with st.expander("Sources"):
        for source in result['sources']:
            st.json(source['metadata'])
```

### With FastAPI REST API

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Initialize once
vector_store = LineageVectorStore()
rag = LineageRAG(vector_store)

class Query(BaseModel):
    question: str
    filters: dict = None

@app.post("/search")
async def search(query: Query):
    result = rag.query(query.question, where=query.filters)
    return result
```

### With Jupyter Notebook

```python
# In notebook cell
%pip install clgraph[nlp]

from clgraph.pipeline import Pipeline
from examples.nlp_vector_database_rag import LineageVectorStore, LineageRAG

# Interactive exploration
vector_store = LineageVectorStore()
vector_store.add_pipeline(pipeline)

rag = LineageRAG(vector_store)

# Try different queries
queries = [
    "PII columns",
    "revenue metrics",
    "user identification"
]

for q in queries:
    print(f"\n{q}:")
    print(rag.query(q)['answer'])
```

## Comparison with Traditional Search

### Traditional Keyword Search
```python
# Must match exact terms
columns = [c for c in pipeline.columns.values()
           if "revenue" in c.column_name.lower()]
```

**Limitations:**
- Only finds exact keyword matches
- Misses synonyms (sales, income, earnings)
- No semantic understanding
- Can't handle natural language

### Semantic Search with Vector DB
```python
# Understands semantics
results = rag.query("Find financial income columns")
```

**Benefits:**
- Finds semantically similar columns
- Understands synonyms and related concepts
- Handles natural language queries
- Ranks by relevance

## Future Enhancements

Possible extensions to this example:

1. **Multi-modal Search** - Search code, documentation, and comments
2. **Query Suggestions** - Suggest related queries based on history
3. **Auto-tagging** - Use LLM to automatically tag columns
4. **Anomaly Detection** - Find unusual patterns in lineage
5. **Question Routing** - Route different query types to different handlers
6. **Feedback Loop** - Learn from user feedback to improve results

## References

- **ChromaDB**: https://www.trychroma.com/
- **sentence-transformers**: https://www.sbert.net/
- **LangChain**: https://python.langchain.com/
- **RAG Pattern**: https://python.langchain.com/docs/use_cases/question_answering/

## Contributing

Have ideas for improving this example? Contributions welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details
