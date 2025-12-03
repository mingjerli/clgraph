"""
Natural Language Queries with Vector Database and RAG

This example demonstrates how to enable natural language queries over
column lineage graphs and metadata using:
- Vector databases (ChromaDB) for semantic search
- RAG (Retrieval-Augmented Generation) for intelligent question answering
- Embeddings for capturing semantic relationships

Use cases:
1. "Which columns contain PII data?"
2. "Show me all transformations applied to revenue"
3. "What are the source columns for user_email?"
4. "Which tables are downstream of orders?"
5. "Find columns related to customer identification"
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("ERROR: chromadb not installed. Install with: pip install chromadb")
    raise

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers not installed. Install with: pip install sentence-transformers")
    raise

from clgraph.pipeline import Pipeline
from clgraph.models import ColumnNode, ColumnEdge


@dataclass
class ColumnDocument:
    """Represents a column as a document for embedding"""
    id: str
    full_name: str
    column_name: str
    table_name: str
    description: str
    node_type: str
    expression: Optional[str]
    operation: Optional[str]
    pii: bool
    owner: Optional[str]
    tags: List[str]
    upstream_columns: List[str]
    downstream_columns: List[str]
    transformation_path: str

    def to_text(self) -> str:
        """Convert column to rich text representation for embedding"""
        parts = [
            f"Column: {self.full_name}",
            f"Table: {self.table_name}",
            f"Name: {self.column_name}",
        ]

        if self.description:
            parts.append(f"Description: {self.description}")

        parts.append(f"Type: {self.node_type}")

        if self.expression:
            parts.append(f"Expression: {self.expression}")

        if self.operation:
            parts.append(f"Operation: {self.operation}")

        if self.pii:
            parts.append("Contains PII: Yes")

        if self.owner:
            parts.append(f"Owner: {self.owner}")

        if self.tags:
            parts.append(f"Tags: {', '.join(self.tags)}")

        if self.upstream_columns:
            parts.append(f"Derived from: {', '.join(self.upstream_columns[:5])}")

        if self.downstream_columns:
            parts.append(f"Used in: {', '.join(self.downstream_columns[:5])}")

        if self.transformation_path:
            parts.append(f"Transformation: {self.transformation_path}")

        return "\n".join(parts)

    def to_metadata(self) -> Dict[str, Any]:
        """Convert to metadata dict for filtering"""
        return {
            "full_name": self.full_name,
            "column_name": self.column_name,
            "table_name": self.table_name,
            "node_type": self.node_type,
            "pii": str(self.pii),  # ChromaDB requires string for boolean
            "owner": self.owner or "",
            "tags": json.dumps(list(self.tags)),  # Store as JSON string
            "has_description": str(bool(self.description)),
            "has_upstream": str(bool(self.upstream_columns)),
            "has_downstream": str(bool(self.downstream_columns)),
        }


class LineageVectorStore:
    """Vector store for column lineage data with semantic search"""

    def __init__(
        self,
        collection_name: str = "column_lineage",
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_directory: Optional[str] = None
    ):
        """
        Initialize vector store

        Args:
            collection_name: Name for the ChromaDB collection
            embedding_model: SentenceTransformer model name
            persist_directory: Directory to persist the database (optional)
        """
        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize ChromaDB client
        if persist_directory:
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
        else:
            self.client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False)
            )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Column lineage and metadata"}
        )

    def extract_column_documents(self, pipeline: Pipeline) -> List[ColumnDocument]:
        """Extract column documents from a pipeline"""
        documents = []

        for full_name, column in pipeline.columns.items():
            # Get upstream columns
            upstream = [
                edge.from_node.full_name
                for edge in pipeline.column_graph.get_edges_to(column)
            ]

            # Get downstream columns
            downstream = [
                edge.to_node.full_name
                for edge in pipeline.column_graph.get_edges_from(column)
            ]

            # Build transformation path description
            transformation_parts = []
            for edge in pipeline.column_graph.get_edges_to(column):
                if edge.transformation:
                    transformation_parts.append(edge.transformation)
                elif edge.edge_type != "direct":
                    transformation_parts.append(edge.edge_type)

            transformation_path = " -> ".join(transformation_parts) if transformation_parts else ""

            doc = ColumnDocument(
                id=full_name,
                full_name=full_name,
                column_name=column.column_name,
                table_name=column.table_name,
                description=column.description or "",
                node_type=column.node_type,
                expression=column.expression,
                operation=column.operation,
                pii=column.pii,
                owner=column.owner,
                tags=list(column.tags) if column.tags else [],
                upstream_columns=upstream,
                downstream_columns=downstream,
                transformation_path=transformation_path
            )
            documents.append(doc)

        return documents

    def add_pipeline(self, pipeline: Pipeline):
        """Add all columns from a pipeline to the vector store"""
        documents = self.extract_column_documents(pipeline)

        if not documents:
            print("No documents to add")
            return

        # Prepare data for ChromaDB
        ids = [doc.id for doc in documents]
        texts = [doc.to_text() for doc in documents]
        metadatas = [doc.to_metadata() for doc in documents]

        # Generate embeddings
        print(f"Generating embeddings for {len(documents)} columns...")
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        ).tolist()

        # Add to collection
        print("Adding to vector store...")
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print(f"✓ Added {len(documents)} columns to vector store")

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic search over columns

        Args:
            query: Natural language query
            n_results: Number of results to return
            where: Metadata filters (e.g., {"pii": "True"})

        Returns:
            List of matching column documents with scores
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True
        ).tolist()

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )

        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })

        return formatted_results

    def clear(self):
        """Clear all data from the collection"""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            metadata={"description": "Column lineage and metadata"}
        )


class LineageRAG:
    """RAG system for answering questions about column lineage"""

    def __init__(
        self,
        vector_store: LineageVectorStore,
        llm: Optional[Any] = None,
        retrieval_k: int = 5
    ):
        """
        Initialize RAG system

        Args:
            vector_store: LineageVectorStore instance
            llm: Language model for generation (LangChain LLM or similar)
            retrieval_k: Number of documents to retrieve
        """
        self.vector_store = vector_store
        self.llm = llm
        self.retrieval_k = retrieval_k

    def query(
        self,
        question: str,
        where: Optional[Dict[str, Any]] = None,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG

        Args:
            question: Natural language question
            where: Metadata filters for retrieval
            return_sources: Whether to return source documents

        Returns:
            Dictionary with answer and optional sources
        """
        # Retrieve relevant documents
        results = self.vector_store.search(
            query=question,
            n_results=self.retrieval_k,
            where=where
        )

        if not results:
            return {
                "answer": "No relevant columns found.",
                "sources": []
            }

        # If no LLM provided, return search results directly
        if not self.llm:
            return {
                "answer": self._format_search_results(results),
                "sources": results if return_sources else None
            }

        # Build context from retrieved documents
        context = self._build_context(results)

        # Generate answer using LLM
        prompt = self._build_prompt(question, context)

        try:
            # Try LangChain-style invocation
            if hasattr(self.llm, 'invoke'):
                response = self.llm.invoke(prompt)
                if hasattr(response, 'content'):
                    answer = response.content
                else:
                    answer = str(response)
            # Try direct call
            elif callable(self.llm):
                answer = self.llm(prompt)
            else:
                answer = str(self.llm)
        except Exception as e:
            print(f"Error calling LLM: {e}")
            answer = self._format_search_results(results)

        return {
            "answer": answer,
            "sources": results if return_sources else None
        }

    def _build_context(self, results: List[Dict[str, Any]]) -> str:
        """Build context string from search results"""
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"[{i}] {result['document']}")
        return "\n\n".join(context_parts)

    def _build_prompt(self, question: str, context: str) -> str:
        """Build prompt for LLM"""
        return f"""You are a data lineage expert. Answer the question based on the provided column information.

Context (Column Metadata):
{context}

Question: {question}

Instructions:
- Answer based ONLY on the provided context
- Be specific and reference column names
- If the context doesn't contain enough information, say so
- Format your answer clearly

Answer:"""

    def _format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results as a readable answer"""
        parts = [f"Found {len(results)} relevant columns:\n"]

        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            parts.append(f"{i}. {metadata['full_name']}")

            if metadata.get('pii') == 'True':
                parts.append("   - Contains PII")

            if metadata.get('owner'):
                parts.append(f"   - Owner: {metadata['owner']}")

            if metadata.get('node_type'):
                parts.append(f"   - Type: {metadata['node_type']}")

        return "\n".join(parts)


def main():
    """Demo: Natural language queries over column lineage"""

    print("=" * 80)
    print("Natural Language Queries with Vector Database and RAG")
    print("=" * 80)
    print()

    # Create example pipeline
    print("Step 1: Creating example pipeline...")
    queries = [
        ("raw_events", """
            CREATE TABLE raw_events AS
            SELECT
                event_id,              -- Unique event identifier [pii: false]
                user_id,               -- User identifier [pii: false, owner: data-team]
                user_email,            -- User email address [pii: true, owner: privacy-team, tags: contact sensitive]
                event_type,            -- Type of event [tags: analytics]
                event_timestamp,       -- Event timestamp [tags: temporal]
                session_id,            -- Session identifier
                revenue_amount         -- Revenue in USD [tags: financial revenue, owner: finance-team]
            FROM source_events
        """),

        ("daily_metrics", """
            CREATE TABLE daily_metrics AS
            WITH daily_aggregates AS (
                SELECT
                    DATE(event_timestamp) as event_date,
                    user_id,
                    user_email,
                    COUNT(*) as event_count,
                    SUM(revenue_amount) as total_revenue,
                    COUNT(DISTINCT session_id) as session_count
                FROM raw_events
                WHERE event_type = 'purchase'
                GROUP BY 1, 2, 3
            )
            SELECT
                event_date,
                user_id,
                user_email,
                event_count,
                total_revenue,
                session_count,
                total_revenue / event_count as avg_revenue_per_event  -- [tags: calculated kpi]
            FROM daily_aggregates
            WHERE event_count > 0
        """),

        ("user_summary", """
            CREATE TABLE user_summary AS
            SELECT
                user_id,
                user_email,
                SUM(total_revenue) as lifetime_revenue,        -- [tags: financial ltv]
                SUM(event_count) as total_events,
                COUNT(DISTINCT event_date) as active_days,
                MAX(event_date) as last_active_date,
                AVG(avg_revenue_per_event) as avg_order_value  -- [tags: kpi business-metric]
            FROM daily_metrics
            GROUP BY user_id, user_email
        """)
    ]

    pipeline = Pipeline(queries, dialect="bigquery")
    pipeline.propagate_all_metadata()

    print(f"✓ Created pipeline with {len(pipeline.columns)} columns")
    print()

    # Initialize vector store
    print("Step 2: Initializing vector store...")
    vector_store = LineageVectorStore(
        collection_name="demo_lineage",
        embedding_model="all-MiniLM-L6-v2"  # Small, fast model
    )
    vector_store.clear()  # Clear any existing data
    print("✓ Vector store ready")
    print()

    # Add pipeline to vector store
    print("Step 3: Adding pipeline to vector store...")
    vector_store.add_pipeline(pipeline)
    print()

    # Initialize RAG system (without LLM for basic demo)
    print("Step 4: Initializing RAG system...")
    rag = LineageRAG(vector_store, llm=None)
    print("✓ RAG system ready (semantic search mode)")
    print()

    # Example queries
    print("=" * 80)
    print("Example Natural Language Queries")
    print("=" * 80)
    print()

    queries_to_test = [
        ("Which columns contain PII data?", {"pii": "True"}),
        ("Show me all revenue-related columns", None),
        ("What columns are owned by the finance team?", None),
        ("Find columns related to user identification", None),
        ("Which columns have KPI tags?", None),
    ]

    for i, (query, where_filter) in enumerate(queries_to_test, 1):
        print(f"Query {i}: {query}")
        if where_filter:
            print(f"Filter: {where_filter}")
        print("-" * 80)

        result = rag.query(query, where=where_filter, return_sources=True)
        print(result['answer'])
        print()

    # Demonstrate lineage tracing integration
    print("=" * 80)
    print("Bonus: Combine NLP Search with Lineage Tracing")
    print("=" * 80)
    print()

    # Find revenue columns using NLP
    revenue_results = vector_store.search("revenue columns", n_results=3)

    for result in revenue_results:
        column_name = result['metadata']['full_name']
        print(f"\n📊 Column: {column_name}")

        # Trace backward to find sources
        table_name, col_name = column_name.rsplit('.', 1)
        sources = pipeline.trace_column_backward(table_name, col_name)

        if sources:
            print(f"   Sources ({len(sources)}):")
            for source in sources[:5]:  # Limit to 5
                print(f"     - {source.full_name}")

        # Trace forward to find impacts
        impacts = pipeline.trace_column_forward(table_name, col_name)
        if impacts:
            print(f"   Impacts ({len(impacts)}):")
            for impact in impacts[:5]:
                print(f"     - {impact.full_name}")

    print()
    print("=" * 80)
    print("✓ Demo complete!")
    print()
    print("To use with LLM:")
    print("  from langchain_ollama import ChatOllama")
    print("  llm = ChatOllama(model='llama2')")
    print("  rag = LineageRAG(vector_store, llm=llm)")
    print("  result = rag.query('Explain the revenue calculation')")
    print()


if __name__ == "__main__":
    main()
