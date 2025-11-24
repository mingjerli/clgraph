"""
E-Commerce Pipeline Metadata Management

This script demonstrates how to use clpipe's metadata capabilities:
1. Parse inline SQL comment metadata
2. Manually assign metadata (PII, owner, tags)
3. Propagate metadata through lineage
4. Generate descriptions (with LLM or fallback)
5. Query columns by metadata
6. Trace PII through lineage
7. Save/load metadata to persist across sessions
8. Export metadata to JSON
"""

import json
import tempfile
from pathlib import Path

from clpipe import Pipeline


def load_sql_queries(sql_dir: Path) -> list[tuple[str, str]]:
    """Load all SQL files from a directory in sorted order."""
    queries = []
    for sql_file in sorted(sql_dir.glob("*.sql")):
        with open(sql_file) as f:
            sql = f.read()
        query_name = sql_file.stem
        queries.append((query_name, sql))
        print(f"  Loaded: {query_name}")
    return queries


def main():
    print("=" * 80)
    print("E-Commerce Pipeline Metadata Management")
    print("=" * 80)
    print()

    # Load SQL files from current directory
    sql_dir = Path(__file__).parent
    queries = load_sql_queries(sql_dir)

    print(f"\nLoaded {len(queries)} SQL files")
    print()

    # Build the pipeline
    print("Building pipeline...")
    pipeline = Pipeline(queries, dialect="duckdb")
    print(f"  Built pipeline with {len(pipeline.table_graph.queries)} queries")
    print(f"  Found {len(pipeline.columns)} columns")
    print()

    # -------------------------------------------------------------------------
    # 1. Inline SQL Comment Metadata Parsing
    # -------------------------------------------------------------------------
    print("1. INLINE SQL COMMENT METADATA")
    print("-" * 80)
    print("""
  clpipe can parse structured metadata from SQL comments in the format:
    <description> [key: value, key2: value2, ...]

  Example SQL:
    SELECT
        email,           -- User email address [pii: true, owner: data-team]
        SUM(amount) as total  /* Total revenue [tags: metric finance] */
    FROM ...

  Supported metadata keys:
    - description: Free-text description (before brackets)
    - pii: Boolean flag (true/false)
    - owner: String identifying data owner
    - tags: Space-separated tags
    - Any custom key-value pairs
""")

    # Show columns that have inline metadata (from SQL comments)
    # Metadata is extracted and stored directly on column properties
    cols_with_inline_metadata = [
        col
        for col in pipeline.columns.values()
        if col.description_source and col.description_source.value == "source"
    ]

    # Deduplicate by (table_name, column_name)
    seen = set()
    unique_cols = []
    for col in cols_with_inline_metadata:
        key = (col.table_name, col.column_name)
        if key not in seen:
            seen.add(key)
            unique_cols.append(col)

    if unique_cols:
        print(f"  Found {len(unique_cols)} columns with inline metadata:")
        for col in unique_cols[:15]:
            print(f"    {col.table_name}.{col.column_name}:")
            if col.description:
                print(f"      description: {col.description}")
            if col.pii:
                print(f"      pii: {col.pii}")
            if col.owner:
                print(f"      owner: {col.owner}")
            if col.tags:
                print(f"      tags: {col.tags}")
        if len(unique_cols) > 15:
            print(f"    ... and {len(unique_cols) - 15} more")
    else:
        print("  No columns with inline metadata found in these SQL files.")
        print("  (Add comments like '-- Description [pii: true]' to columns)")
    print()

    # -------------------------------------------------------------------------
    # 2. Manual Metadata Assignment
    # -------------------------------------------------------------------------
    print("2. MANUAL METADATA ASSIGNMENT")
    print("-" * 80)
    print("""
  For columns without inline metadata, you can assign metadata programmatically:
    col.pii = True
    col.owner = "data-team"
    col.tags.add("metric")
    col.description = "Custom description"
    col.custom_metadata["sensitivity"] = "high"

  Note: Columns in raw_orders, raw_customers, raw_products already have metadata
  from inline SQL comments. Here we add metadata to order_items (which has none).
""")

    # Add metadata to raw_order_items (which doesn't have inline comments)
    order_item_metadata = [
        ("raw_order_items", "order_item_id", "data-platform", None, set()),
        ("raw_order_items", "order_id", "data-platform", None, set()),
        ("raw_order_items", "product_id", "data-platform", None, set()),
        ("raw_order_items", "quantity", "operations", "Number of units ordered", {"metric"}),
        ("raw_order_items", "unit_price", "finance", "Price per unit", {"metric", "revenue"}),
        (
            "raw_order_items",
            "line_total",
            "finance",
            "Total line item amount",
            {"metric", "revenue"},
        ),
    ]

    print("  Adding metadata to raw_order_items:")
    marked_count = 0
    for table, column, owner, description, tags in order_item_metadata:
        for col in pipeline.columns.values():
            if col.table_name == table and col.column_name == column and not col.owner:
                col.owner = owner
                if description:
                    col.description = description
                col.tags.update(tags)
                marked_count += 1
                tag_str = f", tags: {tags}" if tags else ""
                desc_str = f", desc: {description[:30]}..." if description else ""
                print(f"    {table}.{column} [owner: {owner}{tag_str}{desc_str}]")

    print(f"\n  Added metadata to {marked_count} columns programmatically")
    print()

    # -------------------------------------------------------------------------
    # 3. Metadata Propagation
    # -------------------------------------------------------------------------
    print("3. METADATA PROPAGATION")
    print("-" * 80)
    print("""
  Metadata propagates through lineage automatically:
    - PII: If any source column is PII, derived column is PII
    - Owner: First owner found in sources wins
    - Tags: Union of all source tags

  This ensures data governance follows the data through transformations.
""")

    # Check PII before propagation
    pii_before = len(pipeline.get_pii_columns())
    print(f"  PII columns before propagation: {pii_before}")

    # Propagate metadata
    pipeline.propagate_all_metadata()

    # Check PII after propagation
    pii_after = len(pipeline.get_pii_columns())
    print(f"  PII columns after propagation:  {pii_after}")
    print(f"  New PII columns discovered:     {pii_after - pii_before}")
    print()

    # -------------------------------------------------------------------------
    # 4. Description Generation
    # -------------------------------------------------------------------------
    print("4. DESCRIPTION GENERATION")
    print("-" * 80)
    print("""
  clpipe can generate descriptions for columns that don't have them.

  Supported LLM backends:
    - Ollama (local, free): ollama pull llama3.2
    - OpenAI: requires OPENAI_API_KEY

  Usage:
    pipeline.llm = llm_instance
    pipeline.generate_all_descriptions()
""")

    # Count columns without descriptions in derived tables
    cols_without_desc = [
        col
        for col in pipeline.columns.values()
        if not col.description and col.table_name.startswith(("int_", "mart_", "stg_"))
    ]

    # Deduplicate
    seen = set()
    unique_without_desc = []
    for col in cols_without_desc:
        key = (col.table_name, col.column_name)
        if key not in seen:
            seen.add(key)
            unique_without_desc.append(col)

    print(f"  Columns in derived tables without descriptions: {len(unique_without_desc)}")
    print()

    # Try to use Ollama for description generation
    llm_available = False
    ollama_models = ["llama3:latest", "llama3.2", "qwen3-coder:30b"]  # Try these in order

    try:
        from langchain_ollama import ChatOllama

        print("  Attempting to connect to Ollama...")
        for model_name in ollama_models:
            try:
                llm = ChatOllama(
                    model=model_name,
                    temperature=0.3,
                )
                # Test connection with a simple call
                llm.invoke("test")
                pipeline.llm = llm
                llm_available = True
                print(f"  Connected to Ollama (model: {model_name})")
                break
            except Exception:
                continue

        if not llm_available:
            raise Exception("No working Ollama model found")

    except Exception as e:
        print(f"  Ollama not available: {type(e).__name__}")
        print("  To enable LLM descriptions:")
        print("    1. Install Ollama: brew install ollama")
        print("    2. Pull model: ollama pull llama3:latest")
        print("    3. Start server: ollama serve")
    print()

    if llm_available:
        # Generate descriptions for a few sample columns to demonstrate
        print("  Generating descriptions for sample columns...")
        print("  (Generating for 5 columns to save time)")
        print()

        sample_cols = unique_without_desc[:5]
        for col in sample_cols:
            try:
                # Import the generate function
                from clpipe.column import generate_description

                generate_description(col, pipeline.llm, pipeline)
                print(f"    {col.table_name}.{col.column_name}:")
                print(f"      -> {col.description}")
            except Exception as e:
                print(f"    {col.table_name}.{col.column_name}: Error - {e}")
        print()
        print("  To generate all descriptions, run: pipeline.generate_all_descriptions()")
    else:
        print("  Sample columns that would get descriptions:")
        for col in unique_without_desc[:5]:
            print(f"    - {col.table_name}.{col.column_name}")
        if len(unique_without_desc) > 5:
            print(f"    ... and {len(unique_without_desc) - 5} more")
    print()

    # -------------------------------------------------------------------------
    # 5. Querying Columns by Metadata
    # -------------------------------------------------------------------------
    print("5. QUERYING COLUMNS BY METADATA")
    print("-" * 80)

    # 5a. Get PII columns
    print("  5a. PII Columns (get_pii_columns)")
    print("  " + "-" * 40)
    pii_columns = pipeline.get_pii_columns()
    print(f"  Found {len(pii_columns)} PII columns:")

    # Group by table for cleaner output
    pii_by_table = {}
    for col in pii_columns:
        if col.table_name not in pii_by_table:
            pii_by_table[col.table_name] = []
        pii_by_table[col.table_name].append(col.column_name)

    for table in sorted(pii_by_table.keys()):
        cols = sorted(set(pii_by_table[table]))  # dedupe
        print(f"    {table}:")
        for col_name in cols[:5]:
            print(f"      - {col_name}")
        if len(cols) > 5:
            print(f"      ... and {len(cols) - 5} more")
    print()

    # 5b. Get columns by owner
    print("  5b. Columns by Owner (get_columns_by_owner)")
    print("  " + "-" * 40)

    owners = set()
    for col in pipeline.columns.values():
        if col.owner:
            owners.add(col.owner)

    for owner in sorted(owners):
        cols = pipeline.get_columns_by_owner(owner)
        unique_cols = {(c.table_name, c.column_name) for c in cols}
        print(f"    {owner}: {len(unique_cols)} unique columns")
    print()

    # 5c. Get columns by tag
    print("  5c. Columns by Tag (get_columns_by_tag)")
    print("  " + "-" * 40)

    all_tags = set()
    for col in pipeline.columns.values():
        all_tags.update(col.tags)

    for tag in sorted(all_tags):
        cols = pipeline.get_columns_by_tag(tag)
        unique_cols = {(c.table_name, c.column_name) for c in cols}
        print(f"    '{tag}': {len(unique_cols)} unique columns")
    print()

    # -------------------------------------------------------------------------
    # 6. Tracing PII Through Lineage
    # -------------------------------------------------------------------------
    print("6. TRACING PII THROUGH LINEAGE")
    print("-" * 80)
    print("""
  Combine metadata with lineage to understand PII data flow:
    - Forward trace: Where does PII data go?
    - Backward trace: Where did PII originate?
""")

    # Find a PII column in a derived table (not raw_)
    pii_derived_cols = [
        col
        for col in pipeline.get_pii_columns()
        if not col.table_name.startswith("raw_") and not col.table_name.startswith("source_")
    ]

    if pii_derived_cols:
        # Pick one example
        example_col = pii_derived_cols[0]
        print(f"  Example: {example_col.table_name}.{example_col.column_name}")
        print()

        # Trace backward to find PII source
        print("  Backward trace (where did PII originate?):")
        try:
            sources = pipeline.trace_column_backward(
                example_col.table_name, example_col.column_name
            )
            for source in sources:
                pii_flag = " [PII SOURCE]" if source.pii else ""
                print(f"    <- {source.table_name}.{source.column_name}{pii_flag}")
        except Exception as e:
            print(f"    Error: {e}")
    else:
        print("  No PII columns found in derived tables.")

    # Forward trace from a source PII column
    print()
    print("  Forward trace (where does raw_customers.email go?):")
    try:
        impacts = pipeline.trace_column_forward("raw_customers", "email")
        if impacts:
            for impact in impacts[:8]:
                print(f"    -> {impact.table_name}.{impact.column_name}")
            if len(impacts) > 8:
                print(f"    ... and {len(impacts) - 8} more")
        else:
            print("    (no downstream impacts found)")
    except Exception as e:
        print(f"    Error: {e}")
    print()

    # -------------------------------------------------------------------------
    # 7. Save and Load Metadata
    # -------------------------------------------------------------------------
    print("7. SAVE AND LOAD METADATA")
    print("-" * 80)
    print("""
  Metadata can be persisted and reloaded:
    pipeline.save("metadata.pkl")          # Save metadata
    metadata = Pipeline.load_metadata("metadata.pkl")  # Load
    new_pipeline.apply_metadata(metadata)  # Apply to new pipeline
""")

    # Save metadata to temp file
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        temp_path = f.name

    pipeline.save(temp_path)
    print(f"  Saved metadata to: {temp_path}")

    # Load metadata
    loaded_metadata = Pipeline.load_metadata(temp_path)
    print("  Loaded metadata:")
    print(f"    - Version: {loaded_metadata.get('version')}")
    print(f"    - Columns: {len(loaded_metadata.get('columns', {}))}")
    print(f"    - Tables: {len(loaded_metadata.get('tables', {}))}")

    # Create a fresh pipeline and apply metadata
    print("\n  Creating new pipeline and applying loaded metadata...")
    fresh_pipeline = Pipeline(queries, dialect="duckdb")
    pii_before_apply = len(fresh_pipeline.get_pii_columns())
    fresh_pipeline.apply_metadata(loaded_metadata)
    pii_after_apply = len(fresh_pipeline.get_pii_columns())
    print(f"    PII columns before apply: {pii_before_apply}")
    print(f"    PII columns after apply:  {pii_after_apply}")

    # Clean up
    Path(temp_path).unlink()
    print()

    # -------------------------------------------------------------------------
    # 8. Export to JSON
    # -------------------------------------------------------------------------
    print("8. EXPORT TO JSON")
    print("-" * 80)
    print("""
  Export pipeline with metadata to JSON for external tools:
    data = pipeline.to_json(include_metadata=True)
""")

    json_data = pipeline.to_json(include_metadata=True)

    print("  JSON export contains:")
    print(f"    - columns: {len(json_data.get('columns', []))} entries")
    print(f"    - edges: {len(json_data.get('edges', []))} entries")
    print(f"    - tables: {len(json_data.get('tables', []))} entries")

    # Show sample column with metadata
    print("\n  Sample column entry:")
    sample_col = None
    for col_data in json_data.get("columns", []):
        if col_data.get("pii") or col_data.get("owner"):
            sample_col = col_data
            break

    if sample_col:
        # Pretty print with limited fields
        display_data = {
            k: v
            for k, v in sample_col.items()
            if k in ["table_name", "column_name", "pii", "owner", "tags", "description"]
        }
        print(f"    {json.dumps(display_data, indent=6)}")

    # Export to file example
    print("\n  To save to file:")
    print('    with open("lineage.json", "w") as f:')
    print("        json.dump(pipeline.to_json(), f, indent=2)")
    print()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("METADATA SUMMARY")
    print("=" * 80)

    # Count unique columns with each type of metadata
    unique_pii = set()
    unique_owners = {}
    unique_tags = {}

    for col in pipeline.columns.values():
        key = (col.table_name, col.column_name)
        if col.pii:
            unique_pii.add(key)
        if col.owner:
            if col.owner not in unique_owners:
                unique_owners[col.owner] = set()
            unique_owners[col.owner].add(key)
        for tag in col.tags:
            if tag not in unique_tags:
                unique_tags[tag] = set()
            unique_tags[tag].add(key)

    print(f"  Total columns:     {len(pipeline.columns)}")
    print(f"  PII columns:       {len(unique_pii)} unique")
    print(f"  Owned columns:     {sum(len(v) for v in unique_owners.values())} unique")
    print(f"  Tagged columns:    {sum(len(v) for v in unique_tags.values())} unique")
    print()
    print("  Owners:")
    for owner in sorted(unique_owners.keys()):
        print(f"    - {owner}: {len(unique_owners[owner])} columns")
    print()
    print("  Tags:")
    for tag in sorted(unique_tags.keys()):
        print(f"    - {tag}: {len(unique_tags[tag])} columns")
    print()
    print("Metadata management complete!")


if __name__ == "__main__":
    main()
