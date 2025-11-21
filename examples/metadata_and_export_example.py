"""
Comprehensive example: Metadata management and export functionality

This example demonstrates:
1. Building a pipeline with multiple queries
2. Setting source metadata (descriptions, PII flags, ownership)
3. Propagating metadata through lineage
4. Generating descriptions with LLM (optional)
5. Exporting to multiple formats (JSON, CSV, GraphViz)
6. Using diff to track changes
"""

import sys
import tempfile
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clpipe import (
    CSVExporter,
    GraphVizExporter,
    JSONExporter,
    MultiQueryParser,
    PipelineLineageBuilder,
)


def main():
    print("=" * 80)
    print("Metadata Management and Export Example")
    print("=" * 80)
    print()

    # ==========================================================================
    # Step 1: Build a realistic data pipeline
    # ==========================================================================
    print("Step 1: Building data pipeline...")
    print("-" * 80)

    sql_queries = [
        # Stage 1: Raw data ingestion
        """
        CREATE OR REPLACE TABLE raw.user_events AS
        SELECT
            user_id,
            event_type,
            event_timestamp,
            user_email,
            ip_address,
            session_id
        FROM source_system.events
        WHERE event_timestamp >= '2024-01-01'
        """,
        # Stage 2: Daily aggregation
        """
        CREATE OR REPLACE TABLE staging.daily_metrics AS
        SELECT
            user_id,
            DATE(event_timestamp) as activity_date,
            COUNT(*) as event_count,
            COUNT(DISTINCT session_id) as session_count
        FROM raw.user_events
        GROUP BY user_id, DATE(event_timestamp)
        """,
        # Stage 3: User summary
        """
        CREATE OR REPLACE TABLE analytics.user_summary AS
        SELECT
            dm.user_id,
            dm.activity_date,
            dm.event_count,
            dm.session_count,
            ue.user_email
        FROM staging.daily_metrics dm
        JOIN raw.user_events ue ON dm.user_id = ue.user_id
        """,
    ]

    parser = MultiQueryParser()
    table_graph = parser.parse_queries(sql_queries)

    builder = PipelineLineageBuilder()
    lineage_graph = builder.build(table_graph)

    print(
        f"✅ Pipeline built: {len(lineage_graph.columns)} columns, {len(lineage_graph.edges)} edges"
    )
    print()

    # ==========================================================================
    # Step 2: Set source metadata
    # ==========================================================================
    print("Step 2: Setting source metadata...")
    print("-" * 80)

    # Set descriptions and metadata for source columns
    source_metadata = {
        "source_system.events.user_id": {
            "description": "Unique identifier for the user",
            "owner": "data-platform",
            "pii": False,
        },
        "source_system.events.event_type": {
            "description": "Type of event (click, view, purchase, etc.)",
            "owner": "data-platform",
            "pii": False,
        },
        "source_system.events.event_timestamp": {
            "description": "Timestamp when the event occurred",
            "owner": "data-platform",
            "pii": False,
        },
        "source_system.events.user_email": {
            "description": "User's email address",
            "owner": "data-platform",
            "pii": True,  # PII!
            "tags": {"contact_info", "sensitive"},
        },
        "source_system.events.ip_address": {
            "description": "IP address of the user",
            "owner": "data-platform",
            "pii": True,  # PII!
            "tags": {"network", "sensitive"},
        },
        "source_system.events.session_id": {
            "description": "Unique session identifier",
            "owner": "data-platform",
            "pii": False,
        },
    }

    for full_name, metadata in source_metadata.items():
        if full_name in lineage_graph.columns:
            col = lineage_graph.columns[full_name]
            col.set_source_description(metadata["description"])
            col.owner = metadata["owner"]
            col.pii = metadata["pii"]
            if "tags" in metadata:
                col.tags = metadata["tags"]

    print(f"✅ Set metadata for {len(source_metadata)} source columns")
    print()

    # ==========================================================================
    # Step 3: Propagate metadata
    # ==========================================================================
    print("Step 3: Propagating metadata through lineage...")
    print("-" * 80)

    lineage_graph.propagate_all_metadata()

    print("✅ Metadata propagated")
    print()

    # Show PII columns
    pii_columns = lineage_graph.get_pii_columns()
    print(f"⚠️  PII Columns detected: {len(pii_columns)}")
    for col in sorted(pii_columns, key=lambda c: c.full_name):
        print(f"   • {col.full_name}")
    print()

    # ==========================================================================
    # Step 4: Export to different formats
    # ==========================================================================
    print("Step 4: Exporting lineage to different formats...")
    print("-" * 80)

    # Create temporary directory for exports
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Export to JSON
        json_file = tmp_path / "lineage.json"
        JSONExporter.export_to_file(lineage_graph, str(json_file), include_metadata=True)
        print(f"✅ JSON exported: {json_file} ({json_file.stat().st_size} bytes)")

        # Export columns to CSV
        columns_csv = tmp_path / "columns.csv"
        CSVExporter.export_columns_to_file(lineage_graph, str(columns_csv))
        print(f"✅ Columns CSV exported: {columns_csv} ({columns_csv.stat().st_size} bytes)")

        # Export tables to CSV
        tables_csv = tmp_path / "tables.csv"
        CSVExporter.export_tables_to_file(lineage_graph, str(tables_csv))
        print(f"✅ Tables CSV exported: {tables_csv} ({tables_csv.stat().st_size} bytes)")

        # Export to GraphViz DOT
        dot_file = tmp_path / "lineage.dot"
        GraphVizExporter.export_to_file(lineage_graph, str(dot_file), layout="LR", max_columns=20)
        print(f"✅ GraphViz DOT exported: {dot_file} ({dot_file.stat().st_size} bytes)")
        print()

        # Show a sample of the JSON export
        print("Sample JSON export (first 500 chars):")
        print("-" * 80)
        with open(json_file) as f:
            content = f.read()
            print(content[:500] + "..." if len(content) > 500 else content)
        print()

        # Show CSV column metadata
        print("Sample CSV columns export (first 5 rows):")
        print("-" * 80)
        with open(columns_csv) as f:
            lines = f.readlines()
            for i, line in enumerate(lines[:6]):  # Header + 5 rows
                print(f"  {line.rstrip()}")
        print()

    # ==========================================================================
    # Step 5: Demonstrate diff functionality
    # ==========================================================================
    print("Step 5: Demonstrating diff functionality...")
    print("-" * 80)

    # Create a modified version of the pipeline
    modified_queries = sql_queries.copy()
    # Add a new column to the analytics table
    modified_queries[2] = """
        CREATE OR REPLACE TABLE analytics.user_summary AS
        SELECT
            dm.user_id,
            dm.activity_date,
            dm.event_count,
            dm.session_count,
            ue.user_email,
            CURRENT_TIMESTAMP() as last_updated  -- NEW COLUMN
        FROM staging.daily_metrics dm
        JOIN raw.user_events ue ON dm.user_id = ue.user_id
    """

    # Build new graph
    new_table_graph = parser.parse_queries(modified_queries)
    new_lineage_graph = builder.build(new_table_graph)

    # Compare
    diff = new_lineage_graph.diff(lineage_graph)

    print(diff.summary())
    print()

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    print(f"📊 Pipeline Statistics:")
    print(f"   • Total columns: {len(lineage_graph.columns)}")
    print(f"   • Total edges: {len(lineage_graph.edges)}")
    print(f"   • PII columns: {len(pii_columns)}")
    print(f"   • Tables: {len(lineage_graph.table_graph.tables)}")
    print()
    print(f"✅ Exports created:")
    print(f"   • JSON (machine-readable)")
    print(f"   • CSV columns (spreadsheet)")
    print(f"   • CSV tables (spreadsheet)")
    print(f"   • GraphViz DOT (visualization)")
    print()
    print("💡 Next steps:")
    print("   • Use exported JSON for integration with other systems")
    print("   • Open CSV files in Excel/Google Sheets for review")
    print("   • Visualize DOT file with: dot -Tpng lineage.dot -o lineage.png")
    print(
        "   • Set up LLM for automatic description generation (see llm_description_generation.py)"
    )
    print()


if __name__ == "__main__":
    main()
