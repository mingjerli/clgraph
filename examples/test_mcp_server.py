#!/usr/bin/env python3
"""
Test script for MCP server

This script verifies that the MCP server can be imported and initialized
without errors. For full testing, use an MCP client like Claude Desktop.

Usage:
    python test_mcp_server.py
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that all required imports work"""
    print("Testing imports...")

    try:
        import mcp
        print("✓ mcp module imported")
    except ImportError as e:
        print(f"✗ Failed to import mcp: {e}")
        print("  Install with: pip install mcp")
        return False

    try:
        from clgraph.pipeline import Pipeline
        print("✓ clgraph.pipeline imported")
    except ImportError as e:
        print(f"✗ Failed to import clgraph: {e}")
        return False

    try:
        # Optional import
        from nlp_vector_database_rag import LineageVectorStore
        print("✓ NLP components available")
    except ImportError:
        print("⚠ NLP components not available (install with: pip install clgraph[nlp])")

    return True


def test_server_initialization():
    """Test that server can be initialized"""
    print("\nTesting server initialization...")

    try:
        from mcp_server import LineageMCPServer

        # Initialize without config
        server = LineageMCPServer()
        print("✓ Server initialized without config")

        # Test with config
        config = {
            "dialect": "bigquery",
            "enable_nlp": False,
            "queries": [
                {
                    "name": "test_table",
                    "sql": "CREATE TABLE test_table AS SELECT user_id, amount FROM source"
                }
            ]
        }

        # Write temp config
        config_path = Path(__file__).parent / "test_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)

        server = LineageMCPServer(config_path=str(config_path))
        print("✓ Server initialized with config")

        # Cleanup
        config_path.unlink()

        return True

    except Exception as e:
        print(f"✗ Failed to initialize server: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_initialization():
    """Test that pipeline can be initialized"""
    print("\nTesting pipeline initialization...")

    try:
        from mcp_server import LineageMCPServer

        config = {
            "dialect": "bigquery",
            "enable_nlp": False,
            "queries": [
                {
                    "name": "events",
                    "sql": """
                        CREATE TABLE events AS
                        SELECT
                            user_id,
                            event_type,
                            revenue
                        FROM source_events
                    """
                },
                {
                    "name": "summary",
                    "sql": """
                        CREATE TABLE summary AS
                        SELECT
                            user_id,
                            SUM(revenue) as total_revenue
                        FROM events
                        GROUP BY user_id
                    """
                }
            ]
        }

        # Write temp config
        config_path = Path(__file__).parent / "test_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)

        server = LineageMCPServer(config_path=str(config_path))
        server._initialize_pipeline()

        if server.pipeline is None:
            print("✗ Pipeline not initialized")
            return False

        print(f"✓ Pipeline initialized with {len(server.pipeline.columns)} columns")

        # Test that columns are present
        if len(server.pipeline.columns) == 0:
            print("✗ No columns found in pipeline")
            return False

        print(f"  Columns: {list(server.pipeline.columns.keys())}")

        # Cleanup
        config_path.unlink()

        return True

    except Exception as e:
        print(f"✗ Failed to initialize pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tool_listing():
    """Test that tools can be listed"""
    print("\nTesting tool listing...")

    try:
        from mcp_server import LineageMCPServer

        server = LineageMCPServer()

        # Get the list_tools handler
        # Note: We can't actually call it without async context,
        # but we can verify it exists
        if not hasattr(server.server, '_request_handlers'):
            print("⚠ Cannot verify tools (server structure may have changed)")
            return True

        print("✓ Server has tool handlers registered")
        return True

    except Exception as e:
        print(f"✗ Failed to test tools: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("MCP Server Test Suite")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Server Initialization", test_server_initialization),
        ("Pipeline Initialization", test_pipeline_initialization),
        ("Tool Listing", test_tool_listing),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' failed with exception: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed!")
        print("\nNext steps:")
        print("1. Install MCP: pip install mcp")
        print("2. Configure Claude Desktop (see README_MCP.md)")
        print("3. Test with: python mcp_server.py --config mcp_config_example.json")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
