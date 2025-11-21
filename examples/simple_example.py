"""
Simple example demonstrating SQL column lineage analysis
"""

from clpipe import SQLColumnTracer

# Example SQL query with CTE and joins
sql = """
WITH monthly_sales AS (
  SELECT
    user_id,
    DATE_TRUNC(order_date, MONTH) as month,
    SUM(amount) as total_amount,
    COUNT(*) as order_count
  FROM orders
  WHERE status = 'completed'
  GROUP BY 1, 2
),
user_stats AS (
  SELECT
    user_id,
    AVG(total_amount) as avg_monthly_sales,
    MAX(order_count) as max_orders
  FROM monthly_sales
  GROUP BY user_id
)
SELECT
  u.name,
  u.email,
  us.avg_monthly_sales,
  us.max_orders,
  CASE
    WHEN us.avg_monthly_sales > 1000 THEN 'high'
    WHEN us.avg_monthly_sales > 500 THEN 'medium'
    ELSE 'low'
  END as customer_tier
FROM users u
JOIN user_stats us ON u.id = us.user_id
WHERE u.active = true
"""


def main():
    print("=" * 80)
    print("SQL Column Lineage Example")
    print("=" * 80)
    print()

    # Create tracer and analyze the query
    tracer = SQLColumnTracer(sql, dialect="bigquery")
    lineage = tracer.build_column_lineage_graph()

    print("Query Analysis Complete!")
    print()

    # Helper function to trace backward lineage
    def get_backward_lineage(graph, column_name):
        """Find all source columns for a given output column"""
        sources = set()

        # Find the output node
        output_node = None
        for node in graph.get_output_nodes():
            if node.column_name == column_name:
                output_node = node
                break

        if not output_node:
            return sources

        # Traverse backward through edges
        def trace_back(node):
            for edge in graph.get_edges_to(node):
                from_node = edge.from_node
                if from_node.layer == "input":
                    sources.add(from_node.full_name)
                else:
                    trace_back(from_node)

        trace_back(output_node)
        return sources

    # Helper function to trace forward lineage
    def get_forward_lineage(graph, source_name):
        """Find all output columns affected by a source column"""
        impacts = set()

        # Find the source node
        source_node = None
        for node in graph.get_input_nodes():
            if node.full_name == source_name or node.column_name == source_name.split(".")[-1]:
                source_node = node
                break

        if not source_node:
            return impacts

        # Traverse forward through edges
        def trace_forward(node):
            for edge in graph.get_edges_from(node):
                to_node = edge.to_node
                if to_node.layer == "output":
                    impacts.add(to_node.column_name)
                else:
                    trace_forward(to_node)

        trace_forward(source_node)
        return impacts

    # Example 1: Find sources for an output column
    print("1. BACKWARD LINEAGE (Where does 'avg_monthly_sales' come from?)")
    print("-" * 80)
    sources = get_backward_lineage(lineage, "avg_monthly_sales")
    for source in sorted(sources):
        print(f"  → {source}")
    print()

    # Example 2: Find impacts of a source column
    print("2. FORWARD LINEAGE (What uses 'orders.amount'?)")
    print("-" * 80)
    impacts = get_forward_lineage(lineage, "orders.amount")
    for impact in sorted(impacts):
        print(f"  → {impact}")
    print()

    # Example 3: Get all output columns
    print("3. OUTPUT COLUMNS")
    print("-" * 80)
    output_cols = lineage.get_output_nodes()
    for col in output_cols:
        print(f"  • {col.column_name}")
    print()

    # Example 4: Get all source tables
    print("4. SOURCE TABLES")
    print("-" * 80)
    input_nodes = lineage.get_input_nodes()
    source_tables = {node.table_name for node in input_nodes if node.table_name}
    for source in sorted(source_tables):
        print(f"  • {source}")
    print()

    # Example 5: Column with transformation logic
    print("5. COLUMN WITH TRANSFORMATION")
    print("-" * 80)
    print("Column: customer_tier")
    tier_sources = get_backward_lineage(lineage, "customer_tier")
    print(f"Depends on: {', '.join(sorted(tier_sources))}")
    print()

    print("=" * 80)
    print("Analysis complete! See the lineage object for more details.")
    print("=" * 80)


if __name__ == "__main__":
    main()
