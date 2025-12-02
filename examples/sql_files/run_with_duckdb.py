"""
E-Commerce Pipeline Execution with DuckDB

This script demonstrates how to:
1. Generate fake source data
2. Execute a SQL pipeline against DuckDB
3. Use clgraph's pipeline.run() for orchestrated execution
"""

import random
from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb

from clgraph import Pipeline


def generate_fake_data(
    conn: duckdb.DuckDBPyConnection,
    num_customers: int = 100,
    num_products: int = 50,
    num_orders: int = 500,
):
    """Generate fake e-commerce data in DuckDB source tables."""

    print("Generating fake data...")

    # --- Source Customers ---
    customers = []
    for i in range(1, num_customers + 1):
        reg_date = date(2020, 1, 1) + timedelta(days=random.randint(0, 1500))
        customers.append(
            {
                "customer_id": i,
                "email": f"customer{i}@example.com",
                "first_name": f"First{i}",
                "last_name": f"Last{i}",
                "phone_number": f"+1-555-{i:04d}",
                "registration_date": reg_date,
                "country_code": random.choice(["US", "CA", "UK", "DE", "FR"]),
                "city": random.choice(
                    ["New York", "Los Angeles", "Chicago", "Toronto", "London", "Berlin", "Paris"]
                ),
                "loyalty_tier": random.choice(["Bronze", "Silver", "Gold", "Platinum"]),
                "created_at": datetime.now(),
            }
        )

    conn.execute(
        """
        CREATE OR REPLACE TABLE source_customers AS
        SELECT * FROM (VALUES
            {})
        AS t(customer_id, email, first_name, last_name, phone_number,
             registration_date, country_code, city, loyalty_tier, created_at)
    """.format(
            ",\n            ".join(
                [
                    f"({c['customer_id']}, '{c['email']}', '{c['first_name']}', '{c['last_name']}', "
                    f"'{c['phone_number']}', '{c['registration_date']}', '{c['country_code']}', "
                    f"'{c['city']}', '{c['loyalty_tier']}', '{c['created_at']}')"
                    for c in customers
                ]
            )
        )
    )
    print(f"  Created source_customers: {num_customers} rows")

    # --- Source Products ---
    categories = ["Electronics", "Clothing", "Home & Garden", "Sports", "Books"]
    brands = ["BrandA", "BrandB", "BrandC", "BrandD", "BrandE"]
    products = []
    for i in range(1, num_products + 1):
        unit_cost = round(random.uniform(5, 100), 2)
        products.append(
            {
                "product_id": i,
                "sku": f"SKU-{i:05d}",
                "product_name": f"Product {i}",
                "category_name": random.choice(categories),
                "brand": random.choice(brands),
                "unit_cost": unit_cost,
                "unit_price": round(unit_cost * random.uniform(1.2, 2.5), 2),
                "is_active": True,
                "created_at": datetime.now(),
            }
        )

    conn.execute(
        """
        CREATE OR REPLACE TABLE source_products AS
        SELECT * FROM (VALUES
            {})
        AS t(product_id, sku, product_name, category_name, brand,
             unit_cost, unit_price, is_active, created_at)
    """.format(
            ",\n            ".join(
                [
                    f"({p['product_id']}, '{p['sku']}', '{p['product_name']}', '{p['category_name']}', "
                    f"'{p['brand']}', {p['unit_cost']}, {p['unit_price']}, {p['is_active']}, '{p['created_at']}')"
                    for p in products
                ]
            )
        )
    )
    print(f"  Created source_products: {num_products} rows")

    # --- Source Orders ---
    statuses = [
        "completed",
        "completed",
        "completed",
        "completed",
        "shipped",
        "processing",
        "cancelled",
    ]
    channels = ["web", "mobile", "store"]
    devices = ["desktop", "mobile", "tablet"]
    payments = ["credit_card", "debit_card", "paypal", "apple_pay"]

    orders = []
    order_items = []
    order_item_id = 1

    for i in range(1, num_orders + 1):
        customer_id = random.randint(1, num_customers)
        order_date = date(2023, 1, 1) + timedelta(days=random.randint(0, 365))
        order_timestamp = datetime.combine(order_date, datetime.min.time()) + timedelta(
            hours=random.randint(8, 22)
        )

        # Generate order items
        num_items = random.randint(1, 5)
        subtotal = 0
        for _ in range(num_items):
            product = random.choice(products)
            quantity = random.randint(1, 3)
            discount_pct = random.choice([0, 0, 0, 5, 10, 15, 20])
            unit_price = product["unit_price"] * (1 - discount_pct / 100)
            line_total = round(quantity * unit_price, 2)
            subtotal += line_total

            order_items.append(
                {
                    "order_item_id": order_item_id,
                    "order_id": i,
                    "product_id": product["product_id"],
                    "quantity": quantity,
                    "unit_price": round(unit_price, 2),
                    "discount_percent": discount_pct,
                    "line_total": line_total,
                    "created_at": order_timestamp,
                }
            )
            order_item_id += 1

        tax = round(subtotal * 0.08, 2)
        shipping = round(random.uniform(5, 20), 2) if subtotal < 100 else 0
        discount = round(subtotal * random.choice([0, 0, 0.05, 0.1]), 2)
        total = round(subtotal + tax + shipping - discount, 2)

        orders.append(
            {
                "order_id": i,
                "customer_id": customer_id,
                "order_date": order_date,
                "order_timestamp": order_timestamp,
                "status": random.choice(statuses),
                "shipping_address": f"{random.randint(100, 999)} Main St",
                "payment_method": random.choice(payments),
                "subtotal_amount": round(subtotal, 2),
                "tax_amount": tax,
                "shipping_amount": shipping,
                "discount_amount": discount,
                "total_amount": total,
                "channel": random.choice(channels),
                "device_type": random.choice(devices),
                "ip_address": f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}",
                "created_at": order_timestamp,
            }
        )

    conn.execute(
        """
        CREATE OR REPLACE TABLE source_orders AS
        SELECT * FROM (VALUES
            {})
        AS t(order_id, customer_id, order_date, order_timestamp, status,
             shipping_address, payment_method, subtotal_amount, tax_amount,
             shipping_amount, discount_amount, total_amount, channel, device_type,
             ip_address, created_at)
    """.format(
            ",\n            ".join(
                [
                    f"({o['order_id']}, {o['customer_id']}, '{o['order_date']}', '{o['order_timestamp']}', "
                    f"'{o['status']}', '{o['shipping_address']}', '{o['payment_method']}', "
                    f"{o['subtotal_amount']}, {o['tax_amount']}, {o['shipping_amount']}, "
                    f"{o['discount_amount']}, {o['total_amount']}, '{o['channel']}', "
                    f"'{o['device_type']}', '{o['ip_address']}', '{o['created_at']}')"
                    for o in orders
                ]
            )
        )
    )
    print(f"  Created source_orders: {num_orders} rows")

    conn.execute(
        """
        CREATE OR REPLACE TABLE source_order_items AS
        SELECT * FROM (VALUES
            {})
        AS t(order_item_id, order_id, product_id, quantity, unit_price,
             discount_percent, line_total, created_at)
    """.format(
            ",\n            ".join(
                [
                    f"({oi['order_item_id']}, {oi['order_id']}, {oi['product_id']}, "
                    f"{oi['quantity']}, {oi['unit_price']}, {oi['discount_percent']}, "
                    f"{oi['line_total']}, '{oi['created_at']}')"
                    for oi in order_items
                ]
            )
        )
    )
    print(f"  Created source_order_items: {len(order_items)} rows")

    print("  Done generating fake data!")
    print()


def load_sql_queries(sql_dir: Path) -> list[tuple[str, str]]:
    """Load all SQL files from directory in sorted order."""
    queries = []
    for sql_file in sorted(sql_dir.glob("*.sql")):
        with open(sql_file) as f:
            sql = f.read()
        queries.append((sql_file.stem, sql))
    return queries


def main():
    print("=" * 80)
    print("E-Commerce Pipeline Execution with DuckDB")
    print("=" * 80)
    print()

    # Create in-memory DuckDB connection
    conn = duckdb.connect(":memory:")

    # Generate fake source data
    generate_fake_data(conn, num_customers=100, num_products=50, num_orders=500)

    # Load SQL files
    sql_dir = Path(__file__).parent
    queries = load_sql_queries(sql_dir)

    print(f"Loaded {len(queries)} SQL files:")
    for name, _ in queries:
        print(f"  • {name}")
    print()

    # Build pipeline for lineage analysis
    print("Building pipeline (for lineage analysis)...")
    pipeline = Pipeline(queries, dialect="duckdb")
    print(f"  ✓ {len(pipeline.table_graph.queries)} queries")
    print(f"  ✓ {len(pipeline.columns)} columns tracked")
    print()

    # Define executor function for DuckDB
    def execute_sql(sql: str):
        """Execute SQL against DuckDB connection."""
        conn.execute(sql)

    # Execute the pipeline
    print("Executing pipeline...")
    print("-" * 80)
    result = pipeline.run(executor=execute_sql, max_workers=1, verbose=True)
    print()

    # Show execution results
    print("=" * 80)
    print("EXECUTION RESULTS")
    print("=" * 80)
    print(f"  Completed: {len(result['completed'])} queries")
    print(f"  Failed:    {len(result['failed'])} queries")
    print(f"  Time:      {result['elapsed_seconds']:.2f} seconds")
    print()

    if result["failed"]:
        print("Failed queries:")
        for query_id, error in result["failed"]:
            print(f"  ✗ {query_id}: {error}")
        print()

    # Verify tables were created
    print("Tables created in DuckDB:")
    tables = conn.execute("SHOW TABLES").fetchall()
    for (table,) in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  • {table}: {count} rows")
    print()

    # Show sample data from mart tables
    print("Sample data from mart_customer_ltv:")
    print("-" * 80)
    df = conn.execute("""
        SELECT customer_id, customer_full_name, total_orders,
               lifetime_revenue, customer_segment, churn_risk
        FROM mart_customer_ltv
        ORDER BY lifetime_revenue DESC
        LIMIT 5
    """).fetchdf()
    print(df.to_string(index=False))
    print()

    print("Sample data from mart_product_performance:")
    print("-" * 80)
    df = conn.execute("""
        SELECT product_name, category_name, units_sold,
               total_revenue, performance_tier
        FROM mart_product_performance
        ORDER BY total_revenue DESC
        LIMIT 5
    """).fetchdf()
    print(df.to_string(index=False))
    print()

    print("=" * 80)
    print("Pipeline execution complete!")
    print("=" * 80)

    conn.close()


if __name__ == "__main__":
    main()
