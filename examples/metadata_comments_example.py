"""
Example: Extracting metadata from SQL inline comments.

Demonstrates how inline SQL comments can automatically populate
column metadata (descriptions, PII flags, ownership, tags).
"""

from clgraph.pipeline import Pipeline

# SQL with inline metadata comments
sql = """
SELECT
  user_id,  -- User identifier [pii: false]
  email,    -- Email address [pii: true, owner: data-team]

  UPPER(email) as email_upper,  -- Uppercased email [pii: true]

  COUNT(*) as login_count,  -- Number of logins [tags: metric engagement]

  SUM(revenue) as total_revenue  /* Total revenue [pii: false, owner: finance-team, tags: metric revenue] */

FROM user_activity
GROUP BY user_id, email
"""

# Create pipeline
pipeline = Pipeline([("user_metrics", sql)], dialect="bigquery")

# Display extracted metadata
print("=" * 70)
print("SQL COMMENT METADATA EXTRACTION")
print("=" * 70)

for _full_name, col in sorted(pipeline.columns.items()):
    # Only show output columns (those starting with query name)
    if col.full_name.startswith("user_metrics."):
        print(f"\n📊 {col.column_name}")
        print(f"   Full name: {col.full_name}")

        if col.description:
            print(f"   📝 Description: {col.description}")

        if col.pii:
            print("   🔒 PII: Yes")

        if col.owner:
            print(f"   👤 Owner: {col.owner}")

        if col.tags:
            print(f"   🏷️  Tags: {', '.join(sorted(col.tags))}")

        if col.custom_metadata:
            print(f"   ⚙️  Custom: {col.custom_metadata}")

print("\n" + "=" * 70)
print("\n✅ Metadata automatically extracted from SQL comments!")
print("   Format: -- Description [key: value, key2: value2]")
print("\nSupported metadata:")
print("   • description - Natural language description")
print("   • pii - PII flag (true/false)")
print("   • owner - Data owner/team")
print("   • tags - Space-separated tags")
print("   • custom fields - Any other key-value pairs")
print("=" * 70)
