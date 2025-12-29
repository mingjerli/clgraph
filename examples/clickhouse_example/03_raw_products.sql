-- Raw Products Table
-- Source: Product catalog system
-- @layer: raw
-- @refresh: daily

CREATE TABLE IF NOT EXISTS raw_{{env}}.products (
    product_id String,
    name String,
    category String,
    subcategory String,
    price Decimal(10, 2),
    cost Decimal(10, 2),
    supplier_id String,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (product_id);
