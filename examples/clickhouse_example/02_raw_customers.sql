-- Raw Customers Table
-- Source: CRM system
-- @layer: raw
-- @refresh: daily
-- @pii: true

CREATE TABLE IF NOT EXISTS raw_{{env}}.customers (
    customer_id String,
    email String,
    name String,
    phone String,
    country String,
    city String,
    created_at DateTime,
    updated_at DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (customer_id);
