-- Raw Orders Table
-- Source: External order management system
-- @layer: raw
-- @refresh: daily

CREATE TABLE IF NOT EXISTS raw_{{env}}.orders (
    order_id String,
    customer_id String,
    order_date DateTime,
    total_amount Decimal(10, 2),
    status String,
    payment_method String,
    shipping_address String,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (order_date, order_id);
