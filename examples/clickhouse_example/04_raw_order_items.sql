-- Raw Order Items Table
-- Source: Order management system (line items)
-- @layer: raw
-- @refresh: daily

CREATE TABLE IF NOT EXISTS raw_{{env}}.order_items (
    item_id String,
    order_id String,
    product_id String,
    quantity UInt32,
    unit_price Decimal(10, 2),
    discount_pct Decimal(5, 2) DEFAULT 0,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (order_id, item_id);
