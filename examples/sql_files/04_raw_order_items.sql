-- Raw order items from source
CREATE OR REPLACE TABLE raw_order_items AS
SELECT
    order_item_id,
    order_id,
    product_id,
    quantity,
    unit_price,
    discount_percent,
    line_total,
    created_at
FROM source_order_items
WHERE created_at >= '2023-01-01'
