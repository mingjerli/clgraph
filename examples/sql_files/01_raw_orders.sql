-- Raw orders from source
CREATE OR REPLACE TABLE raw_orders AS
SELECT
    order_id,
    customer_id,
    order_date,
    order_timestamp,
    status,
    shipping_address,
    payment_method,
    subtotal_amount,
    tax_amount,
    shipping_amount,
    discount_amount,
    total_amount,
    channel,
    device_type,
    ip_address,
    created_at
FROM source_orders
WHERE order_date >= '2023-01-01'
