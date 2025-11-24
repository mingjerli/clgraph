-- Raw orders from source
CREATE OR REPLACE TABLE raw_orders AS
SELECT
    order_id,  -- Unique order identifier [owner: data-platform]
    customer_id,  -- Reference to customer [owner: data-platform]
    order_date,  -- Date order was placed [owner: finance, tags: time]
    order_timestamp,  -- Timestamp order was placed [owner: finance, tags: time]
    status,  -- Order status [owner: operations, tags: status]
    shipping_address,  -- Customer shipping address [pii: true, owner: data-governance, tags: contact]
    payment_method,  -- Payment method used [owner: finance, tags: payment]
    subtotal_amount,  -- Order subtotal before tax/shipping [owner: finance, tags: metric revenue]
    tax_amount,  -- Tax amount [owner: finance, tags: metric revenue]
    shipping_amount,  -- Shipping cost [owner: finance, tags: metric cost]
    discount_amount,  -- Discount applied [owner: finance, tags: metric]
    total_amount,  -- Total order amount [owner: finance, tags: metric revenue]
    channel,  -- Sales channel [owner: marketing, tags: attribution]
    device_type,  -- Device type used [owner: marketing, tags: attribution]
    ip_address,  -- Customer IP address [pii: true, owner: security, tags: sensitive]
    created_at  -- Record creation timestamp [owner: data-platform]
FROM source_orders
WHERE order_date >= '2023-01-01'
