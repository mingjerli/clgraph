-- Staging Orders Table
-- Cleaned and validated orders with business rules applied
-- @layer: staging
-- @refresh: daily
-- @owner: data_engineering

CREATE OR REPLACE TABLE staging_{{env}}.orders
ENGINE = MergeTree()
ORDER BY (order_id, order_date)
AS
SELECT
    -- Primary key
    order_id,

    -- Foreign key
    customer_id,

    -- Date fields
    toDate(order_date) AS order_date,
    toStartOfMonth(order_date) AS order_month,

    -- Amount fields (cleaned)
    -- @description: Total order amount in USD after validation
    CASE
        WHEN total_amount < 0 THEN 0
        ELSE total_amount
    END AS amount,

    -- Status (normalized)
    -- @description: Order status normalized to standard values
    lower(status) AS status,

    -- Payment info
    payment_method,

    -- Validation flag
    -- @description: Order validity flag based on business rules
    CASE
        WHEN total_amount > 0
            AND status IN ('completed', 'processing', 'shipped', 'pending')
            AND customer_id != ''
        THEN 1
        ELSE 0
    END AS is_valid,

    -- Audit
    now() AS processed_at

FROM raw_{{env}}.orders
WHERE order_date >= '2023-01-01';
