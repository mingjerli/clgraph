-- Analytics: Customer Metrics
-- Aggregated customer behavior and value metrics
-- @layer: analytics
-- @refresh: daily
-- @owner: analytics_team

CREATE OR REPLACE TABLE analytics_{{env}}.customer_metrics
ENGINE = MergeTree()
ORDER BY tuple()
AS
SELECT
    -- Customer identifier
    o.customer_id,

    -- Order metrics
    -- @description: Total number of orders placed by customer
    COUNT(DISTINCT o.order_id) AS total_orders,

    -- @description: Number of valid (non-cancelled) orders
    SUM(o.is_valid) AS valid_orders,

    -- Value metrics
    -- @description: Customer lifetime value (sum of all order amounts)
    SUM(o.amount) AS lifetime_value,

    -- @description: Average order value for the customer
    AVG(o.amount) AS avg_order_value,

    -- @description: Maximum single order amount
    MAX(o.amount) AS max_order_value,

    -- Recency metrics
    -- @description: Date of customer's first order
    MIN(o.order_date) AS first_order_date,

    -- @description: Date of customer's most recent order
    MAX(o.order_date) AS last_order_date,

    -- @description: Days since last order (recency)
    dateDiff('day', MAX(o.order_date), today()) AS days_since_last_order,

    -- Frequency metrics
    -- @description: Average days between orders
    CASE
        WHEN COUNT(DISTINCT o.order_id) > 1
        THEN dateDiff('day', MIN(o.order_date), MAX(o.order_date)) / (COUNT(DISTINCT o.order_id) - 1)
        ELSE 0
    END AS avg_days_between_orders,

    -- Customer segment
    -- @description: Customer value segment based on lifetime value
    CASE
        WHEN SUM(o.amount) >= 10000 THEN 'platinum'
        WHEN SUM(o.amount) >= 5000 THEN 'gold'
        WHEN SUM(o.amount) >= 1000 THEN 'silver'
        ELSE 'bronze'
    END AS value_segment,

    -- Audit
    now() AS calculated_at

FROM staging_{{env}}.orders o
WHERE o.is_valid = 1
GROUP BY o.customer_id;
