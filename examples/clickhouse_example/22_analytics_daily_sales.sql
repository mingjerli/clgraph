-- Analytics: Daily Sales
-- Daily aggregated sales metrics for reporting
-- @layer: analytics
-- @refresh: daily
-- @owner: analytics_team

CREATE OR REPLACE TABLE analytics_{{env}}.daily_sales
ENGINE = MergeTree()
ORDER BY tuple()
AS
SELECT
    -- Date dimension
    o.order_date AS sale_date,
    toStartOfWeek(o.order_date) AS week_start,
    toStartOfMonth(o.order_date) AS month_start,
    toDayOfWeek(o.order_date) AS day_of_week,

    -- Order metrics
    -- @description: Number of orders placed on this day
    COUNT(DISTINCT o.order_id) AS order_count,

    -- @description: Number of unique customers who ordered
    COUNT(DISTINCT o.customer_id) AS unique_customers,

    -- Revenue metrics
    -- @description: Total revenue for the day
    SUM(o.amount) AS total_revenue,

    -- @description: Average order value for the day
    AVG(o.amount) AS avg_order_value,

    -- @description: Minimum order value
    MIN(o.amount) AS min_order_value,

    -- @description: Maximum order value
    MAX(o.amount) AS max_order_value,

    -- Order status breakdown
    -- @description: Number of completed orders
    SUM(CASE WHEN o.status = 'completed' THEN 1 ELSE 0 END) AS completed_orders,

    -- @description: Number of pending orders
    SUM(CASE WHEN o.status = 'pending' THEN 1 ELSE 0 END) AS pending_orders,

    -- @description: Number of cancelled orders
    SUM(CASE WHEN o.status = 'cancelled' THEN 1 ELSE 0 END) AS cancelled_orders,

    -- Conversion metrics
    -- @description: Order completion rate
    round(SUM(CASE WHEN o.status = 'completed' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS completion_rate,

    -- Audit
    now() AS calculated_at

FROM staging_{{env}}.orders o
WHERE o.is_valid = 1
GROUP BY o.order_date;
