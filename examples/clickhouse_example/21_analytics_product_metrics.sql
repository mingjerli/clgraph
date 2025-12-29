-- Analytics: Product Metrics
-- Product performance and profitability analysis
-- @layer: analytics
-- @refresh: daily
-- @owner: analytics_team

CREATE OR REPLACE TABLE analytics_{{env}}.product_metrics
ENGINE = MergeTree()
ORDER BY tuple()
AS
SELECT
    -- Product identifier
    p.product_id,
    p.name AS product_name,
    p.category,
    p.subcategory,
    p.cost,  -- Needed for GROUP BY

    -- Sales volume
    -- @description: Total units sold across all orders
    SUM(oi.quantity) AS total_units_sold,

    -- @description: Number of orders containing this product
    COUNT(DISTINCT oi.order_id) AS order_count,

    -- Revenue metrics
    -- @description: Total revenue from this product
    SUM(oi.quantity * oi.unit_price) AS total_revenue,

    -- @description: Average selling price per unit
    AVG(oi.unit_price) AS avg_selling_price,

    -- Profitability
    -- @description: Total cost of goods sold
    SUM(oi.quantity * p.cost) AS total_cogs,

    -- @description: Gross profit from this product
    SUM(oi.quantity * (oi.unit_price - p.cost)) AS gross_profit,

    -- @description: Actual margin percentage achieved
    CASE
        WHEN SUM(oi.quantity * oi.unit_price) > 0
        THEN round(SUM(oi.quantity * (oi.unit_price - p.cost)) / SUM(oi.quantity * oi.unit_price) * 100, 2)
        ELSE 0
    END AS realized_margin_pct,

    -- Discount analysis
    -- @description: Average discount percentage applied
    AVG(oi.discount_pct) AS avg_discount_pct,

    -- Product performance tier
    -- @description: Product performance tier based on revenue
    CASE
        WHEN SUM(oi.quantity * oi.unit_price) >= 100000 THEN 'star'
        WHEN SUM(oi.quantity * oi.unit_price) >= 50000 THEN 'performer'
        WHEN SUM(oi.quantity * oi.unit_price) >= 10000 THEN 'standard'
        ELSE 'niche'
    END AS performance_tier,

    -- Audit
    now() AS calculated_at

FROM staging_{{env}}.products p
LEFT JOIN raw_{{env}}.order_items oi ON p.product_id = oi.product_id
LEFT JOIN staging_{{env}}.orders o ON oi.order_id = o.order_id AND o.is_valid = 1
GROUP BY p.product_id, p.name, p.category, p.subcategory, p.cost;
