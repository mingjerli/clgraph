-- Marts: Customer 360
-- Comprehensive customer view combining all customer data
-- @layer: marts
-- @refresh: daily
-- @owner: business_intelligence

CREATE OR REPLACE TABLE marts_{{env}}.customer_360
ENGINE = MergeTree()
ORDER BY tuple()
AS
SELECT
    -- Customer identification
    c.customer_id,
    c.first_name,
    c.email_hash,

    -- Geographic info
    -- @description: Customer country
    c.country,
    c.city,

    -- Customer tenure
    -- @description: Date customer first registered
    c.customer_since,

    -- @description: Customer tenure in days
    c.tenure_days,

    -- @description: Customer tenure segment
    CASE
        WHEN c.tenure_days >= 730 THEN 'veteran'
        WHEN c.tenure_days >= 365 THEN 'established'
        WHEN c.tenure_days >= 90 THEN 'growing'
        ELSE 'new'
    END AS tenure_segment,

    -- Order metrics (from customer_metrics)
    -- @description: Total number of orders
    m.total_orders,
    m.valid_orders,

    -- Value metrics
    -- @description: Customer lifetime value in USD
    m.lifetime_value,
    m.avg_order_value,
    m.max_order_value,

    -- @description: Customer value segment
    m.value_segment,

    -- Recency metrics
    -- @description: First order date
    m.first_order_date,

    -- @description: Most recent order date
    m.last_order_date,

    -- @description: Days since last purchase
    m.days_since_last_order,

    -- @description: Customer activity status
    CASE
        WHEN m.days_since_last_order <= 30 THEN 'active'
        WHEN m.days_since_last_order <= 90 THEN 'cooling'
        WHEN m.days_since_last_order <= 180 THEN 'at_risk'
        ELSE 'churned'
    END AS activity_status,

    -- Frequency metrics
    m.avg_days_between_orders,

    -- Combined segments
    -- @description: RFM-based customer segment
    CASE
        WHEN m.value_segment = 'platinum' AND m.days_since_last_order <= 30 THEN 'champion'
        WHEN m.value_segment IN ('platinum', 'gold') AND m.days_since_last_order <= 90 THEN 'loyal'
        WHEN m.value_segment IN ('gold', 'silver') AND m.days_since_last_order > 90 THEN 'needs_attention'
        WHEN m.days_since_last_order > 180 THEN 'lost'
        ELSE 'potential'
    END AS customer_segment,

    -- Audit
    now() AS snapshot_at

FROM staging_{{env}}.customers c
LEFT JOIN analytics_{{env}}.customer_metrics m ON c.customer_id = m.customer_id;
