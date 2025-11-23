-- Staging: Enriched orders with customer and product info
CREATE OR REPLACE TABLE stg_orders_enriched AS

WITH order_items_agg AS (
    SELECT
        oi.order_id,
        COUNT(DISTINCT oi.product_id) AS distinct_products,
        SUM(oi.quantity) AS total_items,
        SUM(oi.line_total) AS items_total,
        MAX(p.unit_price) AS max_item_price,
        MIN(p.unit_price) AS min_item_price
    FROM raw_order_items oi
    INNER JOIN raw_products p ON oi.product_id = p.product_id
    GROUP BY oi.order_id
),

customer_orders_prior AS (
    SELECT
        o.order_id,
        o.customer_id,
        COUNT(*) OVER (
            PARTITION BY o.customer_id
            ORDER BY o.order_timestamp
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS prior_order_count,
        SUM(o.total_amount) OVER (
            PARTITION BY o.customer_id
            ORDER BY o.order_timestamp
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS prior_total_spend,
        LAG(o.order_timestamp) OVER (
            PARTITION BY o.customer_id
            ORDER BY o.order_timestamp
        ) AS previous_order_timestamp
    FROM raw_orders o
)

SELECT
    -- Order identifiers
    o.order_id,
    o.customer_id,
    o.order_date,
    o.order_timestamp,
    o.status,
    o.channel,
    o.device_type,
    o.payment_method,

    -- Financial amounts
    o.subtotal_amount,
    o.tax_amount,
    o.shipping_amount,
    o.discount_amount,
    o.total_amount,

    -- Order composition
    oia.distinct_products,
    oia.total_items,
    oia.items_total,
    oia.max_item_price,
    oia.min_item_price,

    -- Customer info
    c.email AS customer_email,
    c.first_name AS customer_first_name,
    c.last_name AS customer_last_name,
    c.first_name || ' ' || c.last_name AS customer_full_name,
    c.country_code AS customer_country,
    c.city AS customer_city,
    c.loyalty_tier AS customer_loyalty_tier,
    c.registration_date AS customer_registration_date,

    -- Customer history
    COALESCE(cop.prior_order_count, 0) AS customer_prior_orders,
    COALESCE(cop.prior_total_spend, 0) AS customer_prior_spend,
    cop.previous_order_timestamp,

    -- Derived fields
    CASE
        WHEN COALESCE(cop.prior_order_count, 0) = 0 THEN 'New'
        WHEN COALESCE(cop.prior_order_count, 0) BETWEEN 1 AND 3 THEN 'Repeat'
        WHEN COALESCE(cop.prior_order_count, 0) BETWEEN 4 AND 10 THEN 'Loyal'
        ELSE 'VIP'
    END AS customer_segment,

    CASE WHEN o.discount_amount > 0 THEN TRUE ELSE FALSE END AS used_promo,

    o.order_date::DATE - c.registration_date::DATE AS days_since_registration,

    current_timestamp AS etl_loaded_at

FROM raw_orders o
LEFT JOIN raw_customers c ON o.customer_id = c.customer_id
LEFT JOIN order_items_agg oia ON o.order_id = oia.order_id
LEFT JOIN customer_orders_prior cop ON o.order_id = cop.order_id
WHERE o.status != 'cancelled'
