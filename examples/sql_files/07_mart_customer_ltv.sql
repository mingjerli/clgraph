-- Mart: Customer lifetime value
CREATE OR REPLACE TABLE mart_customer_ltv AS

WITH customer_metrics AS (
    SELECT
        customer_id,
        customer_email,
        customer_full_name,
        customer_country,
        customer_loyalty_tier,
        customer_registration_date,

        MIN(order_date) AS first_order_date,
        MAX(order_date) AS last_order_date,
        COUNT(DISTINCT order_id) AS total_orders,
        SUM(total_amount) AS lifetime_revenue,
        SUM(total_items) AS lifetime_items_purchased,
        SUM(discount_amount) AS lifetime_discounts_received,
        AVG(total_amount) AS avg_order_value,
        AVG(total_items) AS avg_items_per_order,

        COUNT(DISTINCT order_date) AS distinct_order_days,
        COUNT(DISTINCT date_trunc('month', order_date::DATE)) AS active_months,

        SUM(CASE WHEN used_promo THEN 1 ELSE 0 END) AS promo_orders

    FROM stg_orders_enriched
    GROUP BY
        customer_id,
        customer_email,
        customer_full_name,
        customer_country,
        customer_loyalty_tier,
        customer_registration_date
),

customer_rfm AS (
    SELECT
        cm.*,

        current_date - last_order_date::DATE AS days_since_last_order,
        ROUND(total_orders::FLOAT / GREATEST(active_months, 1), 2) AS orders_per_month,
        ROUND(lifetime_revenue / GREATEST(active_months, 1), 2) AS monthly_spend,
        current_date - customer_registration_date::DATE AS tenure_days,
        first_order_date::DATE - customer_registration_date::DATE AS days_to_first_order,

        CASE
            WHEN total_orders > 1 THEN
                (last_order_date::DATE - first_order_date::DATE) / (total_orders - 1)
            ELSE NULL
        END AS avg_days_between_orders

    FROM customer_metrics cm
)

SELECT
    customer_id,
    customer_email,
    customer_full_name,
    customer_country,
    customer_loyalty_tier,
    customer_registration_date,

    date_trunc('month', customer_registration_date::DATE) AS registration_cohort,
    date_trunc('month', first_order_date::DATE) AS first_order_cohort,

    first_order_date,
    last_order_date,
    total_orders,
    lifetime_revenue,
    lifetime_items_purchased,
    lifetime_discounts_received,
    avg_order_value,
    avg_items_per_order,

    active_months,
    tenure_days,
    days_since_last_order,
    days_to_first_order,
    avg_days_between_orders,
    orders_per_month,
    monthly_spend,

    promo_orders,
    ROUND(promo_orders::FLOAT / total_orders * 100, 2) AS promo_usage_rate,

    -- RFM Scores
    CASE
        WHEN days_since_last_order <= 30 THEN 4
        WHEN days_since_last_order <= 60 THEN 3
        WHEN days_since_last_order <= 90 THEN 2
        ELSE 1
    END AS recency_score,

    CASE
        WHEN total_orders >= 10 THEN 4
        WHEN total_orders >= 5 THEN 3
        WHEN total_orders >= 2 THEN 2
        ELSE 1
    END AS frequency_score,

    CASE
        WHEN lifetime_revenue >= 1000 THEN 4
        WHEN lifetime_revenue >= 500 THEN 3
        WHEN lifetime_revenue >= 100 THEN 2
        ELSE 1
    END AS monetary_score,

    -- Customer segment
    CASE
        WHEN days_since_last_order <= 30 AND total_orders >= 5 AND lifetime_revenue >= 500 THEN 'Champions'
        WHEN days_since_last_order <= 30 AND total_orders >= 5 THEN 'Loyal Customers'
        WHEN days_since_last_order <= 30 AND lifetime_revenue >= 500 THEN 'Big Spenders'
        WHEN days_since_last_order <= 30 THEN 'Recent Customers'
        WHEN total_orders >= 5 AND lifetime_revenue >= 500 THEN 'At Risk - High Value'
        WHEN days_since_last_order > 60 AND total_orders <= 2 THEN 'Hibernating'
        ELSE 'Needs Attention'
    END AS customer_segment,

    -- Churn risk
    CASE
        WHEN days_since_last_order > 90 AND total_orders > 1 THEN 'High'
        WHEN days_since_last_order > 60 THEN 'Medium'
        WHEN days_since_last_order > 30 THEN 'Low'
        ELSE 'Active'
    END AS churn_risk,

    TRUE AS contains_pii,
    current_timestamp AS etl_loaded_at

FROM customer_rfm
