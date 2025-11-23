-- Intermediate: Daily aggregated metrics
CREATE OR REPLACE TABLE int_daily_metrics AS

WITH daily_orders AS (
    SELECT
        order_date,
        customer_country,
        channel,

        COUNT(DISTINCT order_id) AS order_count,
        COUNT(DISTINCT customer_id) AS unique_customers,
        SUM(total_amount) AS gross_revenue,
        SUM(discount_amount) AS total_discounts,
        SUM(total_amount) - SUM(discount_amount) AS net_revenue,
        SUM(shipping_amount) AS shipping_revenue,

        SUM(total_items) AS items_sold,
        COUNT(DISTINCT CASE WHEN customer_prior_orders = 0 THEN customer_id END) AS new_customers,
        COUNT(DISTINCT CASE WHEN customer_prior_orders > 0 THEN customer_id END) AS returning_customers,
        SUM(CASE WHEN used_promo THEN 1 ELSE 0 END) AS promo_orders,

        AVG(total_amount) AS avg_order_value,
        AVG(total_items) AS avg_items_per_order

    FROM stg_orders_enriched
    GROUP BY order_date, customer_country, channel
),

daily_with_running AS (
    SELECT
        d.*,

        -- Running totals (MTD)
        SUM(gross_revenue) OVER (
            PARTITION BY customer_country, channel, date_trunc('month', order_date::DATE)
            ORDER BY order_date
        ) AS mtd_gross_revenue,

        SUM(order_count) OVER (
            PARTITION BY customer_country, channel, date_trunc('month', order_date::DATE)
            ORDER BY order_date
        ) AS mtd_order_count,

        -- 7-day moving averages
        AVG(gross_revenue) OVER (
            PARTITION BY customer_country, channel
            ORDER BY order_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS revenue_7d_avg,

        -- Previous day values
        LAG(gross_revenue, 1) OVER (
            PARTITION BY customer_country, channel
            ORDER BY order_date
        ) AS prev_day_revenue,

        LAG(order_count, 1) OVER (
            PARTITION BY customer_country, channel
            ORDER BY order_date
        ) AS prev_day_orders,

        -- Same day last week
        LAG(gross_revenue, 7) OVER (
            PARTITION BY customer_country, channel
            ORDER BY order_date
        ) AS same_day_last_week_revenue

    FROM daily_orders d
)

SELECT
    order_date,
    customer_country,
    channel,
    date_trunc('week', order_date::DATE) AS week_start,
    date_trunc('month', order_date::DATE) AS month_start,
    dayofweek(order_date::DATE) AS day_of_week,

    order_count,
    unique_customers,
    gross_revenue,
    net_revenue,
    total_discounts,
    shipping_revenue,
    items_sold,

    new_customers,
    returning_customers,
    promo_orders,

    avg_order_value,
    avg_items_per_order,

    mtd_gross_revenue,
    mtd_order_count,
    revenue_7d_avg,

    prev_day_revenue,
    prev_day_orders,
    same_day_last_week_revenue,

    -- Growth rates
    CASE
        WHEN prev_day_revenue > 0
        THEN ROUND((gross_revenue - prev_day_revenue) / prev_day_revenue * 100, 2)
        ELSE NULL
    END AS revenue_dod_growth_pct,

    CASE
        WHEN same_day_last_week_revenue > 0
        THEN ROUND((gross_revenue - same_day_last_week_revenue) / same_day_last_week_revenue * 100, 2)
        ELSE NULL
    END AS revenue_wow_growth_pct,

    current_timestamp AS etl_loaded_at

FROM daily_with_running
