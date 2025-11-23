-- Mart: Product performance
CREATE OR REPLACE TABLE mart_product_performance AS

WITH product_sales AS (
    SELECT
        oi.product_id,
        p.sku,
        p.product_name,
        p.category_name,
        p.brand,
        p.unit_cost,
        p.unit_price AS list_price,

        o.order_id,
        o.order_date,
        o.customer_id,
        oe.customer_country,
        oe.channel,
        oe.customer_segment,

        oi.quantity,
        oi.unit_price AS sold_price,
        oi.discount_percent,
        oi.line_total,

        oi.line_total - (p.unit_cost * oi.quantity) AS gross_margin

    FROM raw_order_items oi
    INNER JOIN raw_products p ON oi.product_id = p.product_id
    INNER JOIN raw_orders o ON oi.order_id = o.order_id
    LEFT JOIN stg_orders_enriched oe ON oi.order_id = oe.order_id
    WHERE o.status NOT IN ('cancelled', 'refunded')
),

product_aggregates AS (
    SELECT
        product_id,
        sku,
        product_name,
        category_name,
        brand,
        unit_cost,
        list_price,

        COUNT(DISTINCT order_id) AS order_count,
        COUNT(DISTINCT customer_id) AS unique_customers,
        SUM(quantity) AS units_sold,

        SUM(line_total) AS total_revenue,
        SUM(gross_margin) AS total_margin,
        AVG(gross_margin / NULLIF(line_total, 0) * 100) AS avg_margin_pct,

        AVG(sold_price) AS avg_selling_price,
        MIN(sold_price) AS min_selling_price,
        MAX(sold_price) AS max_selling_price,
        AVG(discount_percent) AS avg_discount_pct,

        MIN(order_date) AS first_sale_date,
        MAX(order_date) AS last_sale_date,
        COUNT(DISTINCT order_date) AS sales_days,

        COUNT(DISTINCT CASE WHEN customer_segment = 'New' THEN customer_id END) AS new_customer_buyers,
        COUNT(DISTINCT CASE WHEN customer_segment = 'VIP' THEN customer_id END) AS vip_customer_buyers

    FROM product_sales
    GROUP BY
        product_id, sku, product_name, category_name,
        brand, unit_cost, list_price
),

product_rankings AS (
    SELECT
        pa.*,

        RANK() OVER (ORDER BY total_revenue DESC) AS overall_revenue_rank,
        RANK() OVER (PARTITION BY category_name ORDER BY total_revenue DESC) AS category_revenue_rank,
        RANK() OVER (PARTITION BY brand ORDER BY total_revenue DESC) AS brand_revenue_rank,

        RANK() OVER (ORDER BY units_sold DESC) AS overall_units_rank,
        RANK() OVER (PARTITION BY category_name ORDER BY units_sold DESC) AS category_units_rank,

        RANK() OVER (ORDER BY total_margin DESC) AS overall_margin_rank,

        PERCENT_RANK() OVER (PARTITION BY category_name ORDER BY total_revenue) AS category_revenue_percentile

    FROM product_aggregates pa
),

category_totals AS (
    SELECT
        category_name,
        SUM(total_revenue) AS category_total_revenue,
        SUM(units_sold) AS category_total_units
    FROM product_aggregates
    GROUP BY category_name
)

SELECT
    pr.product_id,
    pr.sku,
    pr.product_name,
    pr.category_name,
    pr.brand,

    pr.unit_cost,
    pr.list_price,
    pr.avg_selling_price,
    pr.min_selling_price,
    pr.max_selling_price,
    pr.avg_discount_pct,

    pr.order_count,
    pr.unique_customers,
    pr.units_sold,
    pr.sales_days,

    pr.total_revenue,
    pr.total_margin,
    pr.avg_margin_pct,

    ROUND(pr.total_revenue / ct.category_total_revenue * 100, 2) AS category_revenue_share_pct,
    ROUND(pr.units_sold::FLOAT / ct.category_total_units * 100, 2) AS category_units_share_pct,

    pr.overall_revenue_rank,
    pr.category_revenue_rank,
    pr.brand_revenue_rank,
    pr.overall_units_rank,
    pr.category_units_rank,
    pr.overall_margin_rank,

    ROUND(pr.category_revenue_percentile * 100, 1) AS category_revenue_percentile,

    pr.new_customer_buyers,
    pr.vip_customer_buyers,
    ROUND(pr.vip_customer_buyers::FLOAT / NULLIF(pr.unique_customers, 0) * 100, 2) AS vip_buyer_pct,

    pr.first_sale_date,
    pr.last_sale_date,
    current_date - pr.last_sale_date::DATE AS days_since_last_sale,

    ROUND(pr.units_sold::FLOAT / NULLIF(pr.sales_days, 0), 2) AS daily_velocity,

    CASE
        WHEN current_date - pr.last_sale_date::DATE > 60 THEN 'Slow Moving'
        WHEN pr.category_revenue_rank <= 10 THEN 'Top Performer'
        WHEN pr.category_revenue_percentile >= 0.75 THEN 'Strong'
        WHEN pr.category_revenue_percentile >= 0.5 THEN 'Average'
        ELSE 'Underperformer'
    END AS performance_tier,

    CASE
        WHEN pr.avg_margin_pct >= 40 THEN 'High Margin'
        WHEN pr.avg_margin_pct >= 20 THEN 'Standard Margin'
        WHEN pr.avg_margin_pct >= 0 THEN 'Low Margin'
        ELSE 'Loss Leader'
    END AS margin_tier,

    current_timestamp AS etl_loaded_at

FROM product_rankings pr
LEFT JOIN category_totals ct ON pr.category_name = ct.category_name
