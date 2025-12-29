-- Marts: Sales Dashboard
-- Executive summary view for sales reporting
-- @layer: marts
-- @refresh: daily
-- @owner: business_intelligence

CREATE OR REPLACE TABLE marts_{{env}}.sales_dashboard
ENGINE = MergeTree()
ORDER BY tuple()
AS
SELECT
    -- Report date
    d.sale_date AS report_date,
    d.week_start,
    d.month_start,

    -- Daily metrics
    -- @description: Number of orders for the day
    d.order_count AS daily_orders,

    -- @description: Number of unique customers for the day
    d.unique_customers AS daily_customers,

    -- @description: Total revenue for the day
    d.total_revenue AS daily_revenue,

    -- @description: Average order value for the day
    d.avg_order_value AS daily_aov,

    -- @description: Order completion rate
    d.completion_rate,

    -- Running totals (MTD)
    -- @description: Month-to-date total revenue
    SUM(d.total_revenue) OVER (
        PARTITION BY d.month_start
        ORDER BY d.sale_date
    ) AS mtd_revenue,

    -- @description: Month-to-date order count
    SUM(d.order_count) OVER (
        PARTITION BY d.month_start
        ORDER BY d.sale_date
    ) AS mtd_orders,

    -- @description: Month-to-date unique customers
    SUM(d.unique_customers) OVER (
        PARTITION BY d.month_start
        ORDER BY d.sale_date
    ) AS mtd_customers,

    -- Week-over-week comparison
    -- @description: Previous week same day revenue
    lagInFrame(d.total_revenue, 7) OVER (ORDER BY d.sale_date) AS prev_week_revenue,

    -- @description: Week-over-week revenue growth percentage
    CASE
        WHEN lagInFrame(d.total_revenue, 7) OVER (ORDER BY d.sale_date) > 0
        THEN round((d.total_revenue - lagInFrame(d.total_revenue, 7) OVER (ORDER BY d.sale_date))
            / lagInFrame(d.total_revenue, 7) OVER (ORDER BY d.sale_date) * 100, 2)
        ELSE 0
    END AS wow_revenue_growth,

    -- Day of week patterns
    d.day_of_week,

    -- @description: Day name for display
    CASE d.day_of_week
        WHEN 1 THEN 'Monday'
        WHEN 2 THEN 'Tuesday'
        WHEN 3 THEN 'Wednesday'
        WHEN 4 THEN 'Thursday'
        WHEN 5 THEN 'Friday'
        WHEN 6 THEN 'Saturday'
        WHEN 7 THEN 'Sunday'
    END AS day_name,

    -- @description: Weekend flag
    CASE WHEN d.day_of_week IN (6, 7) THEN 1 ELSE 0 END AS is_weekend,

    -- Performance indicators
    -- @description: Daily performance vs monthly average
    CASE
        WHEN AVG(d.total_revenue) OVER (PARTITION BY d.month_start) > 0
        THEN round(d.total_revenue / AVG(d.total_revenue) OVER (PARTITION BY d.month_start) * 100, 2)
        ELSE 0
    END AS performance_vs_monthly_avg,

    -- Audit
    now() AS generated_at

FROM analytics_{{env}}.daily_sales d;
