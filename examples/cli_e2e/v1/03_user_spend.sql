-- Mart: user lifetime spend
CREATE OR REPLACE TABLE user_spend AS
SELECT
    u.user_id,
    u.email,
    u.country,
    COUNT(o.order_id) AS total_orders,
    SUM(o.total_amount) AS lifetime_spend,
    MIN(o.order_date) AS first_order_date,
    MAX(o.order_date) AS last_order_date
FROM users u
LEFT JOIN orders o ON u.user_id = o.user_id
GROUP BY u.user_id, u.email, u.country
