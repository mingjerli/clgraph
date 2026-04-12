-- Mart: user lifetime spend (v2 — adds tier and net spend)
CREATE OR REPLACE TABLE user_spend AS
SELECT
    u.user_id,
    u.email,
    u.country,
    u.tier,
    COUNT(o.order_id) AS total_orders,
    SUM(o.total_amount) AS lifetime_spend,
    SUM(o.total_amount - o.discount) AS lifetime_net_spend,
    MIN(o.order_date) AS first_order_date,
    MAX(o.order_date) AS last_order_date
FROM users u
LEFT JOIN orders o ON u.user_id = o.user_id
GROUP BY u.user_id, u.email, u.country, u.tier
