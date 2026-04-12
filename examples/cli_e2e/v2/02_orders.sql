-- Raw orders from source
CREATE OR REPLACE TABLE orders AS
SELECT
    order_id,      -- Unique order identifier [owner: data-platform]
    user_id,       -- Reference to user [owner: data-platform]
    order_date,    -- Date order was placed [owner: finance]
    total_amount,  -- Total order amount [owner: finance, tags: metric revenue]
    discount       -- Discount applied [owner: finance, tags: metric]
FROM source_orders
