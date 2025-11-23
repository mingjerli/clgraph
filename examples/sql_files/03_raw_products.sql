-- Raw products from source
CREATE OR REPLACE TABLE raw_products AS
SELECT
    product_id,
    sku,
    product_name,
    category_name,
    brand,
    unit_cost,
    unit_price,
    is_active,
    created_at
FROM source_products
WHERE is_active = TRUE
