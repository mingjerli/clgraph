-- Raw products from source
CREATE OR REPLACE TABLE raw_products AS
SELECT
    product_id,  -- Unique product identifier [owner: data-platform]
    sku,  -- Stock keeping unit code [owner: inventory, tags: product]
    product_name,  -- Product display name [owner: product, tags: product]
    category_name,  -- Product category [owner: product, tags: product category]
    brand,  -- Product brand [owner: product, tags: product]
    unit_cost,  -- Cost to acquire product [owner: finance, tags: metric cost confidential]
    unit_price,  -- Selling price [owner: finance, tags: metric revenue]
    is_active,  -- Whether product is currently active [owner: product, tags: status]
    created_at  -- Record creation timestamp [owner: data-platform]
FROM source_products
WHERE is_active = TRUE
