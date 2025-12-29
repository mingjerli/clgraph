-- Staging Products Table
-- Cleaned product data with calculated margins
-- @layer: staging
-- @refresh: daily
-- @owner: data_engineering

CREATE OR REPLACE TABLE staging_{{env}}.products
ENGINE = MergeTree()
ORDER BY (product_id)
AS
SELECT
    -- Primary key
    product_id,

    -- Product info
    -- @description: Product display name
    name,

    -- @description: Product category
    category,

    -- @description: Product subcategory for detailed classification
    subcategory,

    -- Pricing
    -- @description: Product retail price in USD
    price,

    -- @description: Product cost in USD
    cost,

    -- Calculated fields
    -- @description: Gross margin amount (price - cost)
    price - cost AS margin_amount,

    -- @description: Gross margin percentage
    CASE
        WHEN price > 0 THEN round((price - cost) / price * 100, 2)
        ELSE 0
    END AS margin_pct,

    -- Supplier
    supplier_id,

    -- Audit
    now() AS processed_at

FROM raw_{{env}}.products
WHERE product_id != '' AND price > 0;
