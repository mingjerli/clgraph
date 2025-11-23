-- Raw customers from source
CREATE OR REPLACE TABLE raw_customers AS
SELECT
    customer_id,
    email,
    first_name,
    last_name,
    phone_number,
    registration_date,
    country_code,
    city,
    loyalty_tier,
    created_at
FROM source_customers
WHERE registration_date >= '2020-01-01'
