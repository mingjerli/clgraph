-- Raw customers from source
CREATE OR REPLACE TABLE raw_customers AS
SELECT
    customer_id,  -- Unique customer identifier [owner: data-platform]
    email,  -- Customer email address [pii: true, owner: data-governance, tags: contact]
    first_name,  -- Customer first name [pii: true, owner: data-governance]
    last_name,  -- Customer last name [pii: true, owner: data-governance]
    phone_number,  -- Customer phone number [pii: true, owner: data-governance, tags: contact]
    registration_date,  -- Date customer registered [owner: marketing]
    country_code,  -- Customer country code [owner: marketing, tags: geo]
    city,  -- Customer city [owner: marketing, tags: geo]
    loyalty_tier,  -- Customer loyalty program tier [owner: marketing, tags: loyalty]
    created_at  -- Record creation timestamp [owner: data-platform]
FROM source_customers
WHERE registration_date >= '2020-01-01'
