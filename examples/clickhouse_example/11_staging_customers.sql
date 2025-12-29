-- Staging Customers Table
-- Cleaned customer data with PII handling
-- @layer: staging
-- @refresh: daily
-- @owner: data_engineering
-- @pii: true

CREATE OR REPLACE TABLE staging_{{env}}.customers
ENGINE = MergeTree()
ORDER BY (customer_id)
AS
SELECT
    -- Primary key
    customer_id,

    -- PII fields (hashed for privacy)
    -- @description: SHA256 hash of customer email for privacy
    -- @pii: true
    SHA256(email) AS email_hash,

    -- @description: Customer display name (first name only for privacy)
    -- @pii: true
    splitByChar(' ', name)[1] AS first_name,

    -- Geographic info
    -- @description: Customer country code
    country,
    city,

    -- Customer tenure
    -- @description: Date when customer first registered
    toDate(created_at) AS customer_since,

    -- @description: Days since customer registration
    dateDiff('day', created_at, now()) AS tenure_days,

    -- Audit
    now() AS processed_at

FROM raw_{{env}}.customers
WHERE customer_id != '';
