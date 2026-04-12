-- Raw users from source
CREATE OR REPLACE TABLE users AS
SELECT
    user_id,       -- Unique user identifier [owner: data-platform]
    email,         -- User email [pii: true, owner: data-governance]
    signup_date,   -- When user signed up [owner: growth]
    country,       -- User country [owner: growth]
    tier           -- Loyalty tier [owner: growth]
FROM source_users
