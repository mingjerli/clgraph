-- Schema initialization for ClickHouse
-- Creates databases for dev and prod environments

-- Create dev environment databases
CREATE DATABASE IF NOT EXISTS raw_dev;
CREATE DATABASE IF NOT EXISTS staging_dev;
CREATE DATABASE IF NOT EXISTS analytics_dev;
CREATE DATABASE IF NOT EXISTS marts_dev;

-- Create prod environment databases
CREATE DATABASE IF NOT EXISTS raw_prod;
CREATE DATABASE IF NOT EXISTS staging_prod;
CREATE DATABASE IF NOT EXISTS analytics_prod;
CREATE DATABASE IF NOT EXISTS marts_prod;
