-- Analyze whether customers get credit card first then loan or vice versa
-- Assumes a table with customer_id, product_type, and account_open_date

WITH first_products AS (
    -- Get the first credit card and first loan date for each customer
    SELECT
        customer_id,
        MIN(CASE WHEN product_type = 'CREDIT_CARD' THEN account_open_date END) AS first_credit_card_date,
        MIN(CASE WHEN product_type = 'LOAN' THEN account_open_date END) AS first_loan_date
    FROM customer_accounts  -- Replace with your actual table name
    WHERE product_type IN ('CREDIT_CARD', 'LOAN')
    GROUP BY customer_id
),

customers_with_both AS (
    -- Filter to customers who have both products
    SELECT
        customer_id,
        first_credit_card_date,
        first_loan_date,
        CASE
            WHEN first_credit_card_date < first_loan_date THEN 'CREDIT_CARD_FIRST'
            WHEN first_loan_date < first_credit_card_date THEN 'LOAN_FIRST'
            ELSE 'SAME_DAY'
        END AS product_sequence,
        DATEDIFF('day',
            LEAST(first_credit_card_date, first_loan_date),
            GREATEST(first_credit_card_date, first_loan_date)
        ) AS days_between_products
    FROM first_products
    WHERE first_credit_card_date IS NOT NULL
      AND first_loan_date IS NOT NULL
)

-- Summary statistics
SELECT
    product_sequence,
    COUNT(*) AS customer_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS percentage,
    ROUND(AVG(days_between_products), 0) AS avg_days_between,
    MEDIAN(days_between_products) AS median_days_between,
    MIN(days_between_products) AS min_days_between,
    MAX(days_between_products) AS max_days_between
FROM customers_with_both
GROUP BY product_sequence
ORDER BY customer_count DESC;


-- Optional: Detailed customer-level view (uncomment to use)
-- SELECT * FROM customers_with_both ORDER BY product_sequence, days_between_products;
