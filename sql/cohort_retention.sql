-- Monthly cohort retention for CDNOW customers.
-- A customer belongs to the cohort of the month of their first purchase.
-- Retention in month k = share of the cohort that purchased again in the
-- k-th month after their first purchase month.

-- Input: data/processed/cdnow_transactions.csv, written by
-- prodanalytics.data.load_cdnow_transactions().

WITH tx AS (
    SELECT customer_id, CAST(date AS DATE) AS purchase_date
    FROM read_csv_auto('data/processed/cdnow_transactions.csv', types = {'customer_id': 'VARCHAR'})
),
firsts AS (
    SELECT customer_id, date_trunc('month', MIN(purchase_date)) AS cohort_month
    FROM tx
    GROUP BY customer_id
),
activity AS (
    SELECT DISTINCT
        t.customer_id,
        f.cohort_month,
        date_diff('month', f.cohort_month, date_trunc('month', t.purchase_date)) AS months_since_first
    FROM tx t
    JOIN firsts f USING (customer_id)
),
cohort_size AS (
    SELECT cohort_month, COUNT(*) AS customers FROM firsts GROUP BY cohort_month
)
SELECT
    a.cohort_month,
    c.customers                                              AS cohort_size,
    a.months_since_first,
    COUNT(DISTINCT a.customer_id)                            AS active_customers,
    ROUND(COUNT(DISTINCT a.customer_id) / c.customers, 4)    AS retention
FROM activity a
JOIN cohort_size c USING (cohort_month)
GROUP BY a.cohort_month, c.customers, a.months_since_first
ORDER BY a.cohort_month, a.months_since_first;
