-- Headline metrics for each arm of the Cookie Cats experiment.
-- Players were randomly assigned to have the first progress gate at level 30
-- (control) or level 40 (treatment).
--
-- Engagement (game rounds in the first 14 days) is heavily skewed, so it is
-- reported as a median and as a mean capped at the 99.9th percentile to stop
-- a single extreme player dominating the comparison.

WITH players AS (
    SELECT * FROM read_csv_auto('data/raw/cookie_cats.csv')
),
cap AS (
    SELECT quantile_cont(sum_gamerounds, 0.999) AS p999 FROM players
)
SELECT
    version                                                   AS arm,
    COUNT(*)                                                  AS players,
    CAST(SUM(CASE WHEN retention_1 THEN 1 ELSE 0 END) AS INTEGER) AS retained_day1,
    CAST(SUM(CASE WHEN retention_7 THEN 1 ELSE 0 END) AS INTEGER) AS retained_day7,
    ROUND(AVG(CASE WHEN retention_1 THEN 1.0 ELSE 0 END), 4)  AS day1_retention,
    ROUND(AVG(CASE WHEN retention_7 THEN 1.0 ELSE 0 END), 4)  AS day7_retention,
    MEDIAN(sum_gamerounds)                                    AS median_rounds,
    ROUND(AVG(LEAST(sum_gamerounds, cap.p999)), 2)            AS capped_mean_rounds,
    MAX(sum_gamerounds)                                       AS max_rounds
FROM players, cap
GROUP BY version
ORDER BY version;
