-- Clean analysis table: keep correctly assigned rows, then one row per user
-- (first exposure wins), joined to country.
CREATE OR REPLACE TABLE clean AS
WITH valid AS (
    SELECT *
    FROM ab
    WHERE ("group" = 'control'   AND landing_page = 'old_page')
       OR ("group" = 'treatment' AND landing_page = 'new_page')
),
dedup AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY timestamp) AS rn
    FROM valid
)
SELECT d.user_id, d.timestamp, CAST(d.timestamp AS DATE) AS day,
       d."group", d.converted, c.country
FROM dedup d
LEFT JOIN countries c USING (user_id)
WHERE d.rn = 1;

-- Primary metric by arm
SELECT "group", COUNT(*) AS users, SUM(converted) AS conversions,
       AVG(converted) AS conversion_rate
FROM clean
GROUP BY "group"
ORDER BY "group";

-- Daily conversion by arm (used for the novelty-effect check)
SELECT day, "group", COUNT(*) AS users, AVG(converted) AS conversion_rate
FROM clean
GROUP BY day, "group"
ORDER BY day, "group";

-- Conversion by country and arm (segment analysis)
SELECT country, "group", COUNT(*) AS users, AVG(converted) AS conversion_rate
FROM clean
GROUP BY country, "group"
ORDER BY country, "group";
