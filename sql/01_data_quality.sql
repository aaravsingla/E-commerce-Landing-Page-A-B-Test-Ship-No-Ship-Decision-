-- Data-quality audit of the raw experiment log.
-- Every user should appear once, and group must agree with the page they saw
-- (control -> old_page, treatment -> new_page).

-- 1. Assignment mismatches: rows where group and landing page disagree
SELECT "group", landing_page, COUNT(*) AS n_rows
FROM ab
GROUP BY "group", landing_page
ORDER BY "group", landing_page;

-- 2. Users logged more than once
SELECT COUNT(*) AS duplicated_users
FROM (SELECT user_id FROM ab GROUP BY user_id HAVING COUNT(*) > 1);

-- 3. Experiment window
SELECT MIN(timestamp) AS first_event, MAX(timestamp) AS last_event,
       COUNT(DISTINCT CAST(timestamp AS DATE)) AS n_days
FROM ab;
