-- Lead Conversion Analysis | table: leads (UCI Bank Marketing, cleaned by analysis.py)
-- 1. Overall funnel
SELECT COUNT(*) AS leads, SUM(converted) AS conversions, ROUND(100.0*AVG(converted),2) AS conv_rate_pct FROM leads;

-- 2. Conversion by previous campaign outcome (warm vs cold leads)
SELECT poutcome, COUNT(*) AS leads, SUM(converted) AS conversions,
       ROUND(100.0*AVG(converted),2) AS conv_rate_pct
FROM leads GROUP BY poutcome ORDER BY conv_rate_pct DESC;

-- 3. Channel performance: cellular vs telephone
SELECT contact, COUNT(*) AS leads, ROUND(100.0*AVG(converted),2) AS conv_rate_pct
FROM leads GROUP BY contact ORDER BY conv_rate_pct DESC;

-- 4. Segments worth prioritising (min 100 leads to avoid noise)
SELECT job, COUNT(*) AS leads, ROUND(100.0*AVG(converted),2) AS conv_rate_pct
FROM leads GROUP BY job HAVING COUNT(*) >= 100 ORDER BY conv_rate_pct DESC;

-- 5. Effort vs return: are repeated call attempts worth it?
SELECT CASE WHEN campaign >= 4 THEN '4+ attempts' ELSE '1-3 attempts' END AS attempts_group,
       SUM(campaign) AS total_calls, SUM(converted) AS conversions,
       ROUND(1000.0*SUM(converted)/SUM(campaign),1) AS conversions_per_1000_calls
FROM leads GROUP BY 1;

-- 6. Monthly seasonality (volume vs conversion)
SELECT month, COUNT(*) AS leads, ROUND(100.0*AVG(converted),2) AS conv_rate_pct
FROM leads GROUP BY month;
