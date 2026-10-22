"""Analytics queries a generated statement may legitimately be, per dialect.

The SQL policy allows only the constructs it lists; every query here must pass
it. They are the shapes an analytics question produces -- joins, aggregates,
window functions, CTEs, date arithmetic, CASE, JSON access -- written the way
each engine spells them, against one small schema:

    customers(id, name, email, country, created_at, profile)   -- profile is JSON
    products(id, name, category, price)
    orders(id, customer_id, status, total, created_at)
    order_items(order_id, product_id, quantity, unit_price)

PostgreSQL, SQLite and DuckDB are also run, against :data:`FIXTURE_ROWS`, in
the differential tests, so those three sets must be valid on the engine as well
as to the parser.
"""

from __future__ import annotations

POSTGRES = [
    "SELECT c.name, COUNT(o.id) AS orders, SUM(o.total) AS revenue FROM customers c "
    "JOIN orders o ON o.customer_id = c.id GROUP BY c.name ORDER BY revenue DESC LIMIT 10",
    "SELECT date_trunc('month', o.created_at) AS month, SUM(o.total) FROM orders o "
    "WHERE o.created_at >= TIMESTAMP '2025-06-01' - interval '365 days' GROUP BY 1 ORDER BY 1",
    "SELECT EXTRACT(YEAR FROM created_at) AS year, COUNT(*) FROM customers GROUP BY 1",
    "SELECT id, total, ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY created_at) AS n "
    "FROM orders",
    "SELECT customer_id, total, total - LAG(total) OVER (PARTITION BY customer_id "
    "ORDER BY created_at) AS change FROM orders",
    "SELECT customer_id, SUM(total) OVER (PARTITION BY customer_id ORDER BY created_at "
    "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running FROM orders",
    "WITH monthly AS (SELECT customer_id, date_trunc('month', created_at) AS m, SUM(total) AS t "
    "FROM orders GROUP BY 1, 2) SELECT m, AVG(t) FROM monthly GROUP BY m ORDER BY m",
    "SELECT name, CASE WHEN price > 100 THEN 'premium' WHEN price > 20 THEN 'standard' "
    "ELSE 'budget' END AS tier FROM products",
    "SELECT profile->>'tier' AS tier, COUNT(*) FROM customers GROUP BY 1",
    "SELECT name FROM customers WHERE (profile->>'newsletter')::boolean",
    "SELECT status, COUNT(*) FILTER (WHERE total > 50) AS large FROM orders GROUP BY status",
    "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY total) AS median FROM orders",
    "SELECT country, string_agg(name, ', ' ORDER BY name) FROM customers GROUP BY country",
    "SELECT DISTINCT ON (customer_id) customer_id, created_at, total FROM orders "
    "ORDER BY customer_id, created_at DESC",
    "SELECT p.category, SUM(i.quantity * i.unit_price) AS sales FROM order_items i "
    "JOIN products p ON p.id = i.product_id GROUP BY p.category HAVING SUM(i.quantity) > 1",
    "SELECT c.name FROM customers c WHERE NOT EXISTS (SELECT 1 FROM orders o "
    "WHERE o.customer_id = c.id)",
    "SELECT c.name, COALESCE(SUM(o.total), 0) FROM customers c LEFT JOIN orders o "
    "ON o.customer_id = c.id GROUP BY c.name",
    "SELECT name FROM customers WHERE email ILIKE '%@example.com' AND country IN ('DE', 'FR')",
    "SELECT created_at::date AS day, COUNT(*) FROM orders "
    "WHERE created_at BETWEEN '2024-01-01' AND '2024-12-31' GROUP BY 1",
    "SELECT NULLIF(total, 0), ROUND(total / 3, 2), ABS(total), UPPER(status) FROM orders",
    "SELECT age(now(), created_at) FROM customers",
    "SELECT customer_id, RANK() OVER (ORDER BY SUM(total) DESC) FROM orders GROUP BY customer_id",
    "SELECT status FROM orders UNION SELECT 'none'",
    "SELECT o.id, length(c.name), substring(c.email from 1 for 5) FROM orders o "
    "JOIN customers c ON c.id = o.customer_id",
    "SELECT to_char(created_at, 'YYYY-MM') AS month, COUNT(*) FROM orders GROUP BY 1",
    "SELECT CAST(total AS INTEGER), GREATEST(total, 10), LEAST(total, 1000) FROM orders",
]

MYSQL = [
    "SELECT c.name, COUNT(o.id) AS orders, SUM(o.total) AS revenue FROM customers c "
    "JOIN orders o ON o.customer_id = c.id GROUP BY c.name ORDER BY revenue DESC LIMIT 10",
    "SELECT DATE_FORMAT(created_at, '%Y-%m') AS month, SUM(total) FROM orders "
    "WHERE created_at >= DATE_SUB(NOW(), INTERVAL 365 DAY) GROUP BY month ORDER BY month",
    "SELECT YEAR(created_at), MONTH(created_at), COUNT(*) FROM customers GROUP BY 1, 2",
    "SELECT id, ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY created_at) AS n FROM orders",
    "SELECT customer_id, total - LAG(total) OVER (PARTITION BY customer_id ORDER BY created_at) "
    "FROM orders",
    "WITH monthly AS (SELECT customer_id, DATE_FORMAT(created_at, '%Y-%m') AS m, SUM(total) AS t "
    "FROM orders GROUP BY customer_id, m) SELECT m, AVG(t) FROM monthly GROUP BY m",
    "SELECT name, CASE WHEN price > 100 THEN 'premium' ELSE 'standard' END FROM products",
    "SELECT JSON_EXTRACT(profile, '$.tier') AS tier, COUNT(*) FROM customers GROUP BY tier",
    "SELECT profile->>'$.tier' FROM customers",
    "SELECT country, GROUP_CONCAT(name ORDER BY name SEPARATOR ', ') FROM customers "
    "GROUP BY country",
    "SELECT IFNULL(SUM(total), 0), ROUND(AVG(total), 2) FROM orders WHERE status = 'paid'",
    "SELECT TIMESTAMPDIFF(DAY, created_at, NOW()) AS days FROM customers",
    "SELECT p.category, SUM(i.quantity * i.unit_price) FROM order_items i "
    "JOIN products p ON p.id = i.product_id GROUP BY p.category HAVING SUM(i.quantity) > 1",
    "SELECT c.name FROM customers c LEFT JOIN orders o ON o.customer_id = c.id WHERE o.id IS NULL",
    "SELECT CONCAT(name, ' <', email, '>'), UPPER(country), LENGTH(name) FROM customers",
    "SELECT * FROM orders ORDER BY created_at DESC LIMIT 20 OFFSET 40",
    "SELECT STR_TO_DATE('2024-01-31', '%Y-%m-%d'), CURDATE(), DATE(created_at) FROM orders",
    "SELECT status, COUNT(*) FROM orders GROUP BY status WITH ROLLUP",
    "SELECT `name` FROM `customers` WHERE `country` IN ('DE', 'FR')",
    "SELECT IF(total > 50, 'large', 'small'), COUNT(*) FROM orders GROUP BY 1",
]

TSQL = [
    "SELECT TOP 10 c.name, COUNT(o.id) AS orders, SUM(o.total) AS revenue FROM customers c "
    "JOIN orders o ON o.customer_id = c.id GROUP BY c.name ORDER BY revenue DESC",
    "SELECT DATEPART(year, created_at) AS y, DATEPART(month, created_at) AS m, SUM(total) "
    "FROM orders WHERE created_at >= DATEADD(day, -365, GETDATE()) "
    "GROUP BY DATEPART(year, created_at), DATEPART(month, created_at)",
    "SELECT DATEDIFF(day, created_at, GETDATE()) AS days FROM customers",
    "SELECT id, ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY created_at) AS n FROM orders",
    "SELECT customer_id, SUM(total) OVER (PARTITION BY customer_id ORDER BY created_at "
    "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM orders",
    "WITH monthly AS (SELECT customer_id, EOMONTH(created_at) AS m, SUM(total) AS t "
    "FROM orders GROUP BY customer_id, EOMONTH(created_at)) SELECT m, AVG(t) FROM monthly "
    "GROUP BY m",
    "SELECT name, CASE WHEN price > 100 THEN 'premium' ELSE 'standard' END AS tier FROM products",
    "SELECT JSON_VALUE(profile, '$.tier') AS tier, COUNT(*) FROM customers "
    "GROUP BY JSON_VALUE(profile, '$.tier')",
    "SELECT country, STRING_AGG(name, ', ') FROM customers GROUP BY country",
    "SELECT ISNULL(SUM(total), 0), ROUND(AVG(total), 2) FROM orders",
    "SELECT CONVERT(date, created_at) AS day, COUNT(*) FROM orders "
    "GROUP BY CONVERT(date, created_at)",
    "SELECT CAST(created_at AS date), IIF(total > 50, 'large', 'small') FROM orders",
    "SELECT p.category, SUM(i.quantity * i.unit_price) FROM order_items i "
    "JOIN products p ON p.id = i.product_id GROUP BY p.category HAVING SUM(i.quantity) > 1",
    "SELECT c.name FROM customers c WHERE NOT EXISTS (SELECT 1 FROM orders o "
    "WHERE o.customer_id = c.id)",
    "SELECT * FROM orders ORDER BY created_at DESC OFFSET 40 ROWS FETCH NEXT 20 ROWS ONLY",
    "SELECT [name], LEN([name]), UPPER(country) FROM [dbo].[customers]",
    "SELECT customer_id, RANK() OVER (ORDER BY SUM(total) DESC) FROM orders GROUP BY customer_id",
    "SELECT FORMAT(created_at, 'yyyy-MM') AS month, COUNT(*) FROM orders "
    "GROUP BY FORMAT(created_at, 'yyyy-MM')",
]

SNOWFLAKE = [
    "SELECT c.name, COUNT(o.id) AS orders, SUM(o.total) AS revenue FROM customers c "
    "JOIN orders o ON o.customer_id = c.id GROUP BY c.name ORDER BY revenue DESC LIMIT 10",
    "SELECT DATE_TRUNC('month', created_at) AS month, SUM(total) FROM orders "
    "WHERE created_at >= DATEADD(day, -365, CURRENT_TIMESTAMP()) GROUP BY 1 ORDER BY 1",
    "SELECT DATEDIFF('day', created_at, CURRENT_DATE()) FROM customers",
    "SELECT id, ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY created_at) AS n FROM orders",
    "SELECT customer_id, total FROM orders "
    "QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY created_at DESC) = 1",
    "WITH monthly AS (SELECT customer_id, DATE_TRUNC('month', created_at) AS m, SUM(total) AS t "
    "FROM orders GROUP BY 1, 2) SELECT m, AVG(t) FROM monthly GROUP BY m",
    "SELECT name, IFF(price > 100, 'premium', 'standard') FROM products",
    "SELECT profile:tier::string AS tier, COUNT(*) FROM customers GROUP BY 1",
    "SELECT country, LISTAGG(name, ', ') WITHIN GROUP (ORDER BY name) FROM customers "
    "GROUP BY country",
    "SELECT ZEROIFNULL(SUM(total)), ROUND(AVG(total), 2), MEDIAN(total) FROM orders",
    "SELECT TO_CHAR(created_at, 'YYYY-MM'), COUNT(*) FROM orders GROUP BY 1",
    "SELECT p.category, SUM(i.quantity * i.unit_price) FROM order_items i "
    "JOIN products p ON p.id = i.product_id GROUP BY p.category HAVING SUM(i.quantity) > 1",
    "SELECT c.name, f.value::string FROM customers c, LATERAL FLATTEN(input => c.profile:tags) f",
    "SELECT CASE WHEN total > 50 THEN 'large' ELSE 'small' END, COUNT(*) FROM orders GROUP BY 1",
    "SELECT COALESCE(SUM(total), 0), NVL(MAX(total), 0) FROM orders",
    "SELECT name FROM customers WHERE email ILIKE '%@example.com'",
]

BIGQUERY = [
    "SELECT c.name, COUNT(o.id) AS orders, SUM(o.total) AS revenue FROM `shop.customers` c "
    "JOIN `shop.orders` o ON o.customer_id = c.id GROUP BY c.name ORDER BY revenue DESC LIMIT 10",
    "SELECT DATE_TRUNC(DATE(created_at), MONTH) AS month, SUM(total) FROM `shop.orders` "
    "WHERE DATE(created_at) >= DATE_SUB(CURRENT_DATE(), INTERVAL 365 DAY) GROUP BY month",
    "SELECT EXTRACT(YEAR FROM created_at), COUNT(*) FROM `shop.customers` GROUP BY 1",
    "SELECT id, ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY created_at) AS n "
    "FROM `shop.orders`",
    "SELECT customer_id, total FROM `shop.orders` "
    "QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY created_at DESC) = 1",
    "WITH monthly AS (SELECT customer_id, FORMAT_DATE('%Y-%m', DATE(created_at)) AS m, "
    "SUM(total) AS t FROM `shop.orders` GROUP BY 1, 2) SELECT m, AVG(t) FROM monthly GROUP BY m",
    "SELECT name, CASE WHEN price > 100 THEN 'premium' ELSE 'standard' END FROM `shop.products`",
    "SELECT JSON_VALUE(profile, '$.tier') AS tier, COUNT(*) FROM `shop.customers` GROUP BY tier",
    "SELECT JSON_EXTRACT_SCALAR(profile, '$.tier') FROM `shop.customers`",
    "SELECT country, STRING_AGG(name, ', ' ORDER BY name) FROM `shop.customers` GROUP BY country",
    "SELECT SAFE_DIVIDE(SUM(total), COUNT(*)), COUNTIF(total > 50), IFNULL(MAX(total), 0) "
    "FROM `shop.orders`",
    "SELECT c.name, tag FROM `shop.customers` c, UNNEST(JSON_VALUE_ARRAY(c.profile, '$.tags')) "
    "AS tag",
    "SELECT p.category, SUM(i.quantity * i.unit_price) FROM `shop.order_items` i "
    "JOIN `shop.products` p ON p.id = i.product_id GROUP BY p.category",
    "SELECT TIMESTAMP_TRUNC(created_at, WEEK), COUNT(*) FROM `shop.orders` GROUP BY 1",
    "SELECT DATE_DIFF(CURRENT_DATE(), DATE(created_at), DAY) FROM `shop.customers`",
    "SELECT ARRAY_AGG(id ORDER BY created_at DESC LIMIT 3) FROM `shop.orders` GROUP BY customer_id",
]

DUCKDB = [
    "SELECT c.name, COUNT(o.id) AS orders, SUM(o.total) AS revenue FROM customers c "
    "JOIN orders o ON o.customer_id = c.id GROUP BY c.name ORDER BY revenue DESC LIMIT 10",
    "SELECT date_trunc('month', created_at) AS month, SUM(total) FROM orders "
    "WHERE created_at >= TIMESTAMP '2023-01-01' GROUP BY ALL ORDER BY 1",
    "SELECT year(created_at), COUNT(*) FROM customers GROUP BY 1",
    "SELECT id, ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY created_at) AS n FROM orders",
    "SELECT customer_id, total FROM orders "
    "QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY created_at DESC) = 1",
    "SELECT customer_id, total - LAG(total) OVER (PARTITION BY customer_id ORDER BY created_at) "
    "FROM orders",
    "WITH monthly AS (SELECT customer_id, date_trunc('month', created_at) AS m, SUM(total) AS t "
    "FROM orders GROUP BY 1, 2) SELECT m, AVG(t) FROM monthly GROUP BY m ORDER BY m",
    "SELECT name, CASE WHEN price > 100 THEN 'premium' ELSE 'standard' END FROM products",
    "SELECT profile->>'$.tier' AS tier, COUNT(*) FROM customers GROUP BY 1",
    "SELECT json_extract_string(profile, '$.tier') FROM customers",
    "SELECT country, string_agg(name, ', ' ORDER BY name) FROM customers GROUP BY country",
    "SELECT median(total), quantile_cont(total, 0.9), ROUND(AVG(total), 2) FROM orders",
    "SELECT status, COUNT(*) FILTER (WHERE total > 50) FROM orders GROUP BY status",
    "SELECT strftime(created_at, '%Y-%m') AS month, COUNT(*) FROM orders GROUP BY 1",
    "SELECT p.category, SUM(i.quantity * i.unit_price) FROM order_items i "
    "JOIN products p ON p.id = i.product_id GROUP BY p.category HAVING SUM(i.quantity) > 1",
    "SELECT c.name FROM customers c ANTI JOIN orders o ON o.customer_id = c.id",
    "SELECT COALESCE(SUM(total), 0), NULLIF(MAX(total), 0) FROM orders",
    "SELECT date_diff('day', created_at, TIMESTAMP '2025-01-01') FROM customers",
    "SELECT name FROM customers WHERE email ILIKE '%@example.com' AND country IN ('DE', 'FR')",
    "SELECT * FROM orders ORDER BY created_at DESC LIMIT 2 OFFSET 1",
]

SQLITE = [
    "SELECT c.name, COUNT(o.id) AS orders, SUM(o.total) AS revenue FROM customers c "
    "JOIN orders o ON o.customer_id = c.id GROUP BY c.name ORDER BY revenue DESC LIMIT 10",
    "SELECT strftime('%Y-%m', created_at) AS month, SUM(total) FROM orders "
    "WHERE created_at >= date('2025-06-01', '-365 day') GROUP BY month ORDER BY month",
    "SELECT CAST(strftime('%Y', created_at) AS INTEGER) AS year, COUNT(*) FROM customers "
    "GROUP BY year",
    "SELECT id, ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY created_at) AS n FROM orders",
    "SELECT customer_id, total - LAG(total) OVER (PARTITION BY customer_id ORDER BY created_at) "
    "FROM orders",
    "SELECT customer_id, SUM(total) OVER (PARTITION BY customer_id ORDER BY created_at "
    "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM orders",
    "WITH monthly AS (SELECT customer_id, strftime('%Y-%m', created_at) AS m, SUM(total) AS t "
    "FROM orders GROUP BY 1, 2) SELECT m, AVG(t) FROM monthly GROUP BY m ORDER BY m",
    "SELECT name, CASE WHEN price > 100 THEN 'premium' ELSE 'standard' END FROM products",
    "SELECT json_extract(profile, '$.tier') AS tier, COUNT(*) FROM customers GROUP BY tier",
    "SELECT profile->>'$.tier' FROM customers",
    "SELECT country, group_concat(name, ', ') FROM customers GROUP BY country",
    "SELECT IFNULL(SUM(total), 0), ROUND(AVG(total), 2) FROM orders",
    "SELECT julianday('2025-06-01') - julianday(created_at) AS days FROM customers",
    "SELECT p.category, SUM(i.quantity * i.unit_price) FROM order_items i "
    "JOIN products p ON p.id = i.product_id GROUP BY p.category HAVING SUM(i.quantity) > 1",
    "SELECT c.name FROM customers c WHERE NOT EXISTS (SELECT 1 FROM orders o "
    "WHERE o.customer_id = c.id)",
    "SELECT substr(email, 1, 5), length(name), upper(country), lower(status) FROM customers c "
    "JOIN orders o ON o.customer_id = c.id",
    "SELECT * FROM orders ORDER BY created_at DESC LIMIT 2 OFFSET 1",
    "SELECT COALESCE(MAX(total), 0), NULLIF(MIN(total), 0), ABS(-total) FROM orders",
    "SELECT name FROM customers WHERE email LIKE '%@example.com' AND country IN ('DE', 'FR')",
    "SELECT status, COUNT(*) FROM orders GROUP BY status UNION ALL SELECT 'total', COUNT(*) "
    "FROM orders",
]

REDSHIFT = [
    "SELECT c.name, COUNT(o.id) AS orders, SUM(o.total) AS revenue FROM customers c "
    "JOIN orders o ON o.customer_id = c.id GROUP BY c.name ORDER BY revenue DESC LIMIT 10",
    "SELECT DATE_TRUNC('month', created_at) AS month, SUM(total) FROM orders "
    "WHERE created_at >= DATEADD(day, -365, GETDATE()) GROUP BY 1 ORDER BY 1",
    "SELECT DATEDIFF(day, created_at, GETDATE()) FROM customers",
    "SELECT id, ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY created_at) AS n FROM orders",
    "SELECT customer_id, SUM(total) OVER (PARTITION BY customer_id ORDER BY created_at "
    "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM orders",
    "WITH monthly AS (SELECT customer_id, DATE_TRUNC('month', created_at) AS m, SUM(total) AS t "
    "FROM orders GROUP BY 1, 2) SELECT m, AVG(t) FROM monthly GROUP BY m",
    "SELECT name, CASE WHEN price > 100 THEN 'premium' ELSE 'standard' END FROM products",
    "SELECT JSON_EXTRACT_PATH_TEXT(profile, 'tier') AS tier, COUNT(*) FROM customers GROUP BY 1",
    "SELECT country, LISTAGG(name, ', ') WITHIN GROUP (ORDER BY name) FROM customers "
    "GROUP BY country",
    "SELECT NVL(SUM(total), 0), ROUND(AVG(total), 2), MEDIAN(total) FROM orders",
    "SELECT TO_CHAR(created_at, 'YYYY-MM'), COUNT(*) FROM orders GROUP BY 1",
    "SELECT p.category, SUM(i.quantity * i.unit_price) FROM order_items i "
    "JOIN products p ON p.id = i.product_id GROUP BY p.category HAVING SUM(i.quantity) > 1",
    "SELECT c.name FROM customers c LEFT JOIN orders o ON o.customer_id = c.id WHERE o.id IS NULL",
    "SELECT age(created_at) FROM customers",
    "SELECT customer_id, RANK() OVER (ORDER BY SUM(total) DESC) FROM orders GROUP BY customer_id",
    "SELECT name FROM customers WHERE email ILIKE '%@example.com'",
]

CORPUS: dict[str, list[str]] = {
    "postgres": POSTGRES,
    "mysql": MYSQL,
    "tsql": TSQL,
    "snowflake": SNOWFLAKE,
    "bigquery": BIGQUERY,
    "duckdb": DUCKDB,
    "sqlite": SQLITE,
    "redshift": REDSHIFT,
}

#: Rows for the differential tests: enough that every corpus query on an engine
#: that runs it returns something, and that orders differ in time and amount.
FIXTURE_ROWS: dict[str, list[tuple[object, ...]]] = {
    "customers": [
        (
            1,
            "Ada",
            "ada@example.com",
            "DE",
            "2023-02-01 09:00:00",
            '{"tier": "gold", "newsletter": true, "tags": ["b2b"]}',
        ),
        (
            2,
            "Grace",
            "grace@example.org",
            "FR",
            "2024-05-12 14:30:00",
            '{"tier": "silver", "newsletter": false, "tags": []}',
        ),
        (
            3,
            "Linus",
            "linus@example.com",
            "FI",
            "2024-11-30 08:15:00",
            '{"tier": "gold", "newsletter": true, "tags": ["vip"]}',
        ),
    ],
    "products": [
        (1, "Desk", "furniture", 250.0),
        (2, "Lamp", "lighting", 45.5),
        (3, "Pen", "stationery", 2.25),
    ],
    "orders": [
        (1, 1, "paid", 120.0, "2024-03-01 10:00:00"),
        (2, 1, "paid", 80.5, "2024-07-15 16:20:00"),
        (3, 2, "refunded", 45.5, "2024-08-02 11:05:00"),
        (4, 1, "paid", 300.0, "2025-01-10 09:45:00"),
    ],
    "order_items": [
        (1, 1, 1, 120.0),
        (2, 2, 1, 45.5),
        (2, 3, 4, 2.25),
        (3, 2, 1, 45.5),
        (4, 1, 1, 250.0),
        (4, 3, 2, 25.0),
    ],
}

FIXTURE_COLUMNS: dict[str, list[tuple[str, str]]] = {
    "customers": [
        ("id", "INTEGER"),
        ("name", "TEXT"),
        ("email", "TEXT"),
        ("country", "TEXT"),
        ("created_at", "TIMESTAMP"),
        ("profile", "JSON"),
    ],
    "products": [("id", "INTEGER"), ("name", "TEXT"), ("category", "TEXT"), ("price", "NUMERIC")],
    "orders": [
        ("id", "INTEGER"),
        ("customer_id", "INTEGER"),
        ("status", "TEXT"),
        ("total", "NUMERIC"),
        ("created_at", "TIMESTAMP"),
    ],
    "order_items": [
        ("order_id", "INTEGER"),
        ("product_id", "INTEGER"),
        ("quantity", "INTEGER"),
        ("unit_price", "NUMERIC"),
    ],
}


def policy_tables(sql: str, dialect: str) -> set[str]:
    """The tables the SQL policy sees *sql* read, by name, less its CTEs."""
    import sqlglot  # noqa: PLC0415
    from sqlglot import exp  # noqa: PLC0415

    tree = sqlglot.parse_one(sql, read=dialect)
    ctes = {cte.alias_or_name for cte in tree.find_all(exp.CTE)}
    return {table.name for table in tree.find_all(exp.Table)} - ctes
