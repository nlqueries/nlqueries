-- Synthetic fixtures for the benchmark stack (benchmarks/docker-compose.yaml).
--
-- Mounted into Postgres's docker-entrypoint-initdb.d, so it runs once, when the
-- postgres_bench_data volume is first created. To reseed, remove the volume:
--   docker compose -f benchmarks/docker-compose.yaml down -v
--
-- Small on purpose: enough rows for joins, aggregates and filters to return
-- something to check, not a load test. Every value is made up.

CREATE TABLE customers (
    id          integer PRIMARY KEY,
    name        text    NOT NULL,
    region      text    NOT NULL,
    signed_up   date    NOT NULL
);

CREATE TABLE products (
    id          integer PRIMARY KEY,
    name        text    NOT NULL,
    category    text    NOT NULL,
    price       numeric(10, 2) NOT NULL
);

CREATE TABLE orders (
    id          integer PRIMARY KEY,
    customer_id integer NOT NULL REFERENCES customers (id),
    ordered_at  date    NOT NULL,
    status      text    NOT NULL
);

CREATE TABLE order_items (
    order_id    integer NOT NULL REFERENCES orders (id),
    product_id  integer NOT NULL REFERENCES products (id),
    quantity    integer NOT NULL,
    PRIMARY KEY (order_id, product_id)
);

INSERT INTO customers (id, name, region, signed_up) VALUES
    (1, 'Alder Supply',     'East',  '2025-01-14'),
    (2, 'Birch & Co',       'West',  '2025-02-03'),
    (3, 'Cedar Works',      'East',  '2025-03-21'),
    (4, 'Dogwood Traders',  'North', '2025-05-09'),
    (5, 'Elm Street Goods', 'West',  '2025-07-30');

INSERT INTO products (id, name, category, price) VALUES
    (1, 'Widget',      'Hardware', 4.50),
    (2, 'Gadget',      'Hardware', 12.00),
    (3, 'Manual',      'Books',    18.25),
    (4, 'Support pack','Services', 99.00);

INSERT INTO orders (id, customer_id, ordered_at, status) VALUES
    (101, 1, '2026-01-05', 'shipped'),
    (102, 2, '2026-01-17', 'shipped'),
    (103, 1, '2026-02-02', 'cancelled'),
    (104, 3, '2026-02-20', 'shipped'),
    (105, 4, '2026-03-11', 'pending'),
    (106, 5, '2026-03-29', 'shipped'),
    (107, 2, '2026-04-08', 'shipped');

INSERT INTO order_items (order_id, product_id, quantity) VALUES
    (101, 1, 10), (101, 3, 1),
    (102, 2, 4),
    (103, 1, 2),
    (104, 2, 1), (104, 4, 1),
    (105, 3, 2),
    (106, 1, 25), (106, 2, 5),
    (107, 4, 2);

ANALYZE;
