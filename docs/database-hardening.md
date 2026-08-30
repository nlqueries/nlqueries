# Hardening the database NLQueries reads

NLQueries turns a natural-language question into SQL with a language model, and
then runs that SQL against your database. The application checks the statement
first, but a validator is a program with bugs in it, and the model's output is
not something anyone controls. The only boundary that does not depend on either
of those being correct is the database's own permission system.

So: **give NLQueries a login that cannot write, and cannot reach anything you
would not put in an answer.** Everything below is what that means concretely for
each supported engine.

---

## What NLQueries already does, and what it does not

The connector opens every query in a read-only transaction (`SET TRANSACTION
READ ONLY`). PostgreSQL applies this to what a statement *does* rather than to
how it is written, so it refuses DML and DDL anywhere in the call graph,
including within a function invoked by a `SELECT`, and refuses sequence
functions by name.

It is not sufficient, and it is important to know exactly where it stops. A
read-only transaction still permits:

| Still allowed in a read-only transaction | Why it matters |
|---|---|
| `pg_advisory_lock(...)` | holds a lock other sessions wait on |
| `pg_sleep(...)` | ties up a connection; bounded here by `statement_timeout`, not by the transaction |
| `pg_read_file(...)`, `pg_ls_dir(...)` | reads files off the database host — if the role has the privilege |
| `SELECT` on any table the role can reach | read-only is not the same as least-privilege |

Every row of that table is a privilege question, and privileges are granted by
you. The rest of this page is how.

SQLite and DuckDB are the exceptions to "the rest is yours to configure". Their
file access is reachable from any `SELECT` rather than granted by a DBA, so the
connectors close it in code. See [SQLite](#sqlite) and [DuckDB](#duckdb) below.

---

## PostgreSQL

### 1. A role that owns nothing and cannot write

```sql
-- A group that carries the read-only grants, and a login that inherits them.
-- Separating the two means you can rotate the login without re-granting.
CREATE ROLE nlqueries_readonly;
CREATE ROLE nlqueries LOGIN PASSWORD 'use-a-generated-secret' IN ROLE nlqueries_readonly;

ALTER ROLE nlqueries
    NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
```

`NOBYPASSRLS` matters if you use row-level security: a role with `BYPASSRLS`
reads every row regardless of policy. `NOREPLICATION` keeps it away from the
write-ahead log, which contains the writes you just forbade.

### 2. Connect to one database, and create nothing

```sql
REVOKE ALL ON DATABASE mydb FROM PUBLIC;
GRANT CONNECT ON DATABASE mydb TO nlqueries_readonly;

-- No temporary tables: they are writes, and they are a way to stage data.
REVOKE TEMPORARY ON DATABASE mydb FROM nlqueries_readonly, PUBLIC;

-- No objects of its own, anywhere.
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
```

That last line is not about NLQueries. Any role that can create objects in a
schema on the `search_path` can shadow a function or operator that another
query resolves to — so leaving `PUBLIC` able to create there undoes work you do
elsewhere.

### 3. Read exactly what you meant to expose

```sql
GRANT USAGE ON SCHEMA analytics TO nlqueries_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA analytics TO nlqueries_readonly;

-- Tables created tomorrow, too. Without this the grant silently rots.
ALTER DEFAULT PRIVILEGES IN SCHEMA analytics
    GRANT SELECT ON TABLES TO nlqueries_readonly;
```

Grant per schema, not per database. If a schema holds something that should
never appear in an answer, the cheapest control is not granting `USAGE` on it.

### 4. Sequences, functions, and the file-reading roles

```sql
-- Sequences: read-only transactions already refuse nextval(), but the grant
-- should not be there either.
REVOKE ALL ON ALL SEQUENCES IN SCHEMA analytics FROM nlqueries_readonly, PUBLIC;

-- Functions default to EXECUTE for PUBLIC. That includes any function a
-- colleague adds later, whatever it does.
REVOKE EXECUTE ON ALL FUNCTIONS IN SCHEMA analytics FROM PUBLIC;
ALTER DEFAULT PRIVILEGES IN SCHEMA analytics
    REVOKE EXECUTE ON FUNCTIONS FROM PUBLIC;
```

Then check the predefined roles, which are the ones that turn a `SELECT` into a
file read:

```sql
-- Should return no rows.
SELECT r.rolname
FROM pg_auth_members m
JOIN pg_roles r ON r.oid = m.roleid
JOIN pg_roles g ON g.oid = m.member
WHERE g.rolname = 'nlqueries'
  AND r.rolname IN (
      'pg_read_server_files',
      'pg_write_server_files',
      'pg_execute_server_program',
      'pg_read_all_data'
  );
```

`pg_read_all_data` is the one people are surprised by: it is read-only, sounds
harmless, and defeats every per-schema grant above.

### 5. A search path that cannot be shadowed, and budgets

```sql
ALTER ROLE nlqueries SET search_path = pg_catalog, analytics;
ALTER ROLE nlqueries SET statement_timeout = '30s';
ALTER ROLE nlqueries SET idle_in_transaction_session_timeout = '60s';
ALTER ROLE nlqueries SET default_transaction_read_only = on;
```

No `$user` and no writable schema in the path. `default_transaction_read_only`
belongs here as well as in the application: it is the same control, held one
layer down, where an application bug cannot skip it.

### 6. Check what you built

```sql
-- Connect as nlqueries, then:
SELECT current_user, current_setting('transaction_read_only');   -- expect: on
SELECT rolsuper, rolcreatedb, rolcreaterole, rolbypassrls
FROM pg_roles WHERE rolname = current_user;                      -- expect: all false
CREATE TABLE should_fail (x int);                                -- expect: error
```

If the third line succeeds, nothing above is in effect.

**NLQueries also checks this for you.** Each time the pool opens a connection,
the connector reads the login's privileges and records what it finds. The result
appears in `nlqueries health`:

```
[OK]   Database (shop) identity: nlqueries@shop: least privilege
[WARN] Database (shop) identity: postgres@shop: connects as a superuser, which
       bypasses every privilege check — see docs/database-hardening.md
[FAIL] Database (shop) identity: identity could not be determined: <reason>
```

The check reports; it does not refuse a connection. The privileges are yours to
grant, and a deployment that is working should not stop working because the
application disagrees with a decision you made.

An identity that could not be read is a failure rather than a warning. The check
exists to answer the question, and an unknown answer recorded as "fine" is the
outcome it is meant to remove.

What it looks for: `SUPERUSER`, `BYPASSRLS`, `CREATEDB`, `CREATEROLE`,
`REPLICATION`, and membership of `pg_read_server_files`,
`pg_write_server_files`, `pg_execute_server_program`, `pg_read_all_data` or
`pg_write_all_data`. It also reports `search_path` and whether the role sets
`default_transaction_read_only`, so a path or a default you did not intend is
visible.

---

## Transport security

The connector refuses a server that offers no TLS. What it verifies beyond that
depends on `ssl_mode` and whether `ssl_ca_cert` is configured.

Measured against PostgreSQL 16 and libpq, using servers presenting a correct
certificate, one naming a different host, one signed by an untrusted CA, and an
expired one:

| `ssl_mode` | untrusted CA | expired | wrong hostname |
|---|---|---|---|
| `require`, no `ssl_ca_cert` | accepted | accepted | accepted |
| `require` + `ssl_ca_cert` | refused | refused | accepted |
| `verify-ca` + `ssl_ca_cert` | refused | refused | accepted |
| `verify-full` + `ssl_ca_cert` | refused | refused | refused |

`require` without a root certificate performs no verification. It encrypts the
connection to whichever server answers, which does not exclude an attacker
positioned between NLQueries and the database.

`ssl_ca_cert` is not ignored under `require`: libpq documents that `require`
with a valid root certificate behaves as `verify-ca`, and the measurements
confirm it. The remaining difference is hostname verification.

**Configure `ssl_ca_cert`.** When you do and set no explicit `ssl_mode`, the
connector uses `verify-full`, which is the only setting in the table that
refuses all three. Set `ssl_mode` explicitly only to select something weaker,
and `nlqueries health` will report what that leaves unverified:

```
[OK]   Database (shop) transport: ssl_mode 'verify-full': chain and hostname verified
[WARN] Database (shop) transport: ssl_mode is 'require' with no ssl_ca_cert, so the
       server's certificate is not verified …
```

A certificate whose subject does not name the host you connect to will be
refused under `verify-full`. Connecting by IP address to a certificate issued
for a hostname is the usual cause.

### The generic connector

Everything above describes the per-vendor connectors, which take `ssl_mode` and
`ssl_ca_cert` as configuration. The generic `sqlalchemy` connector takes a whole
SQLAlchemy URL instead, and its posture comes from that URL.

It applies `ssl_mode`, `ssl_ca_cert`, `ssl_client_cert` and `ssl_client_key` when
the URL names a libpq-based PostgreSQL driver — `postgresql://` or
`postgresql+psycopg://` — because those are the drivers whose parameter names
map onto these settings directly.

For any other driver it **refuses to connect** rather than ignoring them:

```
SQLAlchemyConnector cannot apply ['ssl_ca_cert'] to a 'mysql+pymysql' URL:
the TLS parameter names are driver-specific and only the libpq-based
PostgreSQL drivers are mapped. Put the equivalent settings in the URL's
query string instead.
```

That is deliberate. Accepting a setting and then not applying it leaves you with
a plaintext session you believe is verified, and no way to tell — which is the
failure the `require` default elsewhere in this document exists to prevent. Put
the driver's own parameters in the URL query string instead, for example
`mysql+pymysql://…/shop?ssl_ca=/etc/ssl/ca.pem`.

Where it does apply them it resolves the mode the same way the per-vendor
connectors do, so `ssl_ca_cert` with no `ssl_mode` selects `verify-full` rather
than leaving you on libpq's `prefer`.

**Set each parameter in one place.** SQLAlchemy lets `connect_args` overrule the
URL's own query parameters, so a URL saying `?sslmode=verify-full` beside a
connector setting of `ssl_mode: require` would connect as `require` with the
stricter setting discarded and nothing said. Rather than pick a winner, the
connector refuses:

```
SQLAlchemyConnector will not silently overrule the URL: both it and this
connector's TLS credentials set ['sslmode'], and SQLAlchemy would let the
credentials win without saying so. Configure each setting in one place —
either the URL's query string or the ssl_* credentials.
```

Splitting them across the two is fine, because nothing is then overruled:
`?sslrootcert=/etc/ssl/ca.pem` in the URL alongside `ssl_mode: verify-full` in
the connector's settings connects as you would expect.

`nlqueries health` reports the posture for a `sqlalchemy` connector whenever the
connector's own `ssl_*` settings decided it, exactly as for the per-vendor ones.
Where the URL alone decides, it reports nothing rather than guessing — the
posture is in the URL, and this connector never saw it.

A `sqlalchemy` URL with no TLS settings configured alongside it is untouched:
the URL alone decides, including its defaults. For PostgreSQL that default is
libpq's `prefer`, which falls back to plaintext silently — so prefer the
`postgres` connector, which defaults to `require`, unless you have a reason not
to.

---

## Row-level security

If tenants share tables, RLS is the mechanism. Two caveats apply.

A table's **owner** bypasses its own policies unless the table is set to `FORCE
ROW LEVEL SECURITY`, so the NLQueries role must not own the tables it reads.
And a policy keyed on a session variable that submitted SQL can itself change is
not a boundary. Prefer a per-tenant login, or set the value in a connection hook
the query cannot reach, and reset pooled connections on return.

---

## SQLite

**The connector opens the database read-only and installs an authorizer.
Neither can be disabled through configuration.** Read-only alone is not
sufficient: measured on SQLite 3.50.4, a connection opened `mode=ro` will still
`ATTACH` another database file and read its contents. Read-only restricts what
may be written, not which files may be opened.

| | before | after |
|---|---|---|
| `INSERT`, `UPDATE`, `CREATE`, `DROP` | wrote to your database | refused (`mode=ro`) |
| `ATTACH '/some/other.db'` | **opened it and read its rows** | refused (authorizer) |
| `PRAGMA writable_schema`, `temp_store`, `database_list` | allowed | refused (allow-list) |
| `PRAGMA table_info`, `foreign_key_list` | allowed | allowed — the schema needs them |
| `load_extension()` | already refused by Python's `sqlite3` | refused |
| ordinary `SELECT` | worked | works |

Pragmas are allow-listed rather than deny-listed, because those requiring
restriction are not confined to an obvious set.

`mode=ro` is supplied as a URI parameter, so the database path requires care: a
`database` credential ending `?mode=rwc&` would turn a URI built by string
concatenation into one carrying two `mode` parameters, of which SQLite applies
the first. The path is percent-encoded, so such a value remains part of the
filename.

A path that does not exist is refused rather than created, for the same reason
as DuckDB: an empty database answers every question with "no tables", which is a
misconfiguration reported as a success.

Still worth doing yourself:

- make the file and its directory read-only at the filesystem level, because a
  read-only handle is an application-level promise and file permissions are not.

`:memory:` is not a substitute for a missing file: it succeeds, returns answers
about no data, and presents as working. It is also the one case not opened
read-only, since SQLite does not permit it, but the authorizer still applies
because `ATTACH` reaches the filesystem from an in-memory database.

---

## DuckDB

DuckDB reads the local filesystem through table functions — `read_csv_auto()`,
`read_text()`, `glob()` — so a file read appears within an ordinary `SELECT`,
and the database file does not bound what a query may reach. Opening the
database read-only does not restrict them.

**The connector closes this, and no setting reopens it.** Every DuckDB
connection is made with external access disabled, extension autoinstall and
autoload disabled, and the configuration locked so that no subsequent `SET` can
alter it. A database on disk is opened read-only; `:memory:` cannot be, as
DuckDB does not permit it, and does not require it.

Measured against DuckDB 1.5.5, before and after:

| | before | after |
|---|---|---|
| `read_csv_auto()`, `read_text()`, `read_blob()` | read any file | refused |
| `glob()` | listed any directory | refused |
| `ATTACH '/some/other.duckdb'` | opened it | refused |
| `COPY (...) TO '/path'` | **wrote a file** | refused |
| `INSTALL httpfs` | reached the network | refused |
| `CREATE TABLE ...` | wrote to the database | refused (read-only) |
| ordinary `SELECT` against your tables | worked | works |

Two consequences to note before upgrading:

- **A database whose tables are views over external parquet or CSV files will
  no longer function**, since reading those files is what is being refused.
  Materialise that data into the DuckDB file. This is a deliberate trade-off.
- **A path that does not exist is refused rather than created.** DuckDB would
  otherwise create an empty database, so a misconfiguration would present as
  success.

Running DuckDB in an environment with nothing else to read — a container with
only the database file mounted, no host secrets and no egress — remains
worthwhile. The sandbox above is defence in depth and does not replace it.

---

## What each connector enforces

The read-only transaction described above is a PostgreSQL mechanism. It is not
available on every engine, and several connectors apply nothing equivalent. The
table records what the connector does, not what the engine could support.

| dialect | read-only mechanism | statement timeout | verified in this repository |
|---|---|---|---|
| `postgres` | `SET TRANSACTION READ ONLY` | `SET LOCAL statement_timeout` | yes |
| `sqlite` | `mode=ro` + authorizer | watchdog `interrupt()` | yes |
| `duckdb` | `read_only=True` + locked sandbox | watchdog | yes |
| `mssql` | **none** | **none per query** | no |
| `redshift` | `SET TRANSACTION READ ONLY` | `SET statement_timeout` | measured by hand |
| `snowflake` | **none** | `cursor.execute(timeout=…)` | no |
| `bigquery` | **none** | `job_timeout_ms` | no |
| `sqlalchemy` | **none** | best-effort per dialect | no |

`nlqueries health` reports this per connector, so the row that applies to a
deployment does not have to be looked up here.

**Where the mechanism is "none", the only thing preventing a write is the
privilege granted to the login.** For those dialects the sections above are not
defence in depth — they are the whole defence.

Two entries deserve particular attention.

**Redshift enforces both, but no test here reaches a cluster.** CI cannot
provision one, so the mechanisms were measured by hand against Redshift
Serverless: a write is refused with SQLSTATE 25006 (`transaction is read-only`)
and a query over its budget is cancelled with SQLSTATE 57014. A read-only user
and a WLM query-monitoring rule remain worth having — the read-only transaction
restricts what a statement may do, not what the login may reach.

**The generic `sqlalchemy` connector reaches any engine SQLAlchemy supports**,
so no single statement about its behaviour holds. Nothing read-only is applied.

"Verified in this repository" means the mechanism is exercised by a test against
a real engine. Snowflake, BigQuery and Redshift require accounts that a test run
cannot provision, so they are recorded as unverified whatever their
documentation says.

### What to grant, per vendor

The same shape, in each vendor's vocabulary:

- a dedicated identity that owns nothing,
- `SELECT` on an explicit list of schemas or datasets and nothing wider,
- no ability to create, load, copy, or call arbitrary functions,
- a statement timeout and a cost or slot cap,
- and no membership in any role that reads the host filesystem or all data.

---

## If you take one thing

Create the role. Every other control here is defence in depth around it, and the
application-side ones are written by people who also write the bugs.
