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

The PostgreSQL connector opens every query in a read-only transaction
(`SET TRANSACTION READ ONLY`); what the other engines do is not the same
everywhere, and [What each connector enforces](#what-each-connector-enforces)
below is the per-engine account. PostgreSQL applies this to what a statement
*does* rather than to how it is written, so it refuses DML and DDL anywhere in the call graph,
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
`pg_write_all_data`. It also reads `search_path` and
`default_transaction_read_only`, but does not report or judge either: the health
line names only the login, the database and the findings above. Check those two
yourself with `SHOW search_path;` and the first query in step 6.

---

## Transport security

This section is about the `postgres` connector, the only per-vendor connector
that reads `ssl_mode` and `ssl_ca_cert` (with `ssl_client_cert` and
`ssl_client_key`); the generic connector applies the same settings to libpq
PostgreSQL URLs, [below](#the-generic-connector). The others take no
certificate settings: `redshift` always connects with `ssl=True` and offers no
mode or CA; `mssql` passes pymssql an `encryption` level, which encrypts the
session only when it is `require` ([SQL Server](#sql-server)); and `connect mysql`
sets no TLS options, so its sessions are encrypted only if the server insists.
`nlqueries health` prints a transport line for `postgres` and `mssql`
connectors and for `sqlalchemy` connectors that settle a mode (see below).

With no `ssl_mode` set, the `postgres` connector uses `require` and so refuses a
server that offers no TLS. An explicit `ssl_mode` is honoured as given,
including `disable`, `allow` and `prefer`, which permit an unencrypted connection
(`nlqueries health` reports that). What it verifies beyond encryption depends on
`ssl_mode` and whether `ssl_ca_cert` is configured.

**Where to set them.** `nlqueries connect` has no TLS options. Register the
connector first, then add the keys by hand to its entry in `connectors.yaml` (the
path in `CONNECTORS_FILE` if set, otherwise `$NLQ_STATE_DIR/connectors.yaml`,
which is `~/.nlqueries/connectors.yaml` by default and
`/data/nlqueries/connectors.yaml` in the Docker image):

```yaml
postgres:db.internal:shop:
  db_type: postgres
  # ... the other keys `connect` wrote ...
  ssl_ca_cert: /etc/ssl/certs/db-ca.pem
```

`connect` tests the connection under the default `require` before it writes the
entry, so a server that offers no TLS cannot be registered this way. Running
`connect` again for the same connector replaces its whole entry, so add the
`ssl_*` keys back afterwards.

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

### SQL Server

The `mssql` connector passes pymssql's `encryption` argument on every
connection, from the `encryption` key in the connector's entry. Measured
against SQL Server 2022, which offers TLS without forcing it, reading
`sys.dm_exec_connections.encrypt_option` for the session:

| `encryption` | pymssql 2.4.3 to 2.4.5 | pymssql 2.3.0 to 2.4.2 |
|---|---|---|
| `require` | encrypted | **not encrypted**: the setting is ignored |
| `request` (default) | not encrypted | not encrypted |
| `off` | not encrypted | not encrypted |

So `request` changes nothing for an existing deployment, and only `require`
encrypts the session; it refuses a server that does not offer TLS. Any other
value is refused when the connector opens. The `mssql` extra now requires
pymssql 2.4.3. With an older pymssql installed, the connector refuses
`require`, since it would connect unencrypted, and `nlqueries health` reports
that the setting is not applied. Whether the server's certificate is verified
is FreeTDS configuration (a `ca file`), which the connector does not set and
cannot read back; under `require` it accepted a self-signed certificate. So
`nlqueries health` reports every level as a warning:

```
[WARN] Database (sql) transport: encryption is 'require'; the server's certificate
       is verified only if FreeTDS is configured with a CA file …
```

Set `require` once the server is known to offer TLS, or have the server force
encryption. `connect` writes no `encryption` key, so add it to the entry by hand
as for the `ssl_*` keys above.

### The generic connector

Everything above describes the `postgres` connector, which takes `ssl_mode` and
`ssl_ca_cert` as configuration. The generic `sqlalchemy` connector takes a whole
SQLAlchemy URL instead, and its posture comes from that URL.

It applies `ssl_mode`, `ssl_ca_cert`, `ssl_client_cert` and `ssl_client_key` when
the URL names a libpq-based PostgreSQL driver — `postgresql://`,
`postgresql+psycopg2://` or `postgresql+psycopg://` — because those are the
drivers whose parameter names map onto these settings directly.

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

Where it does apply them it resolves the mode the same way the `postgres`
connector does, so a root certificate with no `ssl_mode` selects `verify-full`
rather than leaving you on libpq's `prefer` — and the certificate counts whether
you supplied it in the credentials or in the URL's query string.

**It only resolves a mode when you configure at least one `ssl_*` setting on the
connector.** A URL alone does not trigger it. So
`postgresql://…/shop?sslrootcert=/etc/ssl/ca.pem` with no `ssl_*` settings beside
it connects under libpq's `prefer` — which falls back to plaintext and verifies
nothing — even though you supplied a CA, and `nlqueries health` reports nothing,
because the connector did not choose that posture and will not guess at it. Put
`sslmode=verify-full` in the same query string, or set `ssl_ca_cert` on the
connector instead of in the URL.

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

`nlqueries health` reports the posture for a `sqlalchemy` connector exactly as
for a `postgres` one, reading the URL's TLS parameters and the connector's
`ssl_*` settings together — so a split configuration is described by what is
actually in force rather than by half of it.

It reports nothing only when nothing on either side sets a mode. libpq then
applies its own `prefer` default, which is not the default the connector would
have chosen, and naming a mode the connection is not running under would be
worse than saying nothing.

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

**Spills go to a private directory.** A query too large for memory spills to
`temp_directory`. DuckDB's default puts it beside the database file
(`<database>.tmp`), so intermediate results, which are customer data, were
written wherever the database lived, or could not be written at all when that
location was read-only. The connector sets `temp_directory` at connect time to a
directory it creates once per process under the system temporary directory, with
mode `0700`, and removes it when the process exits. It is set before the
configuration is locked, so a query cannot move it.

Running DuckDB in an environment with nothing else to read — a container with
only the database file mounted, no host secrets and no egress — remains
worthwhile. The sandbox above is defence in depth and does not replace it.

**The SQL policy refuses file reads too.** Before a generated statement reaches
any connector, the policy refuses the file-reading functions sqlglot models
(`read_csv`, `read_parquet`, and Snowflake's `DIRECTORY` and `TO_FILE`) and a
string literal used as a table name (`FROM 'orders.csv'`, which DuckDB reads as
a file). The functions sqlglot does not model, `read_csv_auto` and `glob` among
them, were already refused as unrecognised. If the installed sqlglot records no
token positions, the policy cannot tell a string table name from a quoted one,
and refuses any statement with both a string and a table name.

---

## What each connector enforces

The read-only transaction described above is a PostgreSQL mechanism, and it is
not available on every engine. Where it is not, the connector does the next best
thing its engine allows -- usually running the query in a transaction it never
commits -- and the table records what the connector does, not what the engine
could support.

Those are not equivalent guarantees. Refusing a write and undoing one look the
same from the outside and are not the same thing: a rollback still ran the
statement, and anything it did outside the transaction stands.

| dialect | read-only mechanism | statement timeout | verified in this repository |
|---|---|---|---|
| `postgres` | `SET TRANSACTION READ ONLY` | `SET LOCAL statement_timeout` | yes |
| `sqlite` | `mode=ro` + authorizer | watchdog `interrupt()` | yes |
| `duckdb` | `read_only=True` + locked sandbox | watchdog | yes |
| `mssql` | never committed; rolled back either way. T-SQL DDL is transactional, so a `DROP` is undone too | **none per query** | no — asserted against a fake driver |
| `redshift` | `SET TRANSACTION READ ONLY` | `SET statement_timeout` | measured by hand |
| `snowflake` | `BEGIN` … `ROLLBACK`, so DML is undone. **DDL is not transactional and still stands** | `cursor.execute(timeout=…)` | no — asserted against a fake driver |
| `bigquery` | **none** | `job_timeout_ms` | no |
| `sqlalchemy` | DML never committed; rolled back either way, plus `SET TRANSACTION READ ONLY` where the dialect is `postgresql` or `redshift` and `START TRANSACTION READ ONLY` where it is `mysql` or `mariadb`. **DDL is not covered everywhere** — MySQL, MariaDB and Oracle commit implicitly around it, so a `CREATE` or `DROP` stands | best-effort per dialect | MySQL and MariaDB measured by hand (MySQL 8.0 and 8.4, MariaDB 10.11); otherwise asserted against a fake driver |

`nlqueries health` reports this per connector, so the row that applies to a
deployment does not have to be looked up here.

**Where the mechanism is "none", the only thing preventing a write is the
privilege granted to the login.** That is BigQuery, and for it the sections above
are not defence in depth — they are the whole defence.

Three entries deserve particular attention.

**Snowflake and Redshift queries on one connector run one at a time.** A Snowflake
transaction belongs to the session rather than the cursor, and the connector
holds a single connection for its lifetime while connector instances are cached
and callers arrive on threads. Two overlapping queries would otherwise share one
transaction, and the first `ROLLBACK` would end it for both — leaving the other
running under autocommit with nothing left to undo. The `BEGIN` … `ROLLBACK`
span is therefore serialised per connector. A shared session cannot offer both
concurrency and a per-query transaction, and a guard that stops holding under
load is worth less than the throughput it would buy.

Redshift is the same: its connector also holds one connection, and a second
query's `SET TRANSACTION READ ONLY` sent inside the first one's transaction is
refused, while the first one's rollback would end a transaction the second was
still reading in. Its `SET TRANSACTION READ ONLY` … rollback span is serialised
per connector in the same way.

The wait is bounded, so serialising cannot turn one stuck query into a stuck
connector: a caller waits for the statement timeout plus a margin, or five
minutes where no statement timeout applies, and then fails with an error saying
the connector was busy. `CONNECTOR_STATEMENT_TIMEOUT_SECONDS=0` bounds the
*query* but not the *queue* behind it, and those waiters occupy pool threads.

**Snowflake's DDL survives the rollback.** The transaction undoes an `INSERT`; it
does not undo a `CREATE TABLE`. If the role can create objects, it can create them
through a query, so the grant is carrying more of the boundary here than anywhere
else.

**BigQuery has nothing to undo with.** A single query job has no transaction. The
connector pins the job to standard SQL with no session and logs a statement type
other than `SELECT` at warning, but the job has already run by the time the type
is readable — an audit signal, not a control. Setting `BIGQUERY_MAX_BYTES_BILLED`
puts a cost cap on every job that answers a question: BigQuery fails a job that
would bill more before running it ([configuration.md](configuration.md)).

**MySQL and MariaDB open a read-only transaction.** SQLAlchemy's MySQL dialects
send no `BEGIN` of their own; the server opens a transaction implicitly on the
first statement. So the connector's first statement is
`START TRANSACTION READ ONLY`, and the rollback after the query ends it. Inside
it the server refuses a write to any table with error 1792, MyISAM and MEMORY
tables included, which the rollback alone could not undo. The session-scoped
`SET SESSION TRANSACTION READ ONLY` is not used, because it would outlive the
query on a pooled connection. The statement timeout is set after
`START TRANSACTION`, which commits implicitly, so it is in force for the query.
Measured by hand with PyMySQL against MySQL 8.0.46, MySQL 8.4.11 and MariaDB
10.11.14, before and after, with the same results on all three:

| | before | after |
|---|---|---|
| `INSERT` into an InnoDB table | rolled back | refused (1792) |
| `INSERT` into a MyISAM table | **kept** | refused (1792) |
| `SELECT 1; SELECT 2` with `client_flag=65536` in the URL | both ran | refused |
| `LOAD DATA LOCAL INFILE` with `local_infile=1` in the URL | file uploaded | refused by the client |
| `CREATE TABLE` | **kept** | **kept** (implicit commit) |
| a query over its statement timeout | cancelled | cancelled |
| a plain `INSERT` on the same pooled connection afterwards | committed | committed |

`tests/integration/test_mysql_integration.py` repeats this against a server you
name with `NLQ_TEST_MYSQL_URL`; it is skipped otherwise. The URL hardening behind
the two middle rows is described in
[connectors.md](connectors.md#mysql-and-mariadb-what-the-generic-connector-changes-in-the-url).

**Oracle's gap looks closable.** It has a transaction-scoped
`SET TRANSACTION READ ONLY` that must be the first statement of the transaction
-- exactly where the connector applies one -- and refuses DML and DDL thereafter
with ORA-01456. It is left out of `_apply_read_only` only because no test here
can reach an Oracle instance, and adding a statement to the execution path of an
untested engine is how every query against it becomes an error instead. Someone
with an instance should try it and move Oracle out of the column below.

**And the rollback does not reach their DDL.** MySQL, MariaDB and Oracle commit
implicitly around DDL, so a `CREATE` or `DROP` sent through this connector is
permanent whatever the transaction does — on MySQL and MariaDB the implicit
commit ends the read-only transaction before the DDL runs. It is the same limit
Snowflake has, for the same reason, and it applies to any engine behind the
generic SQLAlchemy connector that commits implicitly. On all of these the grant is the only thing
standing between a generated statement and a schema change, which is why the
role in this document is the boundary and the rollback is not.

**Redshift enforces both, but no test here reaches a cluster.** CI cannot
provision one, so the mechanisms were measured by hand against Redshift
Serverless: a write is refused with SQLSTATE 25006 (`transaction is read-only`)
and a query over its budget is cancelled with SQLSTATE 57014. A read-only user
and a WLM query-monitoring rule remain worth having — the read-only transaction
restricts what a statement may do, not what the login may reach.

**The generic `sqlalchemy` connector reaches any engine SQLAlchemy supports**,
so no single statement about its behaviour holds beyond the rollback, which does.
It adds `SET TRANSACTION READ ONLY` when the URL turns out to be `postgresql` or
`redshift`, `START TRANSACTION READ ONLY` for `mysql` or `mariadb`, and nothing
further for a dialect it does not recognise. It refuses `sqlite://` and
`duckdb://` URLs outright: the dedicated connectors open those files read-only
and sandboxed, and through this one the same file opened read-write, with
`ATTACH` and DuckDB's file functions available to a query.

"Verified in this repository" means the mechanism is exercised by a test against
a real engine. Snowflake, BigQuery, Redshift, SQL Server and MySQL require
accounts or servers a test run cannot provision, so they are recorded as
unverified whatever their documentation says — their tests assert the statements
sent to a fake driver, which shows the connector asks for the right thing and not
that the engine honours it.

### What to grant, per vendor

The same shape, in each vendor's vocabulary:

- a dedicated identity that owns nothing,
- `SELECT` on an explicit list of schemas or datasets and nothing wider,
- no ability to create, load, copy, or call arbitrary functions,
- a statement timeout and a cost or slot cap,
- and no membership in any role that reads the host filesystem or all data.

### What `nlqueries health` checks on other engines

The identity line described for PostgreSQL in step 6 is also printed for MySQL
and MariaDB (through the `sqlalchemy` connector), SQL Server and Snowflake. In
each case it names the login and anything it holds beyond reading; a login that
holds nothing more is `least privilege`, and a check that could not read the
grants is a failure, not a pass.

| Engine | Read from | Reported |
|---|---|---|
| MySQL, MariaDB | `SHOW GRANTS` | any privilege other than `SELECT`, `SHOW VIEW` and `USAGE`, with the scope it applies to (`INSERT`, `DROP`, `FILE`, `SUPER`, `ALL PRIVILEGES`, a dynamic `*_ADMIN` privilege, `EXECUTE`...); `WITH GRANT OPTION`; and any role granted, since a role's privileges are not expanded |
| SQL Server | role membership and `fn_my_permissions` for the database and the server | membership of any server role but `public`, and of any database role but `public` and `db_datareader`; any effective database permission beyond `CONNECT`, `SELECT`, `VIEW DEFINITION`, `VIEW DATABASE STATE`, `VIEW DATABASE PERFORMANCE STATE`, `SHOWPLAN` and `REFERENCES`, the two column-key `VIEW` permissions `public` holds in every database, and SQL Server 2022's `VIEW SECURITY DEFINITION`, `VIEW PERFORMANCE DEFINITION`, `VIEW CRYPTOGRAPHICALLY SECURED DEFINITION` and `VIEW DATABASE SECURITY STATE`, which those imply; any server permission beyond `CONNECT SQL`, `VIEW ANY DATABASE`, `VIEW ANY DEFINITION`, `VIEW SERVER STATE`, `VIEW SERVER PERFORMANCE STATE` and `CONNECT ANY DATABASE`, and their server counterparts `VIEW ANY SECURITY DEFINITION`, `VIEW ANY PERFORMANCE DEFINITION`, `VIEW ANY CRYPTOGRAPHICALLY SECURED DEFINITION` and `VIEW SERVER SECURITY STATE`. Effective permissions include what a custom role or a direct grant gives |
| Snowflake | `SHOW GRANTS TO ROLE` for the current role and every role it inherits, up to 50 | `ACCOUNTADMIN`, `ORGADMIN`, `SECURITYADMIN`, `SYSADMIN` or `USERADMIN` in reach, and any privilege other than `USAGE`, `SELECT`, `REFERENCES`, `MONITOR`, `OPERATE`, `READ` and `IMPORTED PRIVILEGES`, counted by privilege and object type. Secondary roles are not followed |

MySQL and SQL Server are read each time the pool opens a connection, as
PostgreSQL is. Snowflake is read the first time `health` asks, since walking the
role hierarchy is one `SHOW GRANTS` per role. As for PostgreSQL, the check reports
and does not refuse a connection.

On SQL Server the server scope is read apart from the database's. Where it cannot
be read (Azure SQL may not offer it from a user database), the line reports the
database and names what it could not read, `(not read: server permissions ...)`,
as a warning rather than a failure or a pass. A `sysadmin` login is still caught, because it is the database
owner everywhere and holds `CONTROL` among the database's effective permissions;
other server roles show only when the server scope can be read. On Snowflake the check waits up
to 30 seconds for a running query to release the session; if the query is still
running after that, the check reports that it could not be determined, and the
next `health` run reads it again.

---

## If you take one thing

Create the role. Every other control here is defence in depth around it, and the
application-side ones are written by people who also write the bugs.
