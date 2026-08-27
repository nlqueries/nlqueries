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
READ ONLY`). That is worth having — PostgreSQL applies it to what a statement
*does* rather than how it is spelled, so it refuses DML and DDL anywhere in the
call graph, including inside a function a `SELECT` calls, and it refuses
sequence functions by name.

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

DuckDB is the exception to "the rest is yours to configure": its file access is
reachable from any `SELECT` rather than granted by a DBA, so the connector
closes it in code. See [DuckDB](#duckdb) below.

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

---

## Row-level security

If tenants share tables, RLS is the mechanism — with two caveats worth stating
plainly.

A table's **owner** bypasses its own policies unless the table is set to `FORCE
ROW LEVEL SECURITY`, so the NLQueries role must not own the tables it reads.
And a policy keyed on a session variable that submitted SQL can itself change is
not a boundary. Prefer a per-tenant login, or set the value in a connection hook
the query cannot reach, and reset pooled connections on return.

---

## SQLite

Point the connector at a database file the process cannot write:

- open it read-only (`file:...?mode=ro`), and
- make the file and its directory read-only at the filesystem level, because a
  read-only handle is an application-level promise and file permissions are not.

`:memory:` is not a substitute for a missing file. It succeeds, answers
questions about nothing, and looks like it is working.

---

## DuckDB

DuckDB reads the local filesystem through ordinary-looking table functions —
`read_csv_auto()`, `read_text()`, `glob()` — so a file read arrives inside a
perfectly well-formed `SELECT`, and the database file is not the boundary.
Opening the database read-only does nothing about any of them.

**The connector now closes this, and there is no setting to reopen it.** Every
DuckDB connection is made with external access off, extension autoinstall and
autoload off, and the configuration locked so no later `SET` can undo it. A
database on disk is opened read-only; `:memory:` cannot be (DuckDB refuses) and
does not need to be.

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

Two consequences worth knowing before you upgrade:

- **A database whose tables are views over external parquet or CSV will stop
  working**, because reading those files is precisely what is being refused.
  Materialise that data into the DuckDB file. This is a deliberate trade, not
  an oversight.
- **A path that does not exist is now refused rather than created.** DuckDB
  would otherwise make an empty database and answer every question with "no
  tables", which is a misconfiguration reported as a success.

Running it somewhere with nothing else to read — a container with only the
database file mounted, no host secrets, no egress — is still worth doing. The
sandbox above is defence in depth, not a reason to skip the boring one.

---

## Snowflake, BigQuery, Redshift, SQL Server

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
