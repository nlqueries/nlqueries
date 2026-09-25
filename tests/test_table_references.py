"""
tests.test_table_references
~~~~~~~~~~~~~~~~~~~~~~~~~~~
How a table is named in SQL the product builds itself.

A bare name resolves only in the connection's default schema. On Snowflake it
did not: `SELECT * FROM CALL_CENTER` failed where `TPCDS_SF10TCL.CALL_CENTER`
worked, and sampling for column descriptions skipped every table silently.
"""

from __future__ import annotations

import pytest
from nlqueries.connectors.base import qualified_table_sql, table_sample_sql


@pytest.mark.parametrize(
    ("dialect", "expected"),
    [
        ("snowflake", '"TPCDS_SF10TCL"."CALL_CENTER"'),
        ("postgresql", '"TPCDS_SF10TCL"."CALL_CENTER"'),  # a db_type sqlglot spells differently
        ("mysql", "`TPCDS_SF10TCL`.`CALL_CENTER`"),
        ("bigquery", "`TPCDS_SF10TCL`.`CALL_CENTER`"),
        ("mssql", "[TPCDS_SF10TCL].[CALL_CENTER]"),
        (None, '"TPCDS_SF10TCL"."CALL_CENTER"'),
    ],
)
def test_a_table_is_named_with_its_schema_quoted_for_the_dialect(
    dialect: str | None, expected: str
) -> None:
    assert qualified_table_sql("CALL_CENTER", "TPCDS_SF10TCL", dialect) == expected


def test_the_catalogue_s_case_is_kept() -> None:
    """Quoted, so a mixed-case name is not folded to lower case and missed."""
    assert qualified_table_sql("CallCenter", "Sales", "postgresql") == '"Sales"."CallCenter"'


def test_a_quote_inside_a_name_is_escaped() -> None:
    assert qualified_table_sql('odd"name', "s") == '"s"."odd""name"'


@pytest.mark.parametrize("schema", ["", None])
def test_without_a_schema_the_name_stands_alone(schema: str | None) -> None:
    """A knowledge base written before `schema` was recorded."""
    assert qualified_table_sql("orders", schema, "snowflake") == '"orders"'


@pytest.mark.parametrize(
    ("dialect", "expected"),
    [
        ("snowflake", 'SELECT * FROM "TPCDS_SF10TCL"."CALL_CENTER" LIMIT 100'),
        ("mysql", "SELECT * FROM `TPCDS_SF10TCL`.`CALL_CENTER` LIMIT 100"),
        # SQL Server has no LIMIT; the string this replaces was invalid T-SQL.
        ("mssql", "SELECT TOP 100 * FROM [TPCDS_SF10TCL].[CALL_CENTER]"),
    ],
)
def test_the_sample_query_is_qualified_and_bounded_for_the_dialect(
    dialect: str, expected: str
) -> None:
    assert table_sample_sql("CALL_CENTER", "TPCDS_SF10TCL", 100, dialect) == expected
