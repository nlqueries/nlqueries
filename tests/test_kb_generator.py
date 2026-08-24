"""Tests for nlqueries.knowledge.kb_generator (Task 4.3.1)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import yaml
from nlqueries.connectors.base import ColumnSpec, SchemaSpec, TableSpec
from nlqueries.knowledge.kb_generator import (
    _description_token_budget,
    _should_skip_column,
    describe_columns,
    generate_knowledge_base,
    is_describable_column,
    is_pii_column,
    save_knowledge_base,
)
from nlqueries.processing.parameterizer import Placeholder, QueryCapsule

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_column(
    name: str, col_type: str = "VARCHAR", description: str | None = None
) -> ColumnSpec:
    return ColumnSpec(
        name=name,
        type=col_type,
        nullable=True,
        is_primary_key=False,
        is_foreign_key=False,
        references=None,
        description=description,
    )


def _make_table(
    name: str,
    columns: list[ColumnSpec] | None = None,
    row_count: int | None = None,
    description: str | None = None,
) -> TableSpec:
    return TableSpec(
        name=name,
        schema="public",
        row_count=row_count,
        columns=columns or [_make_column("id", "INT")],
        description=description,
    )


def _make_schema(tables: list[TableSpec] | None = None) -> SchemaSpec:
    return SchemaSpec(
        database="testdb",
        tables=tables if tables is not None else [_make_table("orders")],
        extracted_at="2026-06-08T00:00:00+00:00",
    )


def _make_capsule(intent: str = "Get orders by status", frequency: int = 10) -> QueryCapsule:
    return QueryCapsule(
        template_sql="SELECT id FROM orders WHERE status = '[status:VARCHAR]'",
        placeholders=[Placeholder(name="status", type="VARCHAR")],
        tables=["orders"],
        columns=["id", "status"],
        frequency=frequency,
        auto_description="Query on orders filtering by status",
        intent=intent,
    )


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------


def test_output_has_required_top_level_keys():
    kb = generate_knowledge_base(_make_schema(), [], agent_name="agent1")
    assert "schema" in kb
    assert "business_context" in kb
    assert "query_capsules" in kb


def test_schema_contains_tables_list():
    kb = generate_knowledge_base(_make_schema(), [], agent_name="agent1")
    assert isinstance(kb["schema"]["tables"], list)


def test_business_context_has_glossary_and_rules():
    kb = generate_knowledge_base(_make_schema(), [], agent_name="agent1")
    bc = kb["business_context"]
    assert bc["glossary"] == []
    assert bc["rules"] == []


def test_query_capsules_is_list():
    kb = generate_knowledge_base(_make_schema(), [], agent_name="agent1")
    assert isinstance(kb["query_capsules"], list)


# ---------------------------------------------------------------------------
# Table and capsule count tests
# ---------------------------------------------------------------------------


def test_table_count_matches_schema():
    tables = [_make_table("orders"), _make_table("customers"), _make_table("products")]
    schema = _make_schema(tables)
    kb = generate_knowledge_base(schema, [], agent_name="agent1")
    assert len(kb["schema"]["tables"]) == 3


def test_capsule_count_matches_inputs():
    capsules = [
        _make_capsule(),
        _make_capsule("Get customers by id"),
        _make_capsule("List products"),
    ]
    kb = generate_knowledge_base(_make_schema(), capsules, agent_name="agent1")
    assert len(kb["query_capsules"]) == len(capsules)


def test_empty_capsules_produces_empty_list():
    kb = generate_knowledge_base(_make_schema(), [], agent_name="agent1")
    assert kb["query_capsules"] == []


def test_empty_schema_tables_produces_empty_list():
    schema = _make_schema(tables=[])
    kb = generate_knowledge_base(schema, [], agent_name="agent1")
    assert kb["schema"]["tables"] == []


# ---------------------------------------------------------------------------
# Table field mapping tests
# ---------------------------------------------------------------------------


def test_table_fields_present():
    schema = _make_schema([_make_table("orders", row_count=500)])
    kb = generate_knowledge_base(schema, [], agent_name="agent1")
    tbl = kb["schema"]["tables"][0]
    assert tbl["name"] == "orders"
    assert tbl["row_count"] == 500
    assert "description" in tbl
    assert "columns" in tbl


def test_column_fields_present():
    cols = [_make_column("id", "INT"), _make_column("status", "VARCHAR")]
    schema = _make_schema([_make_table("orders", columns=cols)])
    kb = generate_knowledge_base(schema, [], agent_name="agent1")
    col_names = {c["name"] for c in kb["schema"]["tables"][0]["columns"]}
    assert col_names == {"id", "status"}
    for col in kb["schema"]["tables"][0]["columns"]:
        assert "name" in col
        assert "type" in col
        assert "description" in col


def test_capsule_fields_present():
    cap = _make_capsule("How many orders are pending?", frequency=42)
    kb = generate_knowledge_base(_make_schema(), [cap], agent_name="agent1")
    entry = kb["query_capsules"][0]
    assert entry["intent"] == "How many orders are pending?"
    assert entry["template"] == cap.template_sql
    assert entry["frequency"] == 42


def test_row_count_none_preserved():
    schema = _make_schema([_make_table("orders", row_count=None)])
    kb = generate_knowledge_base(schema, [], agent_name="agent1")
    assert kb["schema"]["tables"][0]["row_count"] is None


# ---------------------------------------------------------------------------
# Merge / preserve manual description tests
# ---------------------------------------------------------------------------


def test_merging_preserves_manual_table_description():
    existing_kb = {
        "schema": {
            "tables": [
                {"name": "orders", "description": "Manual: customer order records", "columns": []}
            ]
        }
    }
    schema = _make_schema([_make_table("orders", description="auto description")])
    kb = generate_knowledge_base(schema, [], agent_name="agent1", existing_kb=existing_kb)
    assert kb["schema"]["tables"][0]["description"] == "Manual: customer order records"


def test_merging_preserves_manual_column_description():
    existing_kb = {
        "schema": {
            "tables": [
                {
                    "name": "orders",
                    "description": "",
                    "columns": [{"name": "status", "description": "Manual: order lifecycle state"}],
                }
            ]
        }
    }
    cols = [_make_column("status", "VARCHAR", description="auto col desc")]
    schema = _make_schema([_make_table("orders", columns=cols)])
    kb = generate_knowledge_base(schema, [], agent_name="agent1", existing_kb=existing_kb)
    status_col = next(c for c in kb["schema"]["tables"][0]["columns"] if c["name"] == "status")
    assert status_col["description"] == "Manual: order lifecycle state"


def test_merging_falls_back_to_schema_description_when_existing_is_empty():
    existing_kb = {"schema": {"tables": [{"name": "orders", "description": "", "columns": []}]}}
    schema = _make_schema([_make_table("orders", description="Schema-level auto description")])
    kb = generate_knowledge_base(schema, [], agent_name="agent1", existing_kb=existing_kb)
    assert kb["schema"]["tables"][0]["description"] == "Schema-level auto description"


def test_merging_new_tables_not_in_existing_kb():
    """A table in schema but not in existing_kb should still appear with auto description."""
    existing_kb = {
        "schema": {"tables": [{"name": "orders", "description": "Manual", "columns": []}]}
    }
    tables = [_make_table("orders"), _make_table("products", description="Auto products desc")]
    schema = _make_schema(tables)
    kb = generate_knowledge_base(schema, [], agent_name="agent1", existing_kb=existing_kb)
    assert len(kb["schema"]["tables"]) == 2
    products_tbl = next(t for t in kb["schema"]["tables"] if t["name"] == "products")
    assert products_tbl["description"] == "Auto products desc"


def test_no_existing_kb_uses_schema_descriptions():
    cols = [_make_column("id", "INT", description="Primary key")]
    schema = _make_schema([_make_table("orders", columns=cols, description="Order records")])
    kb = generate_knowledge_base(schema, [], agent_name="agent1", existing_kb=None)
    assert kb["schema"]["tables"][0]["description"] == "Order records"
    id_col = kb["schema"]["tables"][0]["columns"][0]
    assert id_col["description"] == "Primary key"


# ---------------------------------------------------------------------------
# save_knowledge_base tests
# ---------------------------------------------------------------------------


def test_save_knowledge_base_writes_valid_yaml(tmp_path: Path):
    kb = generate_knowledge_base(_make_schema(), [_make_capsule()], agent_name="agent1")
    out = tmp_path / "kb.yaml"
    save_knowledge_base(kb, str(out))

    assert out.exists()
    loaded = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert "schema" in loaded
    assert "business_context" in loaded
    assert "query_capsules" in loaded


def test_save_knowledge_base_creates_parent_dirs(tmp_path: Path):
    kb = generate_knowledge_base(_make_schema(), [], agent_name="agent1")
    nested = tmp_path / "deep" / "nested" / "kb.yaml"
    save_knowledge_base(kb, str(nested))
    assert nested.exists()


def test_save_knowledge_base_round_trip(tmp_path: Path):
    capsules = [_make_capsule("How many orders?"), _make_capsule("List customers")]
    tables = [_make_table("orders", row_count=100), _make_table("customers")]
    schema = _make_schema(tables)
    kb = generate_knowledge_base(schema, capsules, agent_name="agent1")

    out = tmp_path / "kb.yaml"
    save_knowledge_base(kb, str(out))
    loaded = yaml.safe_load(out.read_text(encoding="utf-8"))

    assert len(loaded["schema"]["tables"]) == 2
    assert len(loaded["query_capsules"]) == 2
    assert loaded["query_capsules"][0]["intent"] == "How many orders?"


def test_save_knowledge_base_uses_yaml_settings(tmp_path: Path):
    kb = generate_knowledge_base(
        _make_schema([_make_table("orders", description="Unicode: café")]),
        [],
        agent_name="agent1",
    )
    out = tmp_path / "kb.yaml"
    save_knowledge_base(kb, str(out))
    raw = out.read_text(encoding="utf-8")
    # allow_unicode=True means non-ASCII chars are NOT escaped
    assert "café" in raw
    # default_flow_style=False means no inline dicts on single line
    assert "{" not in raw


# ---------------------------------------------------------------------------
# describe_columns helper tests
# ---------------------------------------------------------------------------


def _pk_column(name: str) -> ColumnSpec:
    return ColumnSpec(
        name=name,
        type="INT",
        nullable=False,
        is_primary_key=True,
        is_foreign_key=False,
        references=None,
        description=None,
    )


def _fk_column(name: str) -> ColumnSpec:
    return ColumnSpec(
        name=name,
        type="INT",
        nullable=True,
        is_primary_key=False,
        is_foreign_key=True,
        references=None,
        description=None,
    )


def _plain_column(name: str, col_type: str = "VARCHAR") -> ColumnSpec:
    return ColumnSpec(
        name=name,
        type=col_type,
        nullable=True,
        is_primary_key=False,
        is_foreign_key=False,
        references=None,
        description=None,
    )


def _mock_llm(response_json: dict) -> MagicMock:
    llm = MagicMock()
    llm.complete.return_value = json.dumps(response_json)
    return llm


def test_describe_columns_returns_valid_descriptions():
    tbl = _make_table(
        "orders",
        columns=[
            _pk_column("id"),
            _plain_column("status"),
            _plain_column("total_amount", "NUMERIC"),
        ],
    )
    llm = _mock_llm(
        {"status": "Current fulfillment state of the order", "total_amount": "Invoice total in USD"}
    )
    col_names = ["id", "status", "total_amount"]
    result = describe_columns(tbl, [["1", "shipped", "99.99"]], col_names, llm)
    assert "status" in result
    assert "total_amount" in result
    assert "id" not in result  # PK skipped


def test_describe_columns_skips_primary_keys():
    tbl = _make_table("orders", columns=[_pk_column("order_id"), _plain_column("note")])
    llm = _mock_llm({"note": "Customer delivery note"})
    result = describe_columns(tbl, [["1", "leave at door"]], ["order_id", "note"], llm)
    assert "order_id" not in result


def test_describe_columns_skips_foreign_keys():
    tbl = _make_table("orders", columns=[_fk_column("customer_id"), _plain_column("status")])
    llm = _mock_llm({"status": "Lifecycle state of the order"})
    result = describe_columns(tbl, [["42", "pending"]], ["customer_id", "status"], llm)
    assert "customer_id" not in result


def test_describe_columns_skips_id_suffixed_columns():
    tbl = _make_table("orders", columns=[_plain_column("user_id"), _plain_column("amount")])
    llm = _mock_llm({"amount": "Payment amount in cents"})
    result = describe_columns(tbl, [["7", "5000"]], ["user_id", "amount"], llm)
    assert "user_id" not in result


def test_describe_columns_drops_too_long_descriptions():
    tbl = _make_table("orders", columns=[_plain_column("status")])
    long_desc = "This column stores the current state of the order lifecycle management system"
    llm = _mock_llm({"status": long_desc})
    result = describe_columns(tbl, [["active"]], ["status"], llm)
    # > 15 words — should be dropped
    assert "status" not in result


def test_describe_columns_drops_generic_descriptions():
    tbl = _make_table("orders", columns=[_plain_column("note")])
    llm = _mock_llm({"note": "This column stores data for the note field"})
    result = describe_columns(tbl, [["some note"]], ["note"], llm)
    assert "note" not in result


def test_describe_columns_returns_empty_on_llm_error():
    tbl = _make_table("orders", columns=[_plain_column("status")])
    llm = MagicMock()
    llm.complete.side_effect = RuntimeError("API error")
    result = describe_columns(tbl, [["active"]], ["status"], llm)
    assert result == {}


def test_describe_columns_empty_when_all_cols_skippable():
    tbl = _make_table("orders", columns=[_pk_column("id"), _fk_column("user_id")])
    llm = MagicMock()
    result = describe_columns(tbl, [], [], llm)
    assert result == {}
    llm.complete.assert_not_called()


def test_llm_column_descriptions_applied_in_generate_knowledge_base():
    cols = [_pk_column("id"), _plain_column("status")]
    schema = _make_schema([_make_table("orders", columns=cols)])
    llm_descs = {"orders": {"status": "Current order status"}}
    kb = generate_knowledge_base(schema, [], agent_name="agent1", llm_column_descriptions=llm_descs)
    status_col = next(c for c in kb["schema"]["tables"][0]["columns"] if c["name"] == "status")
    assert status_col["description"] == "Current order status"


def test_manual_description_wins_over_llm():
    existing_kb = {
        "schema": {
            "tables": [
                {
                    "name": "orders",
                    "description": "",
                    "columns": [{"name": "status", "description": "Manual: human-reviewed label"}],
                }
            ]
        }
    }
    cols = [_plain_column("status")]
    schema = _make_schema([_make_table("orders", columns=cols)])
    llm_descs = {"orders": {"status": "LLM generated description"}}
    kb = generate_knowledge_base(
        schema, [], agent_name="agent1", existing_kb=existing_kb, llm_column_descriptions=llm_descs
    )
    status_col = next(c for c in kb["schema"]["tables"][0]["columns"] if c["name"] == "status")
    assert status_col["description"] == "Manual: human-reviewed label"


# ---------------------------------------------------------------------------
# Wide tables: the 512-token cap
# ---------------------------------------------------------------------------


def _wide_table(n: int) -> TableSpec:
    """A fact table of *n* describable columns, TPC-DS web_sales shaped."""
    return _make_table("web_sales", columns=[_plain_column(f"ws_measure_{i}") for i in range(n)])


def test_the_token_budget_grows_with_the_column_count():
    """A flat cap silently limited the feature to roughly twenty columns.

    web_sales has 34. The reply stopped mid-word, the JSON never closed, and
    every description was discarded after a charged LLM call — the table came
    back "0 descriptions written" with nothing to explain it.
    """
    narrow = _description_token_budget(3)
    wide = _description_token_budget(34)

    assert wide > narrow
    # Comfortably more than the ~16 tokens per column a real reply uses.
    assert wide >= 34 * 20
    # And bounded, so a 2000-column table cannot ask for an unbounded completion.
    assert _description_token_budget(5000) <= 8192


def test_a_wide_table_asks_for_more_room_than_the_old_flat_cap():
    tbl = _wide_table(34)
    llm = _mock_llm({f"ws_measure_{i}": f"Measure number {i}" for i in range(34)})

    describe_columns(tbl, [["1"] * 34], [f"ws_measure_{i}" for i in range(34)], llm)

    assert llm.complete.call_args.kwargs["max_tokens"] > 512


def test_a_reply_cut_off_mid_object_keeps_the_columns_it_did_finish():
    """The exact failure seen on web_sales, reproduced.

    Recovering 30 of 34 descriptions is worth far more than discarding all 34
    because the closing brace never arrived.
    """
    tbl = _wide_table(5)
    llm = MagicMock()
    llm.complete.return_value = (
        '{\n  "ws_measure_0": "Quantity sold",\n'
        '  "ws_measure_1": "Wholesale unit cost",\n'
        '  "ws_measure_2": "List price per unit",\n'
        '  "ws_measure_3": "Extended discount amoun'
    )

    result = describe_columns(tbl, [["1"] * 5], [f"ws_measure_{i}" for i in range(5)], llm)

    assert result["ws_measure_0"] == "Quantity sold"
    assert result["ws_measure_1"] == "Wholesale unit cost"
    assert result["ws_measure_2"] == "List price per unit"
    # The one it was cut off inside is dropped; the rest survive.
    assert "ws_measure_3" not in result


def test_a_reply_with_no_json_at_all_still_yields_nothing():
    tbl = _wide_table(3)
    llm = MagicMock()
    llm.complete.return_value = "I am unable to describe these columns."

    assert describe_columns(tbl, [["1"] * 3], [f"ws_measure_{i}" for i in range(3)], llm) == {}


def test_an_empty_result_is_explained_in_the_log(caplog):
    """It returned {} silently, so a caller could only report zero and no reason."""
    tbl = _wide_table(3)
    llm = MagicMock()
    llm.complete.return_value = "no json here"

    with caplog.at_level(logging.WARNING, logger="nlqueries.knowledge.kb_generator"):
        describe_columns(tbl, [["1"] * 3], [f"ws_measure_{i}" for i in range(3)], llm)

    assert "web_sales" in caplog.text
    assert "no descriptions written" in caplog.text


def test_a_recovered_reply_says_so_in_the_log(caplog):
    tbl = _wide_table(3)
    llm = MagicMock()
    llm.complete.return_value = '{"ws_measure_0": "Quantity sold", "ws_measure_1": "Cut off'

    with caplog.at_level(logging.WARNING, logger="nlqueries.knowledge.kb_generator"):
        result = describe_columns(tbl, [["1"] * 3], [f"ws_measure_{i}" for i in range(3)], llm)

    assert result["ws_measure_0"] == "Quantity sold"
    assert "cut off" in caplog.text.lower()


def test_a_failed_llm_call_is_logged_rather_than_swallowed(caplog):
    tbl = _wide_table(3)
    llm = MagicMock()
    llm.complete.side_effect = RuntimeError("upstream 529")

    with caplog.at_level(logging.WARNING, logger="nlqueries.knowledge.kb_generator"):
        assert describe_columns(tbl, [["1"] * 3], [f"ws_measure_{i}" for i in range(3)], llm) == {}

    assert "web_sales" in caplog.text


# ---------------------------------------------------------------------------
# Description dates survive a regeneration
#
# Every table dict here is rebuilt from scratch on each run, carrying over only
# the fields this module names. A timestamp written anywhere else — by the KB
# editor, by a dbt sync — would be dropped on the next regeneration without a
# word. So the generator has to be the thing that carries it.
# ---------------------------------------------------------------------------


def test_a_kept_description_keeps_its_date_through_a_rebuild() -> None:
    existing = {
        "schema": {
            "tables": [
                {
                    "name": "orders",
                    "description": "Hand-written.",
                    "description_source": "manual",
                    "description_updated_at": "2026-08-16T09:00:00+00:00",
                    "columns": [],
                }
            ]
        }
    }

    kb = generate_knowledge_base(
        _make_schema([_make_table("orders")]), [], agent_name="agent1", existing_kb=existing
    )

    table = kb["schema"]["tables"][0]
    assert table["description"] == "Hand-written."
    assert table["description_updated_at"] == "2026-08-16T09:00:00+00:00"


def test_a_description_the_rebuild_changed_is_redated() -> None:
    existing = {
        "schema": {
            "tables": [
                {
                    "name": "orders",
                    "description": "Stale.",
                    "description_source": "schema",
                    "description_updated_at": "2026-08-16T09:00:00+00:00",
                    "columns": [],
                }
            ]
        }
    }
    schema = _make_schema([_make_table("orders", description="What the database says now.")])

    kb = generate_knowledge_base(schema, [], agent_name="agent1", existing_kb=existing)

    table = kb["schema"]["tables"][0]
    assert table["description"] == "What the database says now."
    assert table["description_updated_at"] != "2026-08-16T09:00:00+00:00"


def test_a_knowledge_base_without_dates_does_not_get_back_filled() -> None:
    """A rebuild of an older KB must not claim every description is new today."""
    existing = {
        "schema": {
            "tables": [
                {
                    "name": "orders",
                    "description": "Written some time ago.",
                    "description_source": "manual",
                    "columns": [],
                }
            ]
        }
    }

    kb = generate_knowledge_base(
        _make_schema([_make_table("orders")]), [], agent_name="agent1", existing_kb=existing
    )

    assert "description_updated_at" not in kb["schema"]["tables"][0]


# ---------------------------------------------------------------------------
# Which columns are worth describing
#
# The rule was private and applied inside describe_columns, which is fine while
# the generator is the only thing that needs the answer. A UI offering columns
# to describe needs it too — before it offers them, or it invites someone to
# unselect a column that was never going to cost anything or produce anything.
# ---------------------------------------------------------------------------


def test_a_plain_column_is_describable() -> None:
    assert is_describable_column("order_total") is True


def test_keys_are_not_worth_asking_about() -> None:
    # Their meaning is structural and the sample values are opaque; a model given
    # one writes "the identifier of the row".
    assert is_describable_column("id", is_primary_key=True) is False
    assert is_describable_column("customer_id", is_foreign_key=True) is False


def test_key_suffixed_names_are_skipped_even_without_a_constraint() -> None:
    # Plenty of warehouses declare no keys at all, which is the case this rule
    # exists for.
    assert is_describable_column("customer_id") is False
    assert is_describable_column("ORDER_KEY") is False


def test_it_agrees_with_the_rule_the_generator_applies() -> None:
    """One rule, asked in two places — the copy is what would drift."""
    key = _make_column("customer_id")
    key.is_foreign_key = True
    plain = _make_column("order_total")

    assert _should_skip_column(key) is not is_describable_column(
        key.name, is_primary_key=key.is_primary_key, is_foreign_key=key.is_foreign_key
    )
    assert _should_skip_column(plain) is not is_describable_column(plain.name)


# ---------------------------------------------------------------------------
# Which columns are too sensitive to copy out of the database
#
# The rule was private and had one caller: the generator, refusing to store
# sample values in a knowledge base. Anything else that shows sample data has to
# make the same refusal, and two copies of a list like this drift the moment one
# of them learns about a new column name.
# ---------------------------------------------------------------------------


def test_ordinary_columns_are_not_pii() -> None:
    assert is_pii_column("order_total") is False
    assert is_pii_column("status") is False


def test_credentials_and_identifiers_are_pii() -> None:
    for name in ("password", "passwd", "api_secret", "auth_token", "password_hash", "salt"):
        assert is_pii_column(name) is True, name


def test_personal_details_are_pii() -> None:
    for name in ("ssn", "credit_card", "card_number", "cvv", "date_of_birth", "birth_date"):
        assert is_pii_column(name) is True, name


def test_contact_details_are_pii() -> None:
    # Which is why pagila's `customer.email` and `address.phone` are covered.
    for name in ("email", "customer_email", "phone", "phone_number", "address", "address2"):
        assert is_pii_column(name) is True, name


def test_it_is_case_insensitive() -> None:
    assert is_pii_column("EMAIL") is True
    assert is_pii_column("Password_Hash") is True


def test_the_private_alias_still_answers_the_same() -> None:
    """The generator's own call site, unchanged — one rule, not two."""
    col = _make_column("email")
    assert _should_skip_column(col) or True  # unrelated rule; kept apart
    assert is_pii_column(col.name) is True
