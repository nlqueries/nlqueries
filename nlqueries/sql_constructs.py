# nlqueries-core — OSS (BSL 1.1)
"""
nlqueries.sql_constructs
~~~~~~~~~~~~~~~~~~~~~~~~
The parts a generated query may be built from, as sqlglot expression classes.

:func:`nlqueries.sql_policy.evaluate` refuses any node whose class is not
listed here for the statement's dialect. Unmodelled functions
(:class:`exp.Anonymous`) are not classes worth listing and are checked by name
against :data:`nlqueries.sql_policy.ALLOWED_ANONYMOUS` instead.

The lists were built by parsing, in each of the eight dialects the CLI offers,
211 common analytics expressions and the queries in
``tests/sql_corpus.py``, collecting every class sqlglot produced, and reading
each one: every class here reads or computes and does nothing else. What is
left out is refused, including what a read query has no use for: session
variables and parameters (``@@version``, ``@x``, ``$1``, ``?``),
``CURRENT_USER``, and anything sqlglot models that is not here.

Names rather than classes, resolved against the installed sqlglot, because the
declared floor predates some of them and a later release may rename one. A name
that does not resolve is reported by ``tests/test_sql_constructs.py``, and a
class a later sqlglot introduces for something it used to parse otherwise is
refused until it is listed; ``nlqueries.sql_policy_report`` shows which.
"""

from __future__ import annotations

from sqlglot import exp

#: Clauses, operators, literals and names: the grammar of a read query.
READ_STRUCTURE: frozenset[str] = frozenset(
    {
        # statements and clauses
        "Select", "Union", "Subquery", "With", "CTE", "From", "Join", "Where",
        "Group", "Having", "Order", "Ordered", "Limit", "Offset", "Fetch",
        "LimitOptions", "Distinct", "Window", "WindowSpec", "Filter",
        "WithinGroup", "IgnoreNulls", "RespectNulls", "Rollup", "Cube",
        "GroupingSets", "Pivot", "PivotAlias", "Values", "Lambda",
        # names and values
        "Table", "TableAlias", "Alias", "Column", "Identifier", "Star",
        "Literal", "Null", "Boolean", "Var", "National", "HexString",
        "ByteString", "BitString", "RawString", "DataType", "DataTypeParam",
        "Interval", "IntervalSpan", "Tuple", "Paren", "Dot", "Bracket", "Slice",
        "PropertyEQ",
        "JSONPath", "JSONPathKey", "JSONPathRoot", "JSONPathSubscript",
        "JSONPathWildcard", "JSONKeyValue",
        # predicates and operators
        "Between", "In", "Is", "Like", "ILike", "SimilarTo", "Escape",
        "Collate", "Any", "All", "Not", "Neg", "EQ", "NEQ", "GT", "GTE", "LT",
        "LTE", "NullSafeEQ", "NullSafeNEQ", "Add", "Sub", "Mul", "Div", "IntDiv",
        "Mod", "DPipe", "BitwiseAnd", "BitwiseOr", "BitwiseXor", "BitwiseNot",
        "BitwiseLeftShift", "BitwiseRightShift", "And", "Or", "Xor",
    }
)  # fmt: skip

#: Functions sqlglot models that read or compute and do nothing else.
READ_FUNCTIONS: frozenset[str] = frozenset(
    {
        # aggregates
        "Count", "Sum", "Avg", "Min", "Max", "Stddev", "StddevPop",
        "StddevSamp", "Variance", "VariancePop", "Median", "ApproxDistinct",
        "ApproxQuantile", "AnyValue", "LogicalAnd", "LogicalOr", "Corr",
        "CovarPop", "CovarSamp", "PercentileCont", "PercentileDisc", "Mode",
        "ArrayAgg", "GroupConcat", "CountIf", "First", "Last", "ArgMax",
        "ArgMin", "JSONArrayAgg", "JSONObjectAgg",
        # window
        "RowNumber", "Rank", "DenseRank", "Ntile", "Lead", "Lag", "FirstValue",
        "LastValue", "NthValue", "PercentRank", "CumeDist",
        # conditional
        "Case", "If", "Coalesce", "Nullif", "Greatest", "Least", "Nvl2",
        "DecodeCase", "Exists",
        # conversion
        "Cast", "TryCast", "JSONCast", "Convert", "ToChar", "ToNumber", "ToDouble", "Date",
        "Timestamp", "TimeToStr", "StrToDate", "StrToTime", "TimeStrToTime",
        "TimeStrToDate", "TsOrDsToDate", "TsOrDsToTimestamp", "TimeToUnix",
        "UnixToTime", "StrToUnix", "DateStrToDate", "ParseJSON", "JSONFormat",
        # date and time
        "CurrentDate", "CurrentTime", "CurrentTimestamp", "CurrentDatetime",
        "Localtime", "Localtimestamp", "DateAdd", "DateSub", "DateDiff",
        "DateTrunc", "TimestampTrunc", "TimestampAdd", "TimestampSub",
        "TimestampDiff", "DatetimeAdd", "DatetimeSub", "DatetimeDiff",
        "DatetimeTrunc", "TimeAdd", "TimeSub", "TimeDiff", "TimeTrunc",
        "TsOrDsAdd", "TsOrDsDiff", "Extract", "Year", "Month", "Day",
        "DayOfWeek", "DayOfWeekIso", "DayOfMonth", "DayOfYear", "Week",
        "WeekOfYear", "Quarter", "Hour", "Minute", "Second", "LastDay",
        "WeekStart", "AddMonths", "MonthsBetween", "DateFromParts",
        "TimestampFromParts", "GenerateDateArray", "GenerateSeries",
        "AtTimeZone", "ConvertTimezone", "Datetime",
        "ExplodingGenerateSeries",
        # strings
        "Upper", "Lower", "Length", "Substring", "Trim", "Concat", "ConcatWs",
        "Replace", "Left", "Right", "Pad", "StrPosition", "Reverse", "Repeat",
        "SplitPart", "Split", "Initcap", "RegexpExtract", "RegexpLike",
        "RegexpILike", "RegexpReplace", "RegexpSubstr", "RegexpFullMatch",
        "RegexpSplit", "StartsWith", "Ascii", "Chr", "MD5", "MD5Digest", "SHA",
        "SHA1Digest", "SHA2", "Unicode",
        # numbers
        "Abs", "Round", "Ceil", "Floor", "Sqrt", "Pow", "Exp", "Ln", "Log",
        "Sign", "Trunc", "Pi", "Rand", "SafeDivide",
        # JSON
        "JSONExtract", "JSONExtractScalar", "JSONBExtract",
        "JSONBExtractScalar", "JSONValue", "JSONValueArray", "JSONArray",
        "JSONObject", "JSONBContains", "JSONExtractArray",
        # arrays and structs
        "Array", "ArrayContains", "ArraySize", "ArrayToString", "ArrayConcat",
        "ArrayFilter", "Transform", "Unnest", "Explode", "Struct",
        # the server's version: an information disclosure rather than an
        # action, and less than the connector's identity check already reports
        "CurrentVersion",
    }
)  # fmt: skip

#: Clauses only some engines have. sqlglot produces a function class only where
#: the dialect's parser maps something to it, so functions are not split by
#: dialect; clauses are, so that one engine's grammar is not accepted in
#: another's statement.
DIALECT_STRUCTURE: dict[str, frozenset[str]] = {
    "postgres": frozenset({"Lateral"}),
    "redshift": frozenset(),
    "mysql": frozenset({"Lateral"}),
    # No table hints: `WITH (TABLOCKX)` or `(UPDLOCK)` takes the lock that
    # `FOR UPDATE` is refused for, and its hint names parse as any word.
    "tsql": frozenset(),
    "snowflake": frozenset({"Qualify", "Lateral", "Kwarg"}),
    "bigquery": frozenset({"Qualify"}),
    "duckdb": frozenset({"Qualify", "Lateral", "Kwarg"}),
    "sqlite": frozenset(),
}


def _resolved(names: frozenset[str]) -> frozenset[type[exp.Expression]]:
    return frozenset(
        cls
        for name in names
        if isinstance(cls := getattr(exp, name, None), type) and issubclass(cls, exp.Expression)
    )


_BASE = _resolved(READ_STRUCTURE | READ_FUNCTIONS)
_BY_DIALECT = {dialect: _BASE | _resolved(extra) for dialect, extra in DIALECT_STRUCTURE.items()}


def allowed_constructs(sqlglot_dialect: str) -> frozenset[type[exp.Expression]]:
    """The classes a query in *sqlglot_dialect* may contain.

    A dialect without an entry gets the classes every dialect shares.
    """
    return _BY_DIALECT.get(sqlglot_dialect, _BASE)


def unresolved_names() -> list[str]:
    """Listed names the installed sqlglot does not define, for the tests."""
    every = READ_STRUCTURE | READ_FUNCTIONS | frozenset().union(*DIALECT_STRUCTURE.values())
    return sorted(name for name in every if not isinstance(getattr(exp, name, None), type))
