"""Conversational data analyst agent.

The agent loads a CSV file from the user's Google Drive through the
`google_drive_mcp` connector, then answers questions using a fixed set of
data tools: load, describe columns, filter rows, group-by aggregates.
There is no shell, no sandbox and no code interpreter: the only way the
model can touch the data is through these four tools. Tool calls compose in
the SDK's standard tool-call programs, which lets the connector's file
result flow into `load_csv` without the CSV ever passing through the model.
The loaded dataset is cached per session, so follow-up questions reuse it
without reloading or re-explaining.
"""

import base64
import io
import threading
from typing import Annotated, Any

import pandas as pd
from pydantic import Field

from mistralai import agents

_DRIVE_CONNECTOR = "google_drive_mcp"

_MAX_SAMPLE_ROWS = 50
_MAX_GROUP_VALUES = 200
_MAX_COLUMN_SAMPLE_VALUES = 5

# Session-scoped cache of loaded datasets. Keys are session ids from the
# tool callback context; the worker process holds one entry per conversation.
_DATASETS: dict[str, pd.DataFrame] = {}
_DATASET_SOURCES: dict[str, str] = {}
_LOCK = threading.Lock()


def _dataset_for(ctx: agents.ToolCallbackContext, dataset_name: str) -> pd.DataFrame:
    key = ctx.session.id
    with _LOCK:
        df = _DATASETS.get(key)
        source = _DATASET_SOURCES.get(key)
    if df is None or (dataset_name not in ("default", "current") and source != dataset_name):
        raise agents.ToolError(
            "No dataset is loaded in this session (or the cache was evicted after a restart). "
            "Load the file again: read the CSV from Google Drive with the Drive connector "
            "tools, then call load_csv with its content before answering."
        )
    return df


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def _clean_row(row: dict[str, Any]) -> dict[str, Any]:
    return {str(k): _json_safe(v) for k, v in row.items()}


_FILTER_OPS = {
    "eq", "ne", "gt", "ge", "lt", "le",
    "contains", "startswith", "endswith", "in", "isnull", "notnull",
}

_AGGREGATIONS = {
    "count", "nunique", "sum", "mean", "median", "min", "max", "std", "var",
}

_MISSING = object()


def _apply_conditions(df: pd.DataFrame, conditions: list[dict[str, Any]]) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for condition in conditions:
        column = condition.get("column")
        op = condition.get("op")
        value = condition.get("value", _MISSING)
        if column not in df.columns:
            raise agents.ToolError(f"Unknown column '{column}'. Call describe_columns for the exact names.")
        if op not in _FILTER_OPS:
            raise agents.ToolError(
                f"Unsupported operator '{op}'. Supported: {', '.join(sorted(_FILTER_OPS))}."
            )
        series = df[column]
        if op == "isnull":
            mask &= series.isna()
        elif op == "notnull":
            mask &= series.notna()
        elif op == "in":
            if value is _MISSING or not isinstance(value, list):
                raise agents.ToolError("Operator 'in' needs a 'value' list, e.g. [\"EUR\", \"USD\"].")
            mask &= series.isin(value)
        elif op == "contains":
            mask &= series.astype(str).str.contains(str(value), case=False, na=False)
        elif op == "startswith":
            mask &= series.astype(str).str.startswith(str(value), na=False)
        elif op == "endswith":
            mask &= series.astype(str).str.endswith(str(value), na=False)
        else:
            if value is _MISSING:
                raise agents.ToolError(f"Operator '{op}' needs a 'value'.")
            numeric_series = pd.to_numeric(series, errors="coerce")
            numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            comparable = (
                numeric_series if not numeric_series.isna().all() and not pd.isna(numeric_value) else series
            )
            target = numeric_value if comparable is numeric_series else value
            if op == "eq":
                mask &= comparable == target
            elif op == "ne":
                mask &= comparable != target
            elif op == "gt":
                mask &= comparable > target
            elif op == "ge":
                mask &= comparable >= target
            elif op == "lt":
                mask &= comparable < target
            elif op == "le":
                mask &= comparable <= target
    return df[mask]


_CONDITION_DESCRIPTION = (
    "One condition: {\"column\": str, \"op\": str, \"value\": any}. "
    "Ops: eq, ne, gt, ge, lt, le, contains, startswith, endswith, in, isnull, notnull. "
    "Only 'isnull'/'notnull' omit 'value'; 'in' takes a list."
)


@agents.tool(model_access="both")
async def load_csv(
    ctx: agents.ToolCallbackContext,
    /,
    csv_text: Annotated[str, Field(description="The full CSV content, or base64-encoded CSV content.")],
    file_name: Annotated[str, Field(description="The file's name, as shown in Google Drive.")],
    encoding: Annotated[
        str,
        Field(
            description=(
                "'text' when csv_text holds the raw CSV, 'base64' when it holds the "
                "base64-encoded file content that the Drive connector's download tool returns. "
                "Defaults to 'text', with a base64 fallback when the text does not parse as CSV."
            )
        ),
    ] = "text",
) -> dict[str, Any]:
    """Parse CSV content and cache it for this session.

    Call this right after reading a CSV file from Google Drive with the
    connector's read tool, passing that tool result as `csv_text`. It returns
    the dataset summary: row count, column names and each column's inferred
    type. Later tools (`describe_columns`, `filter_rows`,
    `group_by_aggregate`) operate on this cached dataset; loading a new file
    in the same session replaces it.
    """
    if not csv_text.strip():
        raise agents.ToolError("csv_text is empty: the Drive read returned no content.")
    candidates: list[str] = []
    if encoding == "base64":
        try:
            candidates.append(base64.b64decode(csv_text, validate=True).decode("utf-8", errors="replace"))
        except Exception as error:
            raise agents.ToolError(f"csv_text is not valid base64 ({error}).") from error
    else:
        candidates.append(csv_text)
    df = None
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            parsed = pd.read_csv(io.StringIO(candidate))
        except Exception as error:
            last_error = error
            continue
        looks_like_csv = len(parsed.columns) > 1 or len(parsed) > 0 or any(
            sep in csv_text for sep in (",", "\n", "\r", "\t")
        )
        if looks_like_csv:
            df = parsed
            break
    if df is None:
        try:
            decoded = base64.b64decode(csv_text, validate=True).decode("utf-8", errors="replace")
            df = pd.read_csv(io.StringIO(decoded))
        except Exception:
            raise agents.ToolError(
                f"Not parseable as CSV ({last_error}). Read a CSV file, or export the Sheet as CSV."
            ) from last_error
    key = ctx.session.id
    with _LOCK:
        _DATASETS[key] = df
        _DATASET_SOURCES[key] = file_name
    return {
        "file_name": file_name,
        "rows": int(len(df)),
        "columns": [
            {"name": str(c), "dtype": str(df[c].dtype), "nulls": int(df[c].isna().sum())}
            for c in df.columns
        ],
    }


@agents.tool(model_access="both")
async def describe_columns(ctx: agents.ToolCallbackContext, /) -> dict[str, Any]:
    """Describe the loaded dataset's columns with summary statistics.

    Per column: inferred dtype, non-null and null counts, unique count, and
    for numeric columns min, max, mean, median and standard deviation; for
    text and category columns the most frequent values. Also returns the row
    count and the file the data was loaded from.
    """
    df = _dataset_for(ctx, "default")
    columns: list[dict[str, Any]] = []
    for column in df.columns:
        series = df[column]
        info: dict[str, Any] = {
            "name": str(column),
            "dtype": str(series.dtype),
            "non_null": int(series.notna().sum()),
            "nulls": int(series.isna().sum()),
            "unique": int(series.nunique(dropna=True)),
        }
        non_null_text = series.dropna().astype(str)
        numeric = pd.to_numeric(series, errors="coerce")
        numeric_like = non_null_text.str.fullmatch(r"-?\d+(\.\d+)?").sum()
        if len(non_null_text) > 0 and numeric_like / len(non_null_text) >= 0.9:
            stats = numeric.describe()
            info["min"] = _json_safe(stats.get("min"))
            info["max"] = _json_safe(stats.get("max"))
            info["mean"] = _json_safe(stats.get("mean"))
            info["median"] = _json_safe(stats.get("50%"))
            info["std"] = _json_safe(stats.get("std"))
        top = series.value_counts(dropna=True).head(_MAX_COLUMN_SAMPLE_VALUES)
        if len(top) > 0:
            info["most_frequent"] = [
                {"value": _json_safe(value), "count": int(count)} for value, count in top.items()
            ]
        columns.append(info)
    return {
        "file_name": _DATASET_SOURCES.get(ctx.session.id),
        "rows": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": columns,
    }


@agents.tool(model_access="both")
async def filter_rows(
    ctx: agents.ToolCallbackContext,
    /,
    conditions: Annotated[list[dict[str, Any]], Field(description=f"A list where each entry is {_CONDITION_DESCRIPTION}")],
    columns: Annotated[
        list[str] | None,
        Field(description="Optional: only return these columns. Omit for all columns."),
    ] = None,
    limit: Annotated[
        int, Field(description="Maximum rows to return (up to 50). More rows can exist than are shown.")
    ] = _MAX_SAMPLE_ROWS,
) -> dict[str, Any]:
    """Filter the loaded dataset's rows by conditions and return matching rows.

    Conditions are combined with AND. Returns the total number of matching
    rows, then at most `limit` rows. Use this for lookups and lists; use
    `group_by_aggregate` to compute numbers over matching rows.
    """
    df = _dataset_for(ctx, "default")
    if not conditions:
        raise agents.ToolError(
            "Pass at least one condition, or use group_by_aggregate for whole-dataset numbers."
        )
    unknown = [c for c in (columns or []) if c not in df.columns]
    if unknown:
        raise agents.ToolError(f"Unknown column(s): {', '.join(map(str, unknown))}.")
    matched = _apply_conditions(df, conditions)
    limit = max(1, min(int(limit), _MAX_SAMPLE_ROWS))
    selected = matched[columns].head(limit) if columns else matched.head(limit)
    return {
        "matching_rows": int(len(matched)),
        "returned_rows": int(len(selected)),
        "truncated": bool(len(matched) > len(selected)),
        "rows": [_clean_row(record) for record in selected.to_dict(orient="records")],
    }


@agents.tool(model_access="both")
async def group_by_aggregate(
    ctx: agents.ToolCallbackContext,
    /,
    group_by: Annotated[
        list[str], Field(description="Column(s) to group by, e.g. [\"region\"] or [\"region\", \"product\"].")
    ],
    metrics: Annotated[
        list[dict[str, Any]],
        Field(
            description=(
                "One metric per entry: {\"column\": str, \"agg\": str, \"alias\": str?}. "
                "Aggregations: count, nunique, sum, mean, median, min, max, std, var. "
                "For 'count', set column to any column. Optional 'alias' names the result column."
            )
        ),
    ],
    conditions: Annotated[
        list[dict[str, Any]] | None,
        Field(description=f"Optional filters applied before grouping: a list where each entry is {_CONDITION_DESCRIPTION}"),
    ] = None,
    sort_by: Annotated[
        list[dict[str, Any]] | None,
        Field(description="Optional sort: {\"column\": str, \"descending\": bool?} applied to the result."),
    ] = None,
    limit: Annotated[int, Field(description="Maximum group rows to return (up to 200).")] = _MAX_GROUP_VALUES,
) -> dict[str, Any]:
    """Group the loaded dataset and compute per-group aggregates.

    Optionally filter rows first with `conditions`. Returns one row per group
    with the requested metrics. Compute every number this way: never estimate
    or invent figures.
    """
    df = _dataset_for(ctx, "default")
    if not group_by:
        raise agents.ToolError("Pass at least one group_by column.")
    missing = [c for c in group_by if c not in df.columns]
    if missing:
        raise agents.ToolError(f"Unknown group_by column(s): {', '.join(missing)}.")
    data = _apply_conditions(df, conditions) if conditions else df
    if len(data) == 0:
        return {"rows_grouped": 0, "groups": []}
    invalid = {m.get("agg") for m in metrics} - _AGGREGATIONS
    if invalid:
        raise agents.ToolError(
            f"Unsupported aggregation(s): {', '.join(sorted(str(a) for a in invalid))}. "
            f"Supported: {', '.join(sorted(_AGGREGATIONS))}."
        )
    pieces: dict[str, pd.Series[Any]] = {}
    for metric in metrics:
        column = metric.get("column")
        agg = metric.get("agg")
        alias = metric.get("alias") or f"{agg}_{column}"
        if column not in df.columns:
            raise agents.ToolError(f"Unknown metric column '{column}'.")
        grouped = data.groupby(group_by, dropna=False)[column]
        if agg == "count":
            pieces[alias] = grouped.count()
        elif agg == "nunique":
            pieces[alias] = grouped.nunique(dropna=True)
        else:
            pieces[alias] = grouped.aggregate(_aggregate_one(agg))
    result = pd.DataFrame(pieces).reset_index()
    if sort_by:
        order = [entry for entry in sort_by if entry.get("column") in result.columns]
        if order:
            result = result.sort_values(
                [entry["column"] for entry in order],
                ascending=[not entry.get("descending", False) for entry in order],
            )
    result = result.head(max(1, min(int(limit), _MAX_GROUP_VALUES)))
    return {
        "rows_grouped": int(len(data)),
        "groups_returned": int(len(result)),
        "truncated": bool(data.groupby(group_by, dropna=False).ngroups > len(result)),
        "rows": [_clean_row(record) for record in result.to_dict(orient="records")],
    }


def _aggregate_one(agg: str):
    def aggregate(values: pd.Series) -> Any:
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.notna().any():
            return getattr(numeric.dropna(), agg)()
        plain = values.dropna()
        if len(plain) == 0:
            return None
        if agg in ("min", "max", "median"):
            return getattr(plain, agg)()
        raise agents.ToolError(
            f"Aggregation '{agg}' needs a numeric column; '{values.name}' has no numeric values."
        )

    return aggregate


agent = agents.Agent(
    model="ministral-14b-2512",
    name="data-analyst",
    instructions=(
        "You are a conversational data analyst. The user points you at a CSV file in "
        "their Google Drive (a Drive link, a file name or a description) and asks "
        "questions in plain language.\n\n"
        "Workflow:\n"
        "1. To load a file, find it with the google_drive_mcp connector's search_files "
        "or list_recent_files tool, then download it with download_file_content, and "
        "pass the download result's content to load_csv in the same tool program, so "
        "the CSV content flows from Drive into load_csv. The download returns the "
        "content base64-encoded: that is what the `encoding` parameter is for, it is "
        "NOT decoded/normal text. For Google Sheets, download_file_content exports "
        "text/csv by default; keep that. Loading a new file replaces the dataset of "
        "the session.\n"
        "2. After loading, call describe_columns to learn the schema, and restate it "
        "briefly to the user.\n"
        "3. Answer every question about the data only through describe_columns, "
        "filter_rows and group_by_aggregate. Compute every figure with a tool call: "
        "never estimate, extrapolate or invent numbers. If a value is not in the data, "
        "say so.\n"
        "4. Follow-up questions reuse the session's loaded dataset: never ask the user "
        "to re-share the file. Only reload when a tool explicitly says the cached "
        "dataset is no longer available.\n"
        "5. Prefer group_by_aggregate for counts, sums and averages, including over "
        "filtered subsets via its `conditions`. Use filter_rows for lookups, lists and "
        "examples, keeping `limit` small.\n"
        "6. Report real numbers with their context: dataset file, row counts, filters "
        "applied, and whether a returned sample was truncated.\n\n"
        "You have no shell, no sandbox and no code interpreter: the four data tools are "
        "the only way to touch the data. In tool-call programs, only chain tool calls; "
        "do not compute aggregates inline."
    ),
    connectors=[agents.connector(_DRIVE_CONNECTOR)],
    tools=[load_csv, describe_columns, filter_rows, group_by_aggregate],
)
