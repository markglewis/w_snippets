#!/usr/bin/env python3
"""
sas_sql_converter.py

Extracts PROC SQL blocks from SAS (.sas) files and converts them into
standard SQL suitable for execution via pandas.read_sql() or SQLAlchemy's
text(). It is a best-effort, regex-based translator -- it handles the
common SAS-SQL-isms listed below, and flags constructs that need a human
to finish (SAS macro logic, PUT/INPUT format conversions, etc).

Handled automatically
----------------------
- Strips `proc sql; ... quit;` wrappers, keeps individual statements
- Strips SAS macro comments (`%* ...;`) and `*...;` statement comments
- Converts SAS date literals   'DDMONYYYY'd   ->  DATE 'YYYY-MM-DD'
- Converts SAS datetime literals 'DDMONYYYY:HH:MM:SS'dt -> TIMESTAMP 'YYYY-MM-DD HH:MM:SS'
- Converts word comparison operators eq/ne/lt/le/gt/ge/and/or/not (case-insens,
  word-boundary safe) to symbolic form: = != < <= > >=
- Converts SAS's symbolic NE operators ^= and ~= to != , and SAS's unary
  NOT operators ^ / ~ (e.g. ^(x=1), ^missing(x)) to the NOT keyword
- Converts `libname.table` references to just `table` (configurable map)
- Converts macro variables &var / &var. to SQLAlchemy bind params :var
  (so you can pass params={'var': ...} to pandas/sqlalchemy)
- Converts MONOTONIC() to ROW_NUMBER() OVER () with a warning comment
  (you must confirm the intended ORDER BY)
- Flags CALCULATED <alias> usage with a warning comment (standard SQL can't
  reference a SELECT alias in WHERE; needs to be wrapped in a subquery or
  the expression repeated -- left for manual fix since it's context-specific)
- Flags PUT()/INPUT() format conversions with a warning comment (no generic
  standard-SQL equivalent; usually becomes CAST/TO_CHAR/TO_DATE depending
  on target DB)

Usage
-----
    python sas_sql_converter.py input.sas -o output.sql
    python sas_sql_converter.py input.sas          # prints to stdout

    # Or use as a library:
    from sas_sql_converter import SasSqlConverter
    conv = SasSqlConverter(libname_map={"work": None, "mylib": "public"})
    queries = conv.convert_file("input.sas")
    for q in queries:
        print(q.sql)          # converted SQL text
        print(q.warnings)     # list of things to double check
"""

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

MONTHS = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
    "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
    "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
}

# word-boundary operator map -> applied only to whole words, case-insensitive
_WORD_OPS = {
    r"\beq\b": "=",
    r"\bne\b": "!=",
    r"\ble\b": "<=",
    r"\bge\b": ">=",
    r"\blt\b": "<",
    r"\bgt\b": ">",
}

# SAS symbolic operators that aren't valid/portable in standard SQL.
# Order matters: two-char forms (^=, ~=) must be replaced before the
# leftover bare ^ / ~ (SAS's unary NOT) is turned into the NOT keyword.
_NE_SYMBOL_RE = re.compile(r"\^=|~=")
_UNARY_NOT_RE = re.compile(r"[\^~]")

_PROC_SQL_BLOCK_RE = re.compile(
    r"proc\s+sql[^;]*;(?P<body>.*?)quit\s*;",
    re.IGNORECASE | re.DOTALL,
)

_SAS_DATE_RE = re.compile(r"'(\d{2})([A-Za-z]{3})(\d{2,4})'\s*d\b", re.IGNORECASE)
_SAS_DATETIME_RE = re.compile(
    r"'(\d{2})([A-Za-z]{3})(\d{2,4}):(\d{2}):(\d{2}):(\d{2})'\s*dt\b",
    re.IGNORECASE,
)

_MACRO_COMMENT_RE = re.compile(r"^\s*%\*.*?;\s*$", re.MULTILINE)
_SAS_LINE_COMMENT_RE = re.compile(r"(?m)^\s*\*[^;]*;")
_MACRO_VAR_RE = re.compile(r"&([A-Za-z_]\w*)\.?")
_QUOTED_MACRO_VAR_RE = re.compile(r"""(['"])&([A-Za-z_]\w*)\.?\1""")
_CALCULATED_RE = re.compile(r"\bcalculated\s+(\w+)", re.IGNORECASE)
_MONOTONIC_RE = re.compile(r"\bmonotonic\s*\(\s*\)", re.IGNORECASE)
_PUT_INPUT_RE = re.compile(r"\b(put|input)\s*\(", re.IGNORECASE)


def _expand_year(yy: str) -> str:
    if len(yy) == 4:
        return yy
    yy_i = int(yy)
    # SAS default pivot: 1920-2019 for 2-digit years; adjust if you need a
    # different window
    return f"19{yy}" if yy_i >= 20 else f"20{yy}"


def _sas_date_to_iso(match: "re.Match") -> str:
    dd, mon, yy = match.group(1), match.group(2).upper(), match.group(3)
    mm = MONTHS.get(mon, "01")
    yyyy = _expand_year(yy)
    return f"DATE '{yyyy}-{mm}-{dd}'"


def _sas_datetime_to_iso(match: "re.Match") -> str:
    dd, mon, yy, hh, mi, ss = match.groups()
    mm = MONTHS.get(mon.upper(), "01")
    yyyy = _expand_year(yy)
    return f"TIMESTAMP '{yyyy}-{mm}-{dd} {hh}:{mi}:{ss}'"


@dataclass
class ConvertedQuery:
    sql: str
    warnings: list = field(default_factory=list)


class SasSqlConverter:
    def __init__(self, libname_map: dict | None = None):
        """
        libname_map: optional dict mapping SAS libref (lowercase) -> schema
                     name to substitute, or None to drop the libref entirely
                     and keep just the table name. Librefs not in the map
                     are left untouched.
        """
        self.libname_map = {k.lower(): v for k, v in (libname_map or {}).items()}

    # ---- extraction -----------------------------------------------------

    def extract_proc_sql_blocks(self, sas_text: str) -> list:
        return [m.group("body").strip() for m in _PROC_SQL_BLOCK_RE.finditer(sas_text)]

    # ---- individual transform steps -------------------------------------

    def _strip_comments(self, sql: str) -> str:
        sql = _MACRO_COMMENT_RE.sub("", sql)
        sql = _SAS_LINE_COMMENT_RE.sub("", sql)
        return sql

    def _convert_dates(self, sql: str) -> str:
        sql = _SAS_DATE_RE.sub(_sas_date_to_iso, sql)
        sql = _SAS_DATETIME_RE.sub(_sas_datetime_to_iso, sql)
        return sql

    def _convert_word_operators(self, sql: str) -> str:
        for pattern, repl in _WORD_OPS.items():
            sql = re.sub(pattern, repl, sql, flags=re.IGNORECASE)
        return sql

    def _convert_symbolic_operators(self, sql: str) -> str:
        # ^=  and  ~=   ->  !=
        sql = _NE_SYMBOL_RE.sub("!=", sql)
        # any leftover bare ^ or ~ is SAS's unary NOT (e.g. ^(x=1), ^missing(x))
        sql = _UNARY_NOT_RE.sub("NOT ", sql)
        return sql

    def _convert_libnames(self, sql: str, warnings: list) -> str:
        if not self.libname_map:
            return sql

        def repl(match):
            lib = match.group(1)
            table = match.group(2)
            key = lib.lower()
            if key in self.libname_map:
                schema = self.libname_map[key]
                return f"{schema}.{table}" if schema else table
            return match.group(0)

        return re.sub(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b", repl, sql)

    def _convert_macro_vars(self, sql: str, warnings: list) -> str:
        # Quoted case first: "&var." or '&var.' -> :var (drop the quotes,
        # since a DBAPI bind param supplies its own literal at execution time)
        def repl_quoted(match):
            name = match.group(2)
            warnings.append(
                f"Macro variable &{name} (was quoted) converted to bind param "
                f":{name} -- pass it via params={{'{name}': ...}}"
            )
            return f":{name}"

        sql = _QUOTED_MACRO_VAR_RE.sub(repl_quoted, sql)

        def repl(match):
            name = match.group(1)
            warnings.append(
                f"Macro variable &{name} converted to bind param :{name} "
                f"-- pass it via params={{'{name}': ...}}"
            )
            return f":{name}"

        return _MACRO_VAR_RE.sub(repl, sql)

    def _flag_calculated(self, sql: str, warnings: list) -> str:
        for m in _CALCULATED_RE.finditer(sql):
            warnings.append(
                f"CALCULATED {m.group(1)} found: standard SQL can't reference "
                f"a SELECT alias in WHERE/HAVING directly. Wrap the query in a "
                f"subquery (SELECT * FROM (...) WHERE {m.group(1)} = ...) or "
                f"repeat the expression."
            )
        return _CALCULATED_RE.sub(lambda m: m.group(1), sql)

    def _flag_monotonic(self, sql: str, warnings: list) -> str:
        if _MONOTONIC_RE.search(sql):
            warnings.append(
                "MONOTONIC() converted to ROW_NUMBER() OVER () -- confirm/add "
                "an ORDER BY inside the OVER() clause to match intended row order."
            )
        return _MONOTONIC_RE.sub("ROW_NUMBER() OVER ()", sql)

    def _flag_put_input(self, sql: str, warnings: list) -> str:
        if _PUT_INPUT_RE.search(sql):
            warnings.append(
                "PUT()/INPUT() format conversion found: no generic standard-SQL "
                "equivalent. Typically becomes CAST(... AS type) or a DB-specific "
                "TO_CHAR/TO_DATE/FORMAT function -- convert by hand."
            )
        return sql

    def _cleanup_whitespace(self, sql: str) -> str:
        sql = re.sub(r"[ \t]+", " ", sql)
        sql = re.sub(r"\n\s*\n+", "\n\n", sql)
        return sql.strip()

    # ---- public API -------------------------------------------------------

    def convert(self, sas_sql: str) -> ConvertedQuery:
        warnings: list = []
        sql = sas_sql
        sql = self._strip_comments(sql)
        sql = self._convert_dates(sql)
        sql = self._convert_word_operators(sql)
        sql = self._convert_symbolic_operators(sql)
        sql = self._convert_libnames(sql, warnings)
        sql = self._flag_calculated(sql, warnings)
        sql = self._flag_monotonic(sql, warnings)
        sql = self._flag_put_input(sql, warnings)
        sql = self._convert_macro_vars(sql, warnings)
        sql = self._cleanup_whitespace(sql)
        # split into individual statements on top-level semicolons
        return ConvertedQuery(sql=sql, warnings=warnings)

    def convert_file(self, path: str) -> list:
        text = Path(path).read_text()
        blocks = self.extract_proc_sql_blocks(text)
        return [self.convert(b) for b in blocks]


def _format_output(queries: list) -> str:
    parts = []
    for i, q in enumerate(queries, 1):
        header = f"-- ==== Query {i} " + "=" * 40
        parts.append(header)
        if q.warnings:
            parts.append("-- REVIEW NEEDED:")
            for w in q.warnings:
                parts.append(f"--   * {w}")
        parts.append(q.sql)
        if not q.sql.rstrip().endswith(";"):
            parts.append(";")
        parts.append("")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Path to a .sas file containing one or more PROC SQL blocks")
    parser.add_argument("-o", "--output", help="Path to write converted SQL to (default: stdout)")
    parser.add_argument(
        "--libname",
        action="append",
        default=[],
        metavar="LIBREF=SCHEMA",
        help="Map a SAS libref to a schema name, e.g. --libname mylib=public. "
             "Use --libname mylib= (empty) to drop the libref entirely. "
             "Can be passed multiple times.",
    )
    args = parser.parse_args()

    libname_map = {}
    for item in args.libname:
        if "=" not in item:
            parser.error(f"--libname must be in LIBREF=SCHEMA form, got: {item}")
        lib, schema = item.split("=", 1)
        libname_map[lib.strip()] = schema.strip() or None

    converter = SasSqlConverter(libname_map=libname_map)
    queries = converter.convert_file(args.input)

    if not queries:
        print("No 'proc sql; ... quit;' blocks found in the input file.")
        return

    output = _format_output(queries)

    if args.output:
        Path(args.output).write_text(output)
        print(f"Wrote {len(queries)} converted quer{'y' if len(queries)==1 else 'ies'} to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
