import polars as pl

def format_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """
    Standardizes a Polars DataFrame:
    - Integer columns -> Int64
    - Float/decimal columns -> Float64
    - String columns -> stripped of leading/trailing whitespace
    - Column names -> uppercased
    """
    cast_exprs = []

    for col, dtype in zip(df.columns, df.dtypes):
        if dtype.is_integer():
            cast_exprs.append(pl.col(col).cast(pl.Int64))
        elif dtype.is_float() or dtype == pl.Decimal:
            cast_exprs.append(pl.col(col).cast(pl.Float64))
        elif dtype == pl.Utf8:
            cast_exprs.append(pl.col(col).str.strip_chars())
        else:
            # Leave other types (bool, date, datetime, etc.) untouched
            cast_exprs.append(pl.col(col))

    df = df.with_columns(cast_exprs)
    df.columns = [c.upper() for c in df.columns]

    return df
