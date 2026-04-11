import polars as pl


def _to_dict(df: pl.DataFrame) -> dict:
    """Convert pairwise result to {(id1, id2): distance} dict, sorted keys."""
    rows = df.to_dicts()
    dist_col = [c for c in df.columns if c not in ("id_1", "id_2")][0]
    result = {}
    for r in rows:
        key = tuple(sorted([str(r["id_1"]), str(r["id_2"])]))
        result[key] = r[dist_col]
    return result
