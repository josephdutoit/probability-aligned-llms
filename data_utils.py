"""Small data utilities for the project.

This module provides a helper to convert all CSV files in a directory
to Parquet format. It intentionally keeps dependencies minimal but
relies on `pandas` and a parquet engine such as `pyarrow` or `fastparquet`.

Usage example:
    from data_utils import csvs_to_parquet

    csvs_to_parquet("./data", recursive=False, overwrite=False)

Dependencies:
    pip install polars pyarrow

"""
from pathlib import Path
from typing import List, Optional


def csvs_to_parquet(
    dir_path: str | Path,
    output_dir: Optional[str | Path] = None,
    recursive: bool = False,
    glob_pattern: str = "*.csv",
    overwrite: bool = False,
    compression: Optional[str] = "snappy",
) -> List[Path]:
    """Convert all CSV files in `dir_path` to Parquet files.

    Args:
        dir_path: Directory containing CSV files to convert.
        output_dir: Optional directory to write parquet files to. If None,
            parquet files are written next to the source CSVs.
        recursive: If True, search subdirectories recursively.
        glob_pattern: Glob pattern to match CSV files (default "*.csv").
        overwrite: If False and a target parquet exists, the file is skipped.
        compression: Compression codec for parquet (e.g. 'snappy', 'gzip', None).
        engine: Parquet engine to use for `pandas.DataFrame.to_parquet`.

    Returns:
        A list of `pathlib.Path` objects for the written parquet files.

    Notes:
        - This is a convenience function and reads each CSV into memory using
          `pandas.read_csv`. For very large CSVs you may need a streaming
          approach; that is not implemented here for simplicity.
        - Install `pyarrow` (recommended) or `fastparquet` for parquet support.
    """

    try:
        import polars as pl
    except Exception as e:  # pragma: no cover - helpful runtime error
        raise ImportError(
            "polars is required for csvs_to_parquet. Install with: pip install polars pyarrow"
        ) from e

    src_dir = Path(dir_path)
    if not src_dir.exists() or not src_dir.is_dir():
        raise ValueError(f"dir_path must be an existing directory: {dir_path}")

    if output_dir is None:
        out_dir_base = None
    else:
        out_dir_base = Path(output_dir)
        out_dir_base.mkdir(parents=True, exist_ok=True)

    pattern = "**/" + glob_pattern if recursive else glob_pattern
    files = sorted(src_dir.glob(pattern))
    written: List[Path] = []

    if not files:
        return written

    for csv_path in files:
        if not csv_path.is_file():
            continue

        target_dir = out_dir_base if out_dir_base is not None else csv_path.parent
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = target_dir / (csv_path.stem + ".parquet")
        if parquet_path.exists() and not overwrite:
            # Skip existing file unless overwrite is requested
            written.append(parquet_path)
            continue

        try:
            # Read CSV into Polars DataFrame and write parquet
            df = pl.read_csv(csv_path)
            # Polars supports writing parquet directly
            df.write_parquet(parquet_path, compression=compression)
            written.append(parquet_path)
        except Exception as exc:  # keep going on failure but surface the error
            print(f"Failed to convert {csv_path} -> {parquet_path}: {exc}")

    return written


if __name__ == "__main__":
    # Simple CLI for convenience
    import argparse

    parser = argparse.ArgumentParser(description="Convert CSV files in a directory to Parquet.")
    parser.add_argument("dir", help="Source directory containing CSV files")
    parser.add_argument("--output-dir", help="Directory to write parquet files to", default=None)
    parser.add_argument("--recursive", help="Search subdirectories", action="store_true")
    parser.add_argument("--overwrite", help="Overwrite existing parquet files", action="store_true")
    args = parser.parse_args()

    results = csvs_to_parquet(args.dir, output_dir=args.output_dir, recursive=args.recursive, overwrite=args.overwrite)
    print(f"Wrote {len(results)} parquet files")
