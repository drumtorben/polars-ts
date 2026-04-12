"""Generate MkDocs reference pages from Rust doc comments.

Parses /// and //! doc comments from src/*.rs files and produces
markdown pages under reference/rust/.
"""

from __future__ import annotations

import re
from pathlib import Path

import mkdocs_gen_files

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"


def _parse_rust_file(path: Path) -> dict:
    """Extract module docs and function docs from a single .rs file."""
    lines = path.read_text().splitlines()
    module_docs: list[str] = []
    functions: list[dict] = []

    i = 0
    # Collect //! module-level docs at the top of the file.
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("//!"):
            module_docs.append(stripped.removeprefix("//!").removeprefix(" "))
            i += 1
        elif stripped == "" and module_docs:
            # Allow blank lines within module doc block
            module_docs.append("")
            i += 1
        else:
            break

    # Scan for functions preceded by /// doc comments.
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("///"):
            doc_lines: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("///"):
                doc_lines.append(lines[i].strip().removeprefix("///").removeprefix(" "))
                i += 1

            # Collect attributes (#[pyfunction], #[pyo3(...)], #[polars_expr(...)])
            attrs: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("#["):
                attrs.append(lines[i].strip())
                i += 1

            # Next line should be the function signature
            if i < len(lines):
                sig_line = lines[i].strip()
                # Collect multi-line signature
                sig_lines = [sig_line]
                while i < len(lines) and not sig_lines[-1].rstrip().endswith("{"):
                    i += 1
                    if i < len(lines):
                        sig_lines.append(lines[i].strip())

                full_sig = " ".join(sig_lines)
                # Extract function name
                fn_match = re.search(r"fn\s+(\w+)", full_sig)
                if fn_match:
                    fn_name = fn_match.group(1)
                    is_public = "pub fn" in full_sig or "pub(crate) fn" in full_sig
                    is_pyfunction = any("#[pyfunction]" in a for a in attrs)
                    is_polars_expr = any("#[polars_expr" in a for a in attrs)

                    # Extract signature params from pyo3 signature attribute
                    pyo3_sig = ""
                    for a in attrs:
                        m = re.search(r"signature\s*=\s*\(([^)]*)\)", a)
                        if m:
                            pyo3_sig = m.group(1)

                    functions.append(
                        {
                            "name": fn_name,
                            "doc": "\n".join(doc_lines),
                            "signature": full_sig.split("{")[0].strip(),
                            "is_public": is_public,
                            "is_pyfunction": is_pyfunction,
                            "is_polars_expr": is_polars_expr,
                            "pyo3_signature": pyo3_sig,
                        }
                    )
            i += 1
        else:
            i += 1

    return {
        "module_doc": "\n".join(module_docs).strip(),
        "functions": functions,
    }


def _rust_doc_to_markdown(doc: str) -> str:
    """Convert Rust doc comment conventions to Markdown.

    Translates ``# Section`` headers to ``**Section**`` bold headings,
    and ``* `param` - desc`` argument lists to a Markdown table.
    """
    out_lines: list[str] = []
    in_args = False
    args_rows: list[tuple[str, str]] = []

    for line in doc.splitlines():
        # Convert # Arguments, # Returns, # Errors, # Parameters headings
        heading_match = re.match(r"^#\s+(.+)$", line)
        if heading_match:
            if in_args and args_rows:
                out_lines.extend(_format_args_table(args_rows))
                args_rows = []
            heading = heading_match.group(1)
            in_args = heading in ("Arguments", "Parameters")
            out_lines.append(f"**{heading}**\n")
            continue

        # Argument list items: * `param` - description  OR  - `param`: description
        arg_match = re.match(r"^[\*\-]\s+`(\w+)`\s*[-–—:]\s*(.+)$", line)
        if arg_match and in_args:
            args_rows.append((arg_match.group(1), arg_match.group(2)))
            continue

        # Continuation of argument description (indented line, not a new heading or arg)
        if in_args and args_rows and line.strip() and not re.match(r"^(#\s|\*\s+`)", line):
            name, desc = args_rows[-1]
            args_rows[-1] = (name, desc + " " + line.strip())
            continue

        if in_args and args_rows and line.strip() == "":
            out_lines.extend(_format_args_table(args_rows))
            args_rows = []
            in_args = False
            out_lines.append("")
            continue

        out_lines.append(line)

    if args_rows:
        out_lines.extend(_format_args_table(args_rows))

    return "\n".join(out_lines)


def _format_args_table(rows: list[tuple[str, str]]) -> list[str]:
    """Format argument rows as a Markdown table."""
    lines = ["| Parameter | Description |", "|-----------|-------------|"]
    for name, desc in rows:
        lines.append(f"| `{name}` | {desc} |")
    lines.append("")
    return lines


def _python_signature(fn: dict) -> str:
    """Build a Python-style signature from pyo3 metadata."""
    if not fn["pyo3_signature"]:
        return f"{fn['name']}(...)"

    params = fn["pyo3_signature"]
    return f"{fn['name']}({params})"


# ---------------------------------------------------------------------------
# Module name mapping for friendly display
# ---------------------------------------------------------------------------
MODULE_DISPLAY = {
    "dtw": "Dynamic Time Warping (DTW)",
    "ddtw": "Derivative DTW (DDTW)",
    "wdtw": "Weighted DTW (WDTW)",
    "dtw_multi": "Multivariate DTW",
    "msm": "Move-Split-Merge (MSM)",
    "msm_multi": "Multivariate MSM",
    "erp": "Edit Distance with Real Penalty (ERP)",
    "lcss": "Longest Common Subsequence (LCSS)",
    "twe": "Time Warp Edit Distance (TWE)",
    "mann_kendall": "Mann-Kendall Trend Test",
    "utils": "Utilities",
    "lib": "Module Registration",
}


def main() -> None:
    """Generate Rust reference pages."""
    for rs_file in sorted(SRC.glob("*.rs")):
        stem = rs_file.stem
        if stem == "lib":
            continue  # Skip module registration boilerplate

        parsed = _parse_rust_file(rs_file)
        if not parsed["functions"] and not parsed["module_doc"]:
            continue

        display_name = MODULE_DISPLAY.get(stem, stem)
        doc_path = Path("reference", "rust", f"{stem}.md")

        md_lines: list[str] = []
        md_lines.append(f"# {display_name}\n")
        md_lines.append(f"*Source: `src/{rs_file.name}`*\n")

        if parsed["module_doc"]:
            md_lines.append(_rust_doc_to_markdown(parsed["module_doc"]))
            md_lines.append("")

        # Public API functions (exposed to Python)
        public_fns = [f for f in parsed["functions"] if f["is_pyfunction"] or f["is_polars_expr"]]
        internal_fns = [f for f in parsed["functions"] if not f["is_pyfunction"] and not f["is_polars_expr"]]

        if public_fns:
            md_lines.append("## Python API\n")
            for fn in public_fns:
                py_sig = _python_signature(fn)
                md_lines.append(f"### {fn['name']}\n")
                md_lines.append(f"```python\n{py_sig}\n```\n")
                md_lines.append(_rust_doc_to_markdown(fn["doc"]))
                md_lines.append("")

        if internal_fns:
            md_lines.append("## Internal Functions\n")
            md_lines.append("These are internal Rust functions not directly callable from Python.\n")
            for fn in internal_fns:
                md_lines.append(f"### {fn['name']}\n")
                md_lines.append(_rust_doc_to_markdown(fn["doc"]))
                md_lines.append("")

        with mkdocs_gen_files.open(doc_path, "w") as fd:
            fd.write("\n".join(md_lines))

        mkdocs_gen_files.set_edit_path(doc_path, Path("src") / rs_file.name)

    # Generate the SUMMARY.md entries for literate-nav
    summary_path = Path("reference", "rust", "SUMMARY.md")
    summary_lines = ["# Rust API\n"]
    for rs_file in sorted(SRC.glob("*.rs")):
        stem = rs_file.stem
        if stem == "lib":
            continue
        display_name = MODULE_DISPLAY.get(stem, stem)
        summary_lines.append(f"- [{display_name}]({stem}.md)")

    with mkdocs_gen_files.open(summary_path, "w") as fd:
        fd.write("\n".join(summary_lines))


main()
