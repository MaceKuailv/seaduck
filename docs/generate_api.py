#!/usr/bin/env python
"""Generate MyST API-reference pages for seaduck from the package docstrings.

The new documentation engine (``mystmd`` / Jupyter Book 2) does not run Sphinx,
so it cannot use ``sphinx.ext.autodoc``; it also has no Python-domain directives
(``{py:class}`` and friends are unknown to it).  This small in-house tool
reproduces the essentials with plain Markdown: it imports the package, reads
each object's docstring with :mod:`inspect`, parses the numpydoc sections with
:mod:`numpydoc.docscrape`, and writes headings + fenced ``python`` signature
blocks that ``mystmd`` renders on its own.

The docstrings themselves live only in ``seaduck/*.py``.  The Markdown files
this script writes are build artifacts: they are regenerated on every build and
are git-ignored -- never edit them by hand.

Usage
-----
    python docs/generate_api.py            # writes into docs/api/
    python docs/generate_api.py --out DIR  # custom output directory
"""

from __future__ import annotations

import argparse
import importlib
import inspect
from pathlib import Path

from numpydoc.docscrape import NumpyDocString

# ---------------------------------------------------------------------------
# What to document.  Each entry becomes one ``docs/api/<name>.md`` page.
#
# ``module``  -- document every public member defined in that module
#                (the old ``.. automodule::`` pages).
# ``classes`` -- document exactly these classes from ``module``
#                (the old ``.. autoclass::`` pages, e.g. OceData, RelCoord).
# ---------------------------------------------------------------------------
TARGETS: list[dict] = [
    # -- Public API -------------------------------------------------------
    {
        "name": "OceData",
        "title": "seaduck.OceData",
        "module": "seaduck.ocedata",
        "classes": ["OceData"],
    },
    {"name": "topology", "title": "seaduck.topology", "module": "seaduck.topology"},
    {"name": "eulerian", "title": "seaduck.eulerian", "module": "seaduck.eulerian"},
    {
        "name": "lagrangian",
        "title": "seaduck.lagrangian",
        "module": "seaduck.lagrangian",
    },
    {"name": "OceInterp", "title": "seaduck.OceInterp", "module": "seaduck.oceinterp"},
    {
        "name": "kernelNweight",
        "title": "seaduck.kernel_weight",
        "module": "seaduck.kernel_weight",
    },
    {
        "name": "eulerian_budget",
        "title": "seaduck.eulerian_budget",
        "module": "seaduck.eulerian_budget",
    },
    {
        "name": "lagrangian_budget",
        "title": "seaduck.lagrangian_budget",
        "module": "seaduck.lagrangian_budget",
    },
    # -- Internal API -----------------------------------------------------
    {
        "name": "smartread",
        "title": "seaduck.smart_read",
        "module": "seaduck.smart_read",
    },
    {"name": "getmasks", "title": "seaduck.get_masks", "module": "seaduck.get_masks"},
    {"name": "utils", "title": "seaduck.utils", "module": "seaduck.utils"},
    {
        "name": "RelCoord",
        "title": "seaduck.ocedata.RelCoord and subclasses",
        "module": "seaduck.ocedata",
        "classes": [
            "RelCoord",
            "HRel",
            "VRel",
            "VLinRel",
            "VlRel",
            "VlLinRel",
            "TRel",
            "TLinRel",
        ],
    },
]


def _is_public(name: str) -> bool:
    """Public names are those not starting with an underscore."""
    return not name.startswith("_")


def _signature(obj) -> str:
    """Best-effort call signature; ``()`` if it can't be introspected."""
    try:
        return str(inspect.signature(obj))
    except (ValueError, TypeError):
        return "()"


def _fmt_fields(section) -> list[str]:
    """Render a numpydoc parameter/returns section as a Markdown bullet list.

    numpydoc only splits ``name : type`` on `` : `` (space-colon-space); the
    seaduck docstrings mostly use ``name: type``, so recover the type here.
    Depending on the section the text may land in either field, so normalise
    both.
    """
    lines: list[str] = []
    for param in section:
        name, ptype = param.name, param.type
        if not name and ptype and ":" in ptype:
            name, ptype = (part.strip() for part in ptype.split(":", 1))
        elif not ptype and ":" in name:
            name, ptype = (part.strip() for part in name.split(":", 1))
        desc = " ".join(line.strip() for line in param.desc).strip()
        bullet = f"- **{name}**" if name else "-"
        if ptype:
            bullet += f" (*{ptype}*)"
        if desc:
            bullet += f" -- {desc}"
        lines.append(bullet)
    return lines


def _render_docstring(obj) -> list[str]:
    """Turn one object's numpydoc docstring into Markdown body lines."""
    doc = NumpyDocString(inspect.getdoc(obj) or "")
    body: list[str] = []

    if doc["Summary"]:
        body += doc["Summary"]
    if doc["Extended Summary"]:
        body += ["", *doc["Extended Summary"]]

    for heading, key in (
        ("Parameters", "Parameters"),
        ("Returns", "Returns"),
        ("Yields", "Yields"),
        ("Raises", "Raises"),
    ):
        if doc[key]:
            body += ["", f"**{heading}**", "", *_fmt_fields(doc[key])]

    if doc["Notes"]:
        body += ["", "**Notes**", "", *doc["Notes"]]

    return body


def _code_fence(signature: str) -> list[str]:
    """Wrap a call signature in a fenced ``python`` block."""
    return ["```python", signature, "```"]


def _render_callable(
    name: str, obj, *, level: int, kind: str, qualifier: str = ""
) -> list[str]:
    """Render a function or method: heading, signature fence, docstring.

    ``qualifier`` prefixes the heading name (e.g. ``OceData.`` for a method) so
    a bound method reads as ``OceData.check_readiness``.  ``kind`` adds an
    italic ``*(method)*`` / ``*(property)*`` tag; functions get none.
    """
    hashes = "#" * level
    heading = f"{hashes} {qualifier}{name}"
    if kind in ("method", "property"):
        heading += f"   *({kind})*"
    lines = [heading, ""]
    if kind != "property":
        lines += _code_fence(f"{name}{_signature(obj)}")
        lines.append("")
    lines += _render_docstring(obj)
    return lines


def _render_class(cls, canonical: str) -> list[str]:
    """Render a class heading + signature, then its public members."""
    lines = [f"## {cls.__name__}", ""]
    lines += _code_fence(f"class {cls.__name__}{_signature(cls)}")
    lines.append("")
    lines += _render_docstring(cls)

    for mname, member in inspect.getmembers(cls):
        if not _is_public(mname):
            continue
        is_prop = isinstance(member, property)
        if not (inspect.isfunction(member) or inspect.ismethod(member) or is_prop):
            continue
        target = member.fget if is_prop else member
        if inspect.getdoc(target) is None:
            continue
        lines.append("")
        lines += _render_callable(
            mname,
            target,
            level=3,
            kind="property" if is_prop else "method",
            qualifier=f"{cls.__name__}.",
        )
    return lines


def _render_page(target: dict) -> str:
    module = importlib.import_module(target["module"])
    lines = [f"# {target['title']}", ""]

    if "classes" in target:
        # Explicit class list (the old autoclass pages).
        for cls_name in target["classes"]:
            cls = getattr(module, cls_name)
            lines += _render_class(cls, f"{target['module']}.{cls_name}")
            lines += ["", "---", ""]
    else:
        # Whole-module page (the old automodule pages).
        module_doc = inspect.getdoc(module)
        if module_doc:
            lines += NumpyDocString(module_doc)["Summary"]
            lines.append("")
        classes = [
            (n, o)
            for n, o in inspect.getmembers(module, inspect.isclass)
            if o.__module__ == target["module"] and _is_public(n)
        ]
        funcs = [
            (n, o)
            for n, o in inspect.getmembers(module, inspect.isfunction)
            if o.__module__ == target["module"] and _is_public(n)
        ]
        for _, cls in classes:
            lines += _render_class(cls, f"{target['module']}.{cls.__name__}")
            lines += ["", "---", ""]
        for fn_name, fn in funcs:
            lines += _render_callable(
                fn_name,
                fn,
                level=2,
                kind="function",
            )
            lines += ["", "---", ""]

    # Drop a trailing separator for a tidy ending.
    while lines and lines[-1] in ("", "---"):
        lines.pop()
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "api",
        help="output directory for generated pages (default: docs/api)",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for target in TARGETS:
        page = _render_page(target)
        (args.out / f"{target['name']}.md").write_text(page, encoding="utf8")

    print(f"Wrote {len(TARGETS)} API pages to {args.out}")


if __name__ == "__main__":
    main()
