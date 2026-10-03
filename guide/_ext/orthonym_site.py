"""Site-specific pieces of the Orthonym documentation build.

- ``{tier}`` and ``{lamp}`` roles: a tier id or a tier name with its mark. The
  four marks are the ones drawn in docs/readme/build.py: the SHAPE carries the
  tier and the colour only agrees, so they stay distinct in greyscale.
- the accuracy table: every percentage is recomputed here from the counts in
  _data/accuracy.json, the rule "the columns other than Molecules add up to
  Molecules" is checked, and the build stops if the page's table differs.
- the internals pages: one page per module of the package, written at build
  time into reference/internals/generated/ (not kept in git).
- ``<!-- include: FILE -->``: puts a repository file (CONTRIBUTING.md,
  CHANGELOG.md) into a page at build time, with its links pointed at the site
  or at GitHub.
- llms.txt and llms-full.txt at the site root, from the Markdown sources.
"""
from __future__ import annotations

import html
import inspect
import json
import logging
import re
from pathlib import Path

from docutils import nodes
from sphinx.util import logging as sphinx_logging

LOG = sphinx_logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
GUIDE = HERE.parent
REPO = GUIDE.parent
PKG = REPO / "src" / "orthonym"
GITHUB = "https://github.com/Steinbeck-Lab/Orthonym/blob/main/"

# ------------------------------------------------------------------ tier marks
# Engine tier id -> the lamp it lights (the web app's grouping).
TIER_LAMP = {
    "pin_verified": "pin",
    "pin_unverified": "fallback",
    "systematic_verified": "fallback",
    "best_effort": "best_effort",
    "abstain": "abstain",
}
LAMP_LABEL = {"pin": "PIN", "fallback": "FALLBACK", "best_effort": "BEST EFFORT", "abstain": "NO NAME"}
LAMP_TITLE = {
    "pin": "Preferred IUPAC Name: the strict PIN path built it and verified it",
    "fallback": "A checked name whose preferred status is not certified",
    "best_effort": "A name from the last-resort producers, shown with its own read-back verdict",
    "abstain": "No name: the engine declined",
}
# The 18-unit marks of build.py, stroked in currentColor so the theme sets the colour.
_RING = 'fill="none" stroke="currentColor"'
MARK_BODY = {
    "pin": (f'<circle {_RING} cx="9" cy="9" r="8" stroke-width="1.25"/>'
            f'<circle {_RING} cx="9" cy="9" r="5.5" stroke-width="1.25"/>'
            '<circle cx="9" cy="9" r="3" fill="currentColor"/>'),
    "fallback": (f'<circle {_RING} cx="9" cy="9" r="7" stroke-width="1.75" stroke-dasharray="3.23 2.27"/>'
                 '<circle cx="9" cy="9" r="3" fill="currentColor"/>'),
    "best_effort": f'<circle {_RING} cx="9" cy="9" r="7" stroke-width="1.75" stroke-dasharray="1.1 3.3"/>',
    "abstain": f'<circle {_RING} cx="9" cy="9" r="7" stroke-width="1"/>',
}


def mark_svg(lamp: str, cls: str = "") -> str:
    return (f'<svg class="ot-mark ot-mark--{lamp.replace("_", "-")} {cls}" viewBox="0 0 18 18" '
            f'width="18" height="18" aria-hidden="true" focusable="false">{MARK_BODY[lamp]}</svg>')


def _tier_role(name, rawtext, text, lineno, inliner, options=None, content=None):
    tier = text.strip()
    if tier not in TIER_LAMP:
        msg = inliner.reporter.error(f"unknown tier {tier!r}", line=lineno)
        return [inliner.problematic(rawtext, rawtext, msg)], [msg]
    lamp = TIER_LAMP[tier]
    out = (f'<span class="ot-tier ot-tier--{lamp.replace("_", "-")}" title="{html.escape(LAMP_TITLE[lamp])}">'
           f'{mark_svg(lamp)}<code class="ot-tier-id">{tier}</code></span>')
    return [nodes.raw(rawtext, out, format="html")], []


def _lamp_role(name, rawtext, text, lineno, inliner, options=None, content=None):
    lamp = text.strip()
    if lamp not in LAMP_LABEL:
        msg = inliner.reporter.error(f"unknown lamp {lamp!r}", line=lineno)
        return [inliner.problematic(rawtext, rawtext, msg)], [msg]
    out = (f'<span class="ot-tier ot-tier--{lamp.replace("_", "-")}" title="{html.escape(LAMP_TITLE[lamp])}">'
           f'{mark_svg(lamp)}<span class="ot-lamp-label">{LAMP_LABEL[lamp]}</span></span>')
    return [nodes.raw(rawtext, out, format="html")], []


# ------------------------------------------------------------------ accuracy table
def _fmt(n: int) -> str:
    return f"{n:,}"


def expected_accuracy_rows():
    """The table rows as they must read, recomputed from _data/accuracy.json."""
    data = json.loads((GUIDE / "_data" / "accuracy.json").read_text())
    out = []
    for row in data["rows"]:
        n = row["molecules"]
        parts = [row["named"], row["declined"], row["unreadable"], row["other_layers"]]
        if sum(parts) != n:
            raise RuntimeError(f"accuracy row {row['set']!r}: {parts} add up to {sum(parts)}, not {n}")
        if row.get("wrong", 0) != 0:
            raise RuntimeError(f"accuracy row {row['set']!r}: the source reports wrong structures")
        pct = 100 * row["named"] / n
        out.append([row["set"], _fmt(n), f"{_fmt(row['named'])} ({pct:.2f} %)",
                    _fmt(row["declined"]), _fmt(row["unreadable"]), _fmt(row["other_layers"])])
    return out


def check_accuracy_page(app, text):
    """The table in accuracy.md must equal the recomputed rows, cell for cell."""
    app.env.note_dependency(str(GUIDE / "_data" / "accuracy.json"))
    got = []
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and len(cells) == 6 and cells[0] not in ("Set",) and not cells[0].startswith(":"):
            got.append(cells)
    want = expected_accuracy_rows()
    if got != want:
        raise RuntimeError("guide/accuracy.md: the table does not match _data/accuracy.json:\n"
                           f"page: {got}\nwant: {want}")


# ------------------------------------------------------------------ internals
GROUPS = [
    ("perception", "Perception", "Reading the structure: rings, characteristic groups, stereocentres and their descriptors."),
    ("rules", "Rules", "The nomenclature rules: seniority, parent hydrides, locants, ring systems and name classes."),
    ("assembly", "Assembly", "Writing the name: prefixes, locants, alphanumerical order, enclosing marks, and the general engine."),
    ("validation", "Validation", "The OPSIN round trip, the atom-coverage check and the other checks on a name."),
    ("data", "Data", "The naming tables: retained names, ring systems, stems and prefixes."),
    ("decomposition", "Decomposition", "Naming large structures from their fragments."),
    ("metrics", "Metrics", "Provenance, tier labels and reason codes."),
    ("routing", "Routing", "Choosing the compound class that names a molecule."),
    ("", "Engine and tools", "The top level: the naming pipeline, the command line, the jars and the Java runtime bridge."),
]
BANNER = "Internal API. Names and behaviour may change between releases."


def _modules():
    out = []
    for p in sorted(PKG.rglob("*.py")):
        rel = p.relative_to(PKG.parent).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        out.append(".".join(parts))
    return sorted(set(out))


def _group_of(mod: str) -> str:
    parts = mod.split(".")
    if len(parts) > 2 or (len(parts) == 2 and (PKG / parts[1]).is_dir()):
        return parts[1]
    return ""


def _write_if_changed(path: Path, text: str):
    if not path.exists() or path.read_text() != text:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def generate_internals(app):
    gen = GUIDE / "reference" / "internals" / "generated"
    mods = _modules()
    wanted = set()
    by_group = {g[0]: [] for g in GROUPS}
    for m in mods:
        g = _group_of(m)
        by_group.setdefault(g, []).append(m)
    for key, title, blurb in GROUPS:
        members = by_group.get(key, [])
        slug = key or "top-level"
        toc = "\n".join(members)
        page = (f"# {title}\n\n```{{note}}\n:class: ot-internal\n{BANNER}\n```\n\n"
                f"{blurb} {len(members)} modules.\n\n"
                f"```{{toctree}}\n:maxdepth: 1\n\n{toc}\n```\n")
        _write_if_changed(gen / f"{slug}.md", page)
        wanted.add(f"{slug}.md")
        for m in members:
            if m == "orthonym":
                opts = "   :no-members:\n"
            elif m == "orthonym.errors":
                # is_failure_name is documented on the Python API page
                opts = "   :members:\n   :exclude-members: is_failure_name\n"
            else:
                opts = "   :members:\n"
            body = (f"# `{m}`\n\n```{{note}}\n:class: ot-internal\n{BANNER}\n```\n\n"
                    f"```{{eval-rst}}\n.. automodule:: {m}\n{opts}```\n")
            _write_if_changed(gen / f"{m}.md", body)
            wanted.add(f"{m}.md")
    for old in gen.glob("*.md"):
        if old.name not in wanted:
            old.unlink()
    _write_if_changed(gen / ".gitignore", "*\n")
    app.config.orthonym_internal_count = len(mods)


# The public API objects, whose docstrings are written for readers and must
# render with no warning (they are shown on reference/python-api).
PUBLIC_DOCSTRINGS = {
    "orthonym.name_compound", "orthonym.name_with_tree", "orthonym.classify_limit",
    "orthonym.Orthonym", "orthonym.NamingResult", "orthonym.NameTreeNode",
    "orthonym.OrthonymLimitError", "orthonym.errors.is_failure_name",
}
_DOCSTRING_OF = re.compile(r"docstring of (?P<obj>[A-Za-z0-9_.]+)")


class _InternalsWarningFilter(logging.Filter):
    """Drops the warnings that come from the internals pages.

    The internals pages render developer notes as written; their markup
    problems (and ambiguous names inside them) are counted and reported in
    one line instead of failing the site. A warning about a public API
    docstring, or from any other page, is never dropped.
    """

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.count = 0

    def _is_internal(self, record) -> bool:
        where = f"{getattr(record, 'location', '')} {record.getMessage()}"
        if "reference/internals/generated" in where:
            return True
        m = _DOCSTRING_OF.search(where)
        if m:
            obj = m.group("obj")
            return not any(obj == p or obj.startswith(p + ".") for p in PUBLIC_DOCSTRINGS)
        env = getattr(self.app, "env", None)
        cur = getattr(env, "current_document", None) if env is not None else None
        docname = getattr(cur, "docname", None) if cur is not None else None
        return bool(docname) and docname.startswith(self.app.config.orthonym_internals_prefix + "generated/")

    def filter(self, record):
        if record.levelno < logging.WARNING:
            return True
        if self._is_internal(record):
            self.count += 1
            return False
        return True


# ------------------------------------------------------------------ includes
_INCLUDE = re.compile(r"^<!-- include: (?P<file>[A-Za-z0-9_.\-/]+) -->$", re.M)
# Repository files that have a page on this site.
SITE_PAGES = {
    "guide/how-it-works.md": "how-it-works.md",
    "guide/declines.md": "declines.md",
    "guide/accuracy.md": "accuracy.md",
    "CHANGELOG.md": "project/changelog.md",
    "CONTRIBUTING.md": "project/contributing.md",
}


def _rewrite_links(text: str, docname: str) -> str:
    depth = docname.count("/")
    up = "../" * depth

    def repl(m):
        label, target = m.group(1), m.group(2)
        if re.match(r"^[a-z]+:", target) or target.startswith("#"):
            return m.group(0)
        path, _, frag = target.partition("#")
        if path in SITE_PAGES:
            return f"[{label}]({up}{SITE_PAGES[path]}{'#' + frag if frag else ''})"
        return f"[{label}]({GITHUB}{path}{'#' + frag if frag else ''})"

    return re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", repl, text)


def source_read(app, docname, source):
    text = source[0]

    def repl(m):
        f = REPO / m.group("file")
        app.env.note_dependency(str(f))
        return _rewrite_links(f.read_text(), docname)

    source[0] = _INCLUDE.sub(repl, text)
    if docname == "accuracy":
        check_accuracy_page(app, source[0])


# ------------------------------------------------------------------ llms.txt
_FENCE_RAW = re.compile(r"```\{raw\} html\n.*?\n```\n?", re.S)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_HTML_BLOCK = re.compile(r"^<(div|section|nav|a|p|span|figure|svg)[\s>].*?^</\1>\s*$", re.S | re.M)


def _plain(md: str) -> str:
    md = _FENCE_RAW.sub("", md)
    md = _HTML_COMMENT.sub("", md)
    md = _HTML_BLOCK.sub("", md)
    md = re.sub(r"\{(tier|lamp)\}`([^`]+)`", r"\2", md)
    md = re.sub(r"^---\n.*?\n---\n", "", md, flags=re.S)
    return re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"


def write_llms(app, exception):
    if exception is not None or app.builder.format != "html":
        return
    env = app.env
    order = [d for d in env.found_docs if not d.startswith(app.config.orthonym_internals_prefix + "generated/")]
    # reading order: the toctree order from the root
    seq = []

    def walk(doc):
        if doc in seq:
            return
        seq.append(doc)
        for child in env.toctree_includes.get(doc, []):
            walk(child)

    walk(app.config.root_doc)
    seq = [d for d in seq if d in order] + sorted(d for d in order if d not in seq)
    base = app.config.html_baseurl.rstrip("/")
    titles = {d: (env.titles[d].astext() if d in env.titles else d) for d in seq}
    lines = ["# Orthonym", "",
             "> Orthonym turns a molecular structure (SMILES) into its IUPAC name, following the IUPAC 2013 "
             "recommendations. OPSIN reads most names back for a round-trip check by full InChIKey "
             "before a name is returned; the few names checked another way carry a label that says so "
             "(see checking.html). When no name passes, the engine declines and says why.", "",
             "Install: pip install \"git+https://github.com/Steinbeck-Lab/Orthonym.git\", then "
             "orthonym --fetch-jars. Needs Python 3.10+ and a Java 11+ runtime. The whole text of this "
             "site is in llms-full.txt.", "", "## Pages", ""]
    for d in seq:
        url = f"{base}/{d}.html" if base else f"{d}.html"
        lines.append(f"- [{titles[d]}]({url})")
    lines.append("")
    out = Path(app.outdir)
    (out / "llms.txt").write_text("\n".join(lines))
    full = ["# Orthonym documentation (full text)", "",
            f"Version {app.config.release}. Generated from the Markdown sources of the site.", ""]
    for d in seq:
        src = Path(env.doc2path(d))
        text = src.read_text()
        text = _INCLUDE.sub(lambda m: (REPO / m.group("file")).read_text(), text)
        text = re.sub(r"```\{eval-rst\}\n.*?\n```\n?", "", text, flags=re.S)
        full += [f"<!-- page: {d} -->", _plain(text), ""]
        if d == "reference/python-api":
            full += _api_text()
    (out / "llms-full.txt").write_text("\n".join(full))


def _api_text():
    import orthonym
    from orthonym import errors
    objs = [("orthonym.name_compound", orthonym.name_compound),
            ("orthonym.name_with_tree", orthonym.name_with_tree),
            ("orthonym.Orthonym", orthonym.Orthonym)]
    objs += [(f"orthonym.Orthonym.{m}", getattr(orthonym.Orthonym, m)) for m in
             ("name", "name_tiered", "name_with_tree", "name_with_confidence")]
    objs += [("orthonym.NamingResult", orthonym.NamingResult),
             ("orthonym.NameTreeNode", orthonym.NameTreeNode),
             ("orthonym.classify_limit", orthonym.classify_limit),
             ("orthonym.OrthonymLimitError", orthonym.OrthonymLimitError),
             ("orthonym.errors.is_failure_name", errors.is_failure_name)]
    out = ["## Python API docstrings", ""]
    for name, obj in objs:
        try:
            sig = str(inspect.signature(obj)) if callable(obj) and not inspect.isclass(obj) else ""
        except (TypeError, ValueError):
            sig = ""
        out += [f"### {name}{sig}", "", inspect.getdoc(obj) or "", ""]
    return out


# ------------------------------------------------------------------ internal docstrings
_VERSION_TAG = re.compile(r"^(?:v\d+(?:\.\d+)?(?: [A-Za-z]+\d*(?:\.\d+)?)*|Phase \d+(?:\.\d+)?(?: [A-Za-z-]+\d*)*)\s*:\s*")


def strip_version_tags(app, what, name, obj, options, lines):
    """Drop a leading internal version tag (":...") from a docstring's first line."""
    if lines and not any(name == p or name.startswith(p + ".") for p in PUBLIC_DOCSTRINGS):
        lines[0] = _VERSION_TAG.sub("", lines[0], count=1)


# ------------------------------------------------------------------ setup
def setup(app):
    app.add_config_value("orthonym_internals_prefix", "reference/internals/", "env")
    app.add_config_value("orthonym_internal_count", 0, "env")
    app.add_role("tier", _tier_role)
    app.add_role("lamp", _lamp_role)
    app.connect("builder-inited", generate_internals)
    app.connect("source-read", source_read)
    app.connect("autodoc-process-docstring", strip_version_tags)
    app.connect("build-finished", write_llms)
    filt = _InternalsWarningFilter(app)
    # First in each handler's chain, so a dropped message is never counted.
    for h in logging.getLogger("sphinx").handlers:
        h.filters.insert(0, filt)

    def report(app_, exc):
        if filt.count:
            LOG.info(f"internals pages: {filt.count} docstring markup messages from developer notes not shown")

    app.connect("build-finished", report)
    return {"version": "1.0", "parallel_read_safe": False, "parallel_write_safe": True}
