"""Regression guard: the README must render as literal text on PyPI.

THE DEFECT THIS PINS
--------------------
PyPI renders every release's long_description with its own first-party MathJax
(pypi.org/static/js/utils/mathjax-config.js), configured as:

    window.MathJax={tex:{inlineMath:[["$","$"],["\\(","\\)"]]}};

MathJax pairs `$` delimiters ACROSS the document. Because the README quotes
prices in dollars, `## The $350 hello world` opened a math span that swallowed
everything up to the next `$`. The live PyPI page rendered the cost table as
italic math with the spaces stripped:

    "Total entry cost: free (Tier 3 sim) / clone-tier ~200-350 / official..."

`twine check` PASSES on this -- it validates that the description is
well-formed, not how a browser typesets it. The whole class of defect is
invisible to the packaging toolchain, which is why it needs a test here.

THE FIX
-------
Isolate each prose `$` in its own element (`<span>$</span>`). MathJax does not
pair delimiters across separate text nodes, so the dollars render literally.
Verified end-to-end against PyPI's own config: 10 math containers before,
0 after.

Escape-based fixes were tested and DO NOT WORK: comrak decodes both `\\$` and
`&#36;` back to a bare `$` before MathJax runs.
"""
import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parent.parent / "README.md"


def _split_code_fences(text):
    """Return (prose_lines, fence_lines) with 1-based line numbers."""
    prose, fence = [], []
    in_fence = False
    for i, line in enumerate(text.split("\n"), 1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            fence.append(i)
            continue
        (fence if in_fence else prose).append(i)
    return set(prose), set(fence)


def _text():
    return README.read_text(encoding="utf-8")


class TestReadmeRendersOnPypi:

    def test_no_bare_dollars_in_prose(self):
        """Every prose `$` must be element-isolated, or MathJax eats it."""
        text = _text()
        prose_idx, _ = _split_code_fences(text)
        bare = []
        for i, line in enumerate(text.split("\n"), 1):
            if i not in prose_idx:
                continue
            # Remove the safe, wrapped form first; anything left is a bare `$`.
            stripped = line.replace("<span>$</span>", "")
            if "$" in stripped:
                bare.append((i, line.strip()[:80]))
        assert not bare, (
            "Bare '$' in prose will be parsed as LaTeX by PyPI's MathJax and "
            "will corrupt the rendered page. Wrap each one as <span>$</span>. "
            f"Offending lines: {bare}"
        )

    def test_code_fence_dollars_left_alone(self):
        """Dollars inside code fences are MathJax-safe; they must NOT be wrapped.

        MathJax skips <code>/<pre>, and wrapping them would put literal markup
        into copy-pasteable shell commands.
        """
        text = _text()
        _, fence_idx = _split_code_fences(text)
        for i, line in enumerate(text.split("\n"), 1):
            if i in fence_idx:
                assert "<span>$</span>" not in line, (
                    f"line {i} is inside a code fence but was wrapped; this "
                    "pollutes copy-pasted commands: " + line.strip()[:80]
                )

    def test_no_escape_hacks(self):
        r"""`\$` and `&#36;` do not work -- comrak decodes them before MathJax."""
        text = _text()
        assert "&#36;" not in text, (
            "HTML dollar entities are decoded to a bare '$' by comrak and do "
            "NOT survive to MathJax"
        )
        prose_idx, _ = _split_code_fences(text)
        for i, line in enumerate(text.split("\n"), 1):
            if i in prose_idx and "\\$" in line:
                pytest.fail(
                    f"line {i}: backslash-escaped dollar does not survive the "
                    "comrak pipeline; use <span>$</span> instead"
                )

    def test_price_figures_survive_as_literal_text(self):
        """The headline prices must be present and readable as plain text."""
        text = _text()
        plain = re.sub(r"<[^>]+>", "", text)  # strip markup -> what the user reads
        for phrase in (
            "clone-tier hardware gets you there around $350",
            "official units run ~$725",
            "The $350 hello world",
        ):
            assert phrase in plain, f"missing/lost in markup: {phrase!r}"

    def test_dollar_pairs_are_not_adjacent_in_one_text_node(self):
        """The mechanism: no two unwrapped `$` may share a text node.

        This is the actual invariant that keeps MathJax from pairing them, so
        assert it directly rather than only checking the symptom.
        """
        text = _text()
        prose_idx, _ = _split_code_fences(text)
        for i, line in enumerate(text.split("\n"), 1):
            if i not in prose_idx:
                continue
            # After removing wrapped dollars, any line with 2+ dollars would
            # have two delimiters in the same node -> math span.
            if line.replace("<span>$</span>", "").count("$") >= 2:
                pytest.fail(
                    f"line {i} has multiple bare dollars in one text node, "
                    "which MathJax will pair into a math span: " + line.strip()[:80]
                )
