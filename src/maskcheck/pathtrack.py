"""Incremental JSON path tracking over *partial* text.

This is what lets a per-token decoding signal be attributed to a schema
field. During generation the JSON is always incomplete, so a normal parser
is useless here -- it would raise on every prefix. This is a character
state machine that accepts any prefix of a JSON document and answers
"which path is the cursor inside right now?".

    >>> t = PathTracker(); t.feed('{"lines": [{"sku": "A')
    >>> t.path()
    'lines[0].sku'

Structural characters inside string values are ignored, which is the whole
reason this can't be done with bracket counting.
"""

from typing import List, Optional, Tuple

_WS = " \t\r\n"
_NUM_CHARS = "-+.eE0123456789"


# Where the cursor sits lexically. Displacement on structural and key tokens
# is uninteresting -- a correct grammar is *supposed* to pin those, so a high
# override rate there is the mask working, not the mask misfiring. Only value
# regions carry signal.
REGION_STRUCTURAL = "structural"
REGION_KEY = "key"
REGION_STRING = "string"
REGION_NUMBER = "number"
REGION_LITERAL = "literal"

VALUE_REGIONS = frozenset((REGION_STRING, REGION_NUMBER, REGION_LITERAL))


class _Frame(object):
    __slots__ = ("kind", "key", "idx", "state")

    def __init__(self, kind: str):
        self.kind = kind          # 'obj' | 'arr'
        self.key = None           # type: Optional[str]
        self.idx = -1
        self.state = "expect_key" if kind == "obj" else "expect_value"


class PathTracker(object):
    """Feed text in any chunking; `path()` is valid at every point."""

    def __init__(self):
        self.stack = []           # type: List[_Frame]
        self.in_string = False
        self.escape = False
        self.collecting_key = False
        self._buf = []            # type: List[str]
        self.in_number = False
        self.in_literal = False
        self.complete = False     # top-level value finished

    # ---- public -----------------------------------------------------
    def feed(self, text: str) -> "PathTracker":
        for ch in text:
            self._char(ch)
        return self

    def path(self) -> str:
        parts = []  # type: List[str]
        for f in self.stack:
            if f.kind == "obj":
                if f.key is not None and f.state != "expect_key":
                    parts.append(("." + f.key) if parts else f.key)
            else:
                if f.idx >= 0:
                    parts.append("[{}]".format(f.idx))
        return "".join(parts)

    def depth(self) -> int:
        return len(self.stack)

    def in_key(self) -> bool:
        """True while emitting an object key rather than a value.

        Displacement on key tokens is uninteresting -- the grammar is
        supposed to pin those -- so callers filter these steps out.
        """
        return self.collecting_key

    def region(self) -> str:
        """Lexical region of the cursor right now."""
        if self.in_string:
            return REGION_KEY if self.collecting_key else REGION_STRING
        if self.in_number:
            return REGION_NUMBER
        if self.in_literal:
            return REGION_LITERAL
        return REGION_STRUCTURAL

    def in_value(self) -> bool:
        return self.region() in VALUE_REGIONS

    # ---- internals --------------------------------------------------
    def _char(self, ch: str) -> None:
        if self.in_string:
            self._string_char(ch)
            return

        if self.in_number:
            if ch in _NUM_CHARS:
                return
            self.in_number = False
            self._value_done()
            self._char(ch)   # ch is structural; reprocess
            return

        if self.in_literal:
            if ch.isalpha():
                return
            self.in_literal = False
            self._value_done()
            self._char(ch)
            return

        if ch in _WS:
            return

        if not self.stack:
            if ch == "{":
                self.stack.append(_Frame("obj"))
            elif ch == "[":
                self.stack.append(_Frame("arr"))
            elif ch == '"':
                self.in_string = True
            elif ch in "-0123456789":
                self.in_number = True
            elif ch.isalpha():
                self.in_literal = True
            return

        f = self.stack[-1]
        if f.kind == "obj":
            if f.state == "expect_key":
                if ch == '"':
                    self.in_string = True
                    self.collecting_key = True
                    self._buf = []
                    f.state = "in_key"
                elif ch == "}":
                    self._pop()
            elif f.state == "expect_colon":
                if ch == ":":
                    f.state = "expect_value"
            elif f.state == "expect_value":
                self._start_value(ch)
            elif f.state == "expect_comma":
                if ch == ",":
                    f.state = "expect_key"
                    f.key = None
                elif ch == "}":
                    self._pop()
        else:
            if f.state == "expect_value":
                if ch == "]":
                    self._pop()
                else:
                    f.idx += 1
                    self._start_value(ch)
            elif f.state == "expect_comma":
                if ch == ",":
                    f.state = "expect_value"
                elif ch == "]":
                    self._pop()

    def _string_char(self, ch: str) -> None:
        if self.escape:
            self.escape = False
            if self.collecting_key:
                self._buf.append(ch)
            return
        if ch == "\\":
            self.escape = True
            return
        if ch == '"':
            self.in_string = False
            if self.collecting_key:
                self.collecting_key = False
                f = self.stack[-1]
                f.key = "".join(self._buf)
                f.state = "expect_colon"
                self._buf = []
            else:
                self._value_done()
            return
        if self.collecting_key:
            self._buf.append(ch)

    def _start_value(self, ch: str) -> None:
        if ch == "{":
            self.stack.append(_Frame("obj"))
        elif ch == "[":
            self.stack.append(_Frame("arr"))
        elif ch == '"':
            self.in_string = True
            self.collecting_key = False
        elif ch in "-0123456789":
            self.in_number = True
        elif ch.isalpha():
            self.in_literal = True

    def _value_done(self) -> None:
        if not self.stack:
            self.complete = True
            return
        self.stack[-1].state = "expect_comma"

    def _pop(self) -> None:
        self.stack.pop()
        self._value_done()


def path_at_offsets(text: str, offsets: List[int]) -> List[Tuple[str, str]]:
    """Paths at each (ascending) character offset -- one pass, not N.

    Returns (path, region) per offset. Used to attribute a whole
    generation's per-token signal to fields in O(len(text)) rather than
    re-parsing the prefix for every token.
    """
    t = PathTracker()
    out = []  # type: List[Tuple[str, str]]
    cursor = 0
    for off in offsets:
        if off < cursor:
            raise ValueError("offsets must be ascending")
        t.feed(text[cursor:off])
        cursor = off
        out.append((t.path(), t.region()))
    return out
