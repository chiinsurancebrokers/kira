"""
ui_i18n.py — on-the-fly interface translation for languages that have no
hand-written strings (French, German, … and the partial ones).

How it works
------------
* The app is written in Greek and English. For any other interface language,
  the Streamlit text functions are wrapped (see install()) so every label,
  caption, button and HTML block goes through Translator.tr().
* A string seen for the first time is shown as-is and queued; at the end of
  the page the queue is translated in ONE batched Claude call and the page
  re-renders. Translations are cached in memory for every visitor and saved
  to the app's settings table, so each string is translated once per language.
* <style>/<script> blocks, HTML tags, markdown, emojis, numbers and the brand
  name are kept as they are.
* AI-written content (the chat, reports, analyses) is NOT passed through here —
  it is already produced in the chosen language. The app renders it inside
  `no_translate()`.
"""
from __future__ import annotations

import concurrent.futures as _cf
import hashlib
import json
import os
import re
import threading
import time
import urllib.request

MODEL = "claude-haiku-4-5-20251001"
API_URL = "https://api.anthropic.com/v1/messages"

_BLOCK_RE = re.compile(r"(<style\b.*?</style>|<script\b.*?</script>)", re.S | re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_ENT_RE = re.compile(r"&[a-zA-Z#0-9]+;")
_GREEK_RE = re.compile(r"[Ͱ-Ͽἀ-῿]")
_LETTER_RE = re.compile(r"[^\W\d_]", re.U)


def _visible(text: str) -> str:
    return _ENT_RE.sub(" ", _TAG_RE.sub(" ", text))


def needs_translation(seg: str, lang: str) -> bool:
    vis = _visible(seg)
    if not _LETTER_RE.search(vis):
        return False
    if lang == "en":                       # English UI: only fix stray Greek text
        return bool(_GREEK_RE.search(vis))
    return True


class Translator:
    def __init__(self, get_key, load_blob=None, save_blob=None, log=None):
        self._get_key = get_key
        self._load = load_blob
        self._save = save_blob
        self._log = log
        self.cache: dict[str, dict[str, str]] = {}
        self._loaded: set[str] = set()
        self._failed: dict[str, dict[str, float]] = {}
        self._lock = threading.Lock()

    # ── cache ────────────────────────────────────────────────────────────────
    def _ensure(self, lang: str):
        if lang in self._loaded:
            return
        with self._lock:
            if lang in self._loaded:
                return
            data = {}
            if self._load:
                try:
                    raw = self._load(lang)
                    if raw:
                        data = json.loads(raw) if isinstance(raw, str) else dict(raw)
                except Exception:
                    data = {}
            self.cache.setdefault(lang, {}).update({k: v for k, v in data.items() if isinstance(v, str)})
            self._loaded.add(lang)

    def tr(self, lang: str, text: str):
        """Return (translated_text, missing_segments). Missing segments are shown
        in the original language until they are translated."""
        if not isinstance(text, str) or not text:
            return text, []
        self._ensure(lang)
        c = self.cache.get(lang, {})
        failed = self._failed.get(lang, {})
        out, miss = [], []
        for part in _BLOCK_RE.split(text):
            if not part or _BLOCK_RE.fullmatch(part) or not needs_translation(part, lang):
                out.append(part)
                continue
            hit = c.get(part)
            if hit is None:
                if time.time() - failed.get(part, 0) > 300:
                    miss.append(part)
                out.append(part)
            else:
                out.append(hit)
        return "".join(out), miss

    # ── batch translation ────────────────────────────────────────────────────
    def _call(self, lang_name: str, items: dict[str, str]) -> dict[str, str]:
        if os.environ.get("ASK_I18N_FAKE"):            # local tests only: no API call
            return {k: "⟦" + v + "⟧" for k, v in items.items()}
        key = self._get_key()
        if not key:
            return {}
        system = (
            f"You translate the user interface of Asklepios, a health-information web app, into {lang_name}. "
            "Input: a JSON object of id → text (English or Greek; may contain HTML or markdown). "
            "Output: ONLY a JSON object with the same ids → the translation. Rules: translate only the "
            "human-readable words; keep every HTML tag, attribute, class, style, URL, markdown symbol "
            "(**, #, -, [text](url)), emoji, number, unit, placeholder like {x}, and line break exactly as given; "
            "keep the brand names Asklepios, Claude, PubMed, GDPR, HRV, VO2max, SpO₂ as they are; "
            "use the standard medical terms of the target language; keep it short and natural, the same tone "
            f"(friendly, plain); if a text is already in {lang_name}, return it unchanged. No comments."
        )
        body = json.dumps({
            "model": MODEL, "max_tokens": 8000, "temperature": 0,
            "system": system,
            "messages": [{"role": "user", "content": json.dumps(items, ensure_ascii=False)}],
        }).encode()
        req = urllib.request.Request(API_URL, data=body, headers={
            "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            resp = json.loads(r.read())
        txt = "".join(b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text").strip()
        txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt)
        m = re.search(r"\{.*\}", txt, re.S)
        data = json.loads(m.group(0)) if m else {}
        return {k: v for k, v in data.items() if isinstance(v, str)}

    def flush(self, lang: str, lang_name: str, segments, max_chars: int = 5000) -> int:
        """Translate the given segments (one batched call per ~5k characters,
        run in parallel). Returns how many were added to the cache."""
        self._ensure(lang)
        c = self.cache.setdefault(lang, {})
        todo = [s for s in dict.fromkeys(segments) if s not in c]
        if not todo:
            return 0
        chunks, cur, size = [], {}, 0
        for s in todo:
            if cur and size + len(s) > max_chars:
                chunks.append(cur); cur, size = {}, 0
            cur[hashlib.sha1(s.encode()).hexdigest()[:10]] = s
            size += len(s)
        if cur:
            chunks.append(cur)
        added = 0
        t0 = time.time()
        with _cf.ThreadPoolExecutor(max_workers=4) as ex:
            futs = {ex.submit(self._call, lang_name, ch): ch for ch in chunks}
            for f in _cf.as_completed(futs):
                ch = futs[f]
                try:
                    res = f.result()
                except Exception as e:
                    res = {}
                    if self._log:
                        self._log("ui_translate", ok=False, ms=(time.time() - t0) * 1000, error=str(e)[:200])
                for hid, src in ch.items():
                    dst = res.get(hid)
                    # the translation must keep exactly the same HTML tags, or it is not used
                    if dst and dst.strip() and _TAG_RE.findall(src) == _TAG_RE.findall(dst):
                        c[src] = dst
                        added += 1
                    else:
                        self._failed.setdefault(lang, {})[src] = time.time()
        if added and self._save:
            try:
                self._save(lang, json.dumps(c, ensure_ascii=False))
            except Exception:
                pass
        if self._log:
            self._log("ui_translate", ok=True, ms=(time.time() - t0) * 1000, lang=lang, n=added)
        return added


# ── Streamlit wrapping ─────────────────────────────────────────────────────────
# name → (positional indices to translate, keyword names to translate, options index or None)
_SPEC = {
    "markdown": ((0,), ("body", "help"), None),
    "caption": ((0,), ("body", "help"), None),
    "info": ((0,), ("body",), None),
    "success": ((0,), ("body",), None),
    "warning": ((0,), ("body",), None),
    "error": ((0,), ("body",), None),
    "title": ((0,), ("body",), None),
    "header": ((0,), ("body",), None),
    "subheader": ((0,), ("body",), None),
    "button": ((0,), ("label", "help"), None),
    "link_button": ((0,), ("label", "help"), None),
    "download_button": ((0,), ("label", "help"), None),
    "form_submit_button": ((0,), ("label", "help"), None),
    "checkbox": ((0,), ("label", "help"), None),
    "toggle": ((0,), ("label", "help"), None),
    "text_input": ((0,), ("label", "placeholder", "help"), None),
    "text_area": ((0,), ("label", "placeholder", "help"), None),
    "number_input": ((0,), ("label", "placeholder", "help"), None),
    "date_input": ((0,), ("label", "help"), None),
    "file_uploader": ((0,), ("label", "help"), None),
    "camera_input": ((0,), ("label", "help"), None),
    "expander": ((0,), ("label",), None),
    "popover": ((0,), ("label", "help"), None),
    "chat_input": ((0,), ("placeholder",), None),
    "selectbox": ((0,), ("label", "placeholder", "help"), 1),
    "radio": ((0,), ("label", "help"), 1),
    "multiselect": ((0,), ("label", "placeholder", "help"), 1),
    "segmented_control": ((0,), ("label", "help"), 1),
    "pills": ((0,), ("label", "help"), 1),
    "select_slider": ((0,), ("label", "help"), 1),
    "metric": ((0,), ("label", "help"), None),
}


def install(st, translate, *, enabled):
    """Wrap Streamlit's text functions so their strings go through
    translate(text) whenever enabled() is true."""
    from streamlit.delta_generator import DeltaGenerator

    def _wrap(name, fn, spec):
        pos, kws, opt_idx = spec

        def wrapper(*args, **kwargs):
            if not enabled():
                return fn(*args, **kwargs)
            # bound DeltaGenerator methods: args[0] is self
            off = 1 if args and isinstance(args[0], DeltaGenerator) else 0
            args = list(args)
            for i in pos:
                j = i + off
                if j < len(args) and isinstance(args[j], str):
                    args[j] = translate(args[j])
            for k in kws:
                if isinstance(kwargs.get(k), str):
                    kwargs[k] = translate(kwargs[k])
            if opt_idx is not None:
                ff = kwargs.get("format_func") or str
                kwargs["format_func"] = lambda o, _ff=ff: (lambda v: translate(v) if isinstance(v, str) else v)(_ff(o))
                if name == "radio" and isinstance(kwargs.get("captions"), (list, tuple)):
                    kwargs["captions"] = [translate(c) if isinstance(c, str) else c for c in kwargs["captions"]]
            return fn(*args, **kwargs)

        wrapper.__wrapped__ = fn
        wrapper.__name__ = name
        wrapper._ask_i18n = True
        return wrapper

    for name, spec in _SPEC.items():
        orig = getattr(DeltaGenerator, name, None)
        if orig is None or getattr(orig, "_ask_i18n", False):
            continue
        setattr(DeltaGenerator, name, _wrap(name, orig, spec))
        main = getattr(st, "_main", None)
        if main is not None and hasattr(st, name):
            try:
                setattr(st, name, getattr(main, name))
            except Exception:
                pass

    # module-level helpers
    for name in ("spinner", "toast"):
        orig = getattr(st, name, None)
        if orig is None or getattr(orig, "_ask_i18n", False):
            continue

        def mk(fn):
            def w(text=None, *a, **k):
                if enabled() and isinstance(text, str):
                    text = translate(text)
                return fn(text, *a, **k)
            w.__wrapped__ = fn
            w._ask_i18n = True
            return w
        setattr(st, name, mk(orig))

    # tabs: list of labels
    orig_tabs = DeltaGenerator.tabs
    if not getattr(orig_tabs, "_ask_i18n", False):
        def tabs(self, tabs, *a, **k):
            if enabled():
                tabs = [translate(x) if isinstance(x, str) else x for x in tabs]
            return orig_tabs(self, tabs, *a, **k)
        tabs.__wrapped__ = orig_tabs
        tabs._ask_i18n = True
        DeltaGenerator.tabs = tabs
        if getattr(st, "_main", None) is not None:
            st.tabs = st._main.tabs
