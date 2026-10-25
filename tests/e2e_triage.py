"""End-to-end check of the chat safety floor with Streamlit's AppTest (no AI key needed).
Run: python3 tests/e2e_triage.py   (needs the app requirements)"""
import os, sys
os.environ.pop("TRIAGE_FLOOR", None)
for k in ("Claude_API_Key", "ANTHROPIC_API_KEY", "CLAUDE_API_KEY"):
    os.environ.pop(k, None)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from streamlit.testing.v1 import AppTest

def run(user_text, lang="en", extra=None):
    at = AppTest.from_file(os.path.join(ROOT, "kira_app.py"), default_timeout=90)
    at.session_state["lang"] = lang
    at.session_state["screen"] = "triage"
    at.session_state["_hero_seen"] = True
    at.session_state["_landing_seen"] = True
    at.session_state["profile"] = {"name": "Test", "age": 55, "sex": "M"}
    at.session_state["triage_chat"] = [{"role": "user", "content": user_text}] + (extra or [])
    at.run()
    assert not at.exception, [e.value[:200] for e in at.exception]
    return at

bad = 0
def check(name, ok):
    global bad
    print(("OK   " if ok else "FAIL ") + name)
    bad += (not ok)

# 1) clear emergency: fixed message, no AI call, banner flag, once per rule set
at = run("I have chest pain and it goes down my left arm, I am sweating")
chat = at.session_state["triage_chat"]
check("act-now: fixed 166/112 reply, bypass key set", len(chat) == 2 and chat[1].get("bypass") == "CHEST-ACS" and "166" in chat[1]["content"])
check("act-now: emergency flag set", bool(at.session_state["triage_emergency"]))
at.run()
check("act-now: not repeated on rerun", len(at.session_state["triage_chat"]) == 2)

# 2) Greek
at = run("Έχω πόνο στο στήθος που πάει στο αριστερό χέρι και ιδρώνω", lang="el")
c = at.session_state["triage_chat"]
check("act-now Greek: Greek message", len(c) == 2 and "ΕΚΑΒ" in c[1]["content"])

# 3) AI unavailable (no key) and nothing clear: urgent fallback, never a raw error
at = run("I have a mild headache since yesterday")
c = at.session_state["triage_chat"]
last = c[-1]["content"]
check("AI down: urgent fallback, no raw error", c[-1]["role"] == "assistant" and "urgent" in last.lower() and "Claude" not in last and "API key" not in last)

# 4) AI unavailable but URGENT rule: fallback carries the rule text
at = run("I have a pressure in my chest since this morning")
last = at.session_state["triage_chat"][-1]["content"]
check("AI down + URGENT rule: rule text included", "chest" in last.lower() and "166" in last)

# 5) kill switch: rules off, AI-failure fallback stays
os.environ["TRIAGE_FLOOR"] = "off"
at = run("I have chest pain and it goes down my left arm, I am sweating")
c = at.session_state["triage_chat"]
check("kill switch: no bypass, fallback still protects", not c[-1].get("bypass") and "166" in c[-1]["content"])
print("failures:", bad)
sys.exit(1 if bad else 0)
