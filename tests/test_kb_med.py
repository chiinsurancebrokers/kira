"""Tests for kb_med.py (no deps):  python3 -I tests/test_kb_med.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kb, kb_med

fails = 0
def check(name, cond):
    global fails; fails += (not cond); print(("OK   " if cond else "FAIL ") + name)
def ids(t, alert=()): return [f.chunk.id for f in kb_med.retrieve([t], alert)[1]]

cs = kb_med.chunks()
check("corpus: unique ids, text and label on every chunk", len({c.id for c in cs}) == len(cs) and all(c.text.strip() and c.title for c in cs))
check("corpus: one chunk per triage rule", sum(c.id.startswith("rule:") for c in cs) == 22)
check("corpus: every rule chunk names a published source", all(c.source.startswith("https://") for c in cs if c.id.startswith("rule:")))
check("corpus: no lifestyle chunk without a verified source", all(c.source for c in cs if c.id.startswith("life:")))
check("corpus: lifestyle chunks are unreviewed while REVIEWED_BY is empty", all(not c.reviewed for c in cs if c.id.startswith("life:")))

check("chest pain + sweating (EL)", "rule:CHEST-ACS" in ids("πονάει το στήθος και ιδρώνω και ναυτία"))
check("chest pain (EN, everyday wording)", "rule:CHEST-ACS" in ids("my chest hurts and my left arm feels numb"))
check("face and speech (EL) -> stroke", "rule:STROKE-FAST" in ids("μου μούδιασε το πρόσωπο και δεν μιλάω καλά"))
check("infant fever (EL)", "rule:INFANT-FEVER" in ids("η μικρή μου έχει πυρετό 38.5 και είναι 2 μηνών"))
check("osteoporosis + exercise (EL) -> lifestyle text", "life:bone:exercise" in ids("έχω οστεοπόρωση, τι άσκηση να κάνω"))
check("osteoporosis + exercise (EN) -> lifestyle text", "life:bone:exercise" in ids("does osteoporosis need exercise"))
check("unrelated message retrieves nothing", ids("ποιος είναι ο πρωθυπουργός") == [])
check("a lone common symptom word retrieves nothing", ids("έχω πονοκέφαλο") == [])
check("rules that already fired are not repeated", "rule:CHEST-ACS" not in ids("πονάει το στήθος και ιδρώνω και ναυτία", alert=("CHEST-ACS",)))

blk, found = kb_med.retrieve(["πονάει το στήθος και ιδρώνω"])
check("block: tags + 'pending clinician review' for policy rules", "[S1]" in blk and "pending clinician review" in blk)
out = kb_med.cite("Κάλεσε το 166 [S1]. Μην περιμένεις [S8].", found, "el")
check("cite: tag removed, source listed once, invented tag dropped", "[S" not in out and out.count("Πηγές:") == 1)
os.environ["KB_RAG"] = "off"
check("KB_RAG=off retrieves nothing", kb_med.retrieve(["πονάει το στήθος και ιδρώνω"]) == ("", []))
print(f"\n{fails} failed"); sys.exit(1 if fails else 0)
