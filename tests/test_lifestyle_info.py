"""Run: python3 -I tests/test_lifestyle_info.py   (no dependencies)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lifestyle_info as li

def exam(title, rows=None, concl="", cols=("Region", "BMD", "T-score", "Z-score")):
    return {"exam_title": title, "exam_type": "dxa",
            "sections": [{"title": "r", "columns": list(cols), "rows": rows or []}],
            "conclusion": [{"item": "WHO Classification", "detail": concl}] if concl else []}

HIP = exam("DXA Αριστερό Ισχίο", [["Neck", "0.701", "-1.3", "-0.8"], ["Troch", "0.596", "-1.1", "-0.7"],
           ["Inter", "1.172", "0.5", "0.7"], ["Total", "0.951", "0.1", "0.4"], ["Ward's", "0.576", "-1.4", "-0.3"]], "Osteopenia")
SPINE = exam("DXA Οσφυϊκή", [["L1", "1", "0.5", "1"], ["Total", "1", "0.2", "0.8"]], "Normal")
NORMAL_HIP = exam("DXA Hip", [["Neck", "1", "-0.4", "0"], ["Ward's", "1", "-1.9", "0"], ["Total", "1", "0.1", "0"]], "Normal")
OSTEOPOROSIS = exam("DXA Spine", [["L1", "0.7", "−2,7", "-1"], ["Total", "0.8", "-2.6", "-1"]], "Osteoporosis")
NO_OP = exam("DXA Spine", [["Total", "1", "-0.5", "0"]], "no osteoporosis")
NOT_BONE = {"exam_title": "Complete blood count", "exam_type": "lab", "sections": [
    {"columns": ["x", "T-score"], "rows": [["a", "-3"]]}], "conclusion": []}
fails = []
def check(name, cond):
    if not cond:
        fails.append(name)

check("anna hip -> low", li.bone_level(HIP) == "low")
check("anna spine -> none", li.bone_level(SPINE) is None)
check("ward-only low is ignored", li.bone_level(NORMAL_HIP) is None)
check("T<=-2.5 -> osteoporosis", li.bone_level(OSTEOPOROSIS) == "osteoporosis")
check("'no osteoporosis' not triggered", li.bone_level(NO_OP) is None)
check("non-bone exam ignored", li.bone_level(NOT_BONE) is None)
check("worst wins", li.topics_for_exams([HIP, OSTEOPOROSIS]) == {"bone": "osteoporosis"})
check("nothing -> build None", li.build([SPINE, NOT_BONE]) is None)

for lang in ("el", "en"):
    sec = li.build([SPINE, HIP], lang)
    check(f"{lang} built", sec is not None)
    txt = " ".join(sec["blocks"][0]["lines"]).lower()
    check(f"{lang} no personal 'you should'", "you should" not in txt and "πρέπει να" not in txt)
    check(f"{lang} no doses", not any(w in txt for w in (" mg", " iu", " μg", "χιλιοστόγραμμα")))
    check(f"{lang} unreviewed flag", "not yet been reviewed" in sec["review"] or "δεν έχουν ακόμη" in sec["review"])
    check(f"{lang} osteoporosis extra line", li.build([OSTEOPOROSIS], lang)["blocks"][0]["lines"][0] in
          (li.EXTRA_OSTEOPOROSIS[lang],))
    h = li.render_html(sec, 3)
    check(f"{lang} html has doi links", "https://doi.org/10.1136/bmj-2025-088050" in h)

# every shown item cites a verified source; hidden items have none
for topic, items in li.ITEMS.items():
    for it in items:
        if it["sources"]:
            check(f"{it['id']} el+en text", bool(it["el"]) and bool(it["en"]))
            check(f"{it['id']} sources known", all(s in li.SOURCES for s in it["sources"]))
        else:
            check(f"{it['id']} hidden has no text", not it["el"] and not it["en"])
for k, s in li.SOURCES.items():
    check(f"{k} pmid/doi", s["pmid"].isdigit() and s["doi"].startswith("10."))

# supplements must never be recommended outright
sup = next(i for i in li.ITEMS["bone"] if i["id"] == "supplements")
check("kanis source verified", li.SOURCES["kanis2019"]["pmid"] == "30324412")
check("supplements text defers to doctor", "ρωτήστε" in sup["el"] and "ask your doctor" in sup["en"])

# off by default
os.environ.pop("LIFESTYLE_INFO", None); check("off by default", not li.enabled())
os.environ["LIFESTYLE_INFO"] = "on"; check("on via env", li.enabled())

# dossier integration
import dossier
d = {"patient": {"name": "ΑΝΝΑ"}, "meds": [], "exams": [SPINE, HIP], "points": []}
os.environ["LIFESTYLE_INFO"] = "off"
check("html off: no section", "lifestyle" not in dossier.build_html(d, "el"))
os.environ["LIFESTYLE_INFO"] = "on"
h = dossier.build_html(d, "el")
check("html on: section + heading", "lifestyle" in h and "Γενικές πληροφορίες για συζήτηση" in h)
check("html on: footer exception", "Εξαίρεση" in h)
check("docx builds with section", len(dossier.build_docx(d, "en")) > 1000)

print("FAIL:", fails) if fails else print("OK: lifestyle_info tests passed")
sys.exit(1 if fails else 0)
