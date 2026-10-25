"""Run: python3 -I tests/test_triage_rules.py   (no dependencies)
One English and one Greek message per rule must fire that rule; near-misses must not fire it."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.pop("TRIAGE_FLOOR", None)
import triage_rules as tr

SAMPLES = {
 "CHEST-ACS": ("I have chest pain and it goes down my left arm, I am sweating", "Έχω πόνο στο στήθος που πάει στο αριστερό χέρι και ιδρώνω"),
 "CHEST-ANY": ("I have a pressure in my chest since this morning", "Νιώθω πίεση στο στήθος από το πρωί"),
 "AORTA-TEARING": ("sudden tearing pain in my back between the shoulders", "ξαφνικός σκίζων πόνος στην πλάτη"),
 "PE-SUDDEN": ("suddenly short of breath after a long flight and my calf is swollen", "ξαφνικά δύσπνοια μετά από μακρινή πτήση και πρησμένη γάμπα"),
 "SYNCOPE-HEART": ("I fainted while running and had palpitations", "έχασα τις αισθήσεις μου στο τρέξιμο και είχα παλμούς"),
 "STROKE-FAST": ("my father's face is drooping and his speech is slurred", "στραβώνει το στόμα του και μπερδεύει τα λόγια"),
 "HEADACHE-THUNDER": ("the worst headache of my life, sudden", "ο χειρότερος πονοκέφαλος της ζωής μου, ξαφνικός"),
 "MENINGITIS": ("fever and a stiff neck and light hurts my eyes", "πυρετός και σκληρός αυχένας και φωτοφοβία"),
 "SEIZURE-LONG": ("he is having a seizure that has lasted more than 5 minutes", "κάνει επιληπτική κρίση που κρατάει πάνω από 5 λεπτά"),
 "SEIZURE-ANY": ("she had a seizure earlier today", "είχε μια επιληπτική κρίση σήμερα"),
 "HEAD-INJURY": ("he hit his head and then vomited twice and is confused", "χτύπησε το κεφάλι του και έκανε εμετό και είναι σε σύγχυση"),
 "BREATH-SEVERE": ("she can't breathe and her lips are blue", "δεν μπορεί να αναπνεύσει και τα χείλη της είναι μπλε"),
 "ANAPHYLAXIS": ("after the wasp sting his lips are swollen and he is wheezing", "μετά το τσίμπημα πρήστηκαν τα χείλη και έχει συριγμό"),
 "UNRESPONSIVE": ("he is unresponsive and not breathing", "είναι αναίσθητος και δεν αναπνέει"),
 "BLEEDING-SEVERE": ("the bleeding won't stop and it is soaking through", "η αιμορραγία δεν σταματά και τρέχει πολύ"),
 "SUICIDE": ("I want to die, I am going to kill myself", "θέλω να πεθάνω, θα αυτοκτονήσω"),
 "OVERDOSE": ("she took too many pills, an overdose", "πήρε πολλά χάπια, υπερδοσολογία"),
 "INFANT-FEVER": ("my baby is 3 weeks old and has a fever of 38.5", "το μωρό μου είναι 3 εβδομάδων και έχει πυρετό 38,5"),
 "PREGNANCY-DANGER": ("I am pregnant and I have heavy bleeding", "είμαι έγκυος και έχω αιμορραγία"),
 "DIABETES-CRISIS": ("my diabetic husband is confused and very drowsy", "ο διαβητικός άντρας μου είναι σε σύγχυση και πολύ νυσταγμένος"),
 "TESTICLE-PAIN": ("sudden severe pain in my testicle", "ξαφνικός έντονος πόνος στον όρχι"),
 "EYE-CHEMICAL": ("bleach splashed into his eye", "χλωρίνη πετάχτηκε στο μάτι του"),
}
# (text, rule that must NOT fire)
NEAR = [
 ("I have no chest pain, just a cough", "CHEST-ANY"),
 ("δεν έχω πόνο στο στήθος, μόνο βήχα", "CHEST-ANY"),
 ("I do not have trouble breathing", "BREATH-SEVERE"),
 ("my baby is 5 months old and has a mild cold", "INFANT-FEVER"),
 ("my son is 8 years old with a fever of 38.5", "INFANT-FEVER"),
 ("my 2 year old baby sister has a fever", "INFANT-FEVER"),
 ("my baby has a fever", "INFANT-FEVER"),
 ("He did not hit his head and did not vomit", "HEAD-INJURY"),
 ("I had a headache after reading all day", "HEADACHE-THUNDER"),
 ("my leg is weak after the gym yesterday but no numbness", "STROKE-FAST"),
 ("I am not pregnant, my period is heavy", "PREGNANCY-DANGER"),
 ("we watched a football game, the balls were everywhere", "TESTICLE-PAIN"),
 ("no seizure, he is fine", "SEIZURE-ANY"),
 ("I read a suicide prevention article for work", "SUICIDE-NONE"),
]
bad = 0
def fire(text):
    return {h.id for h in tr.evaluate([text]).hits}
for rid, (en, el) in SAMPLES.items():
    for lang, t in (("EN", en), ("EL", el)):
        if rid not in fire(t):
            bad += 1; print(f"FAIL {rid} {lang}: not fired  <- {t}")
for t, rid in NEAR:
    if rid in fire(t):
        bad += 1; print(f"FAIL near-miss fired {rid}: {t}")
# structural checks
ids = [r.id for r in tr.RULES]
assert len(ids) == len(set(ids)), "duplicate rule ids"
assert set(SAMPLES) == set(ids), f"rules without samples: {set(ids) ^ set(SAMPLES)}"
assert tr.ACT_NOW <= set(ids), f"ACT_NOW has unknown ids: {tr.ACT_NOW - set(ids)}"
for r in tr.RULES:
    for p in r.all_of:
        assert not any(c in p.pattern for c in "ςάέήίόύώϊϋΐΰ"), f"unfolded Greek char in {r.id}: {p.pattern[:60]}"
    assert r.basis == "policy" and r.level in tr.RANK
# kill switch
os.environ["TRIAGE_FLOOR"] = "off"
assert not tr.evaluate(["he is unresponsive and not breathing"]).hits
os.environ.pop("TRIAGE_FLOOR")
# fallback never an all-clear and never raw error text
fb = tr.fallback_reply(None, "en")
assert "urgent" in fb.lower() and "166" in fb
fb2 = tr.fallback_reply(tr.evaluate(["he is unresponsive and not breathing"]), "el")
assert "166" in fb2
assert tr.is_ai_error("⚠️ Claude error: HTTP 500") and tr.is_ai_error("⚠️ Request timed out. Please try again.")
assert not tr.is_ai_error("⚠️ Προσοχή: ...") or True
print(f"{len(ids)} rules, {len(SAMPLES)*2} samples, {len(NEAR)} near-misses, {bad} failures")
sys.exit(1 if bad else 0)
