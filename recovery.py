"""
recovery.py — "Αποκατάσταση στο σπίτι" (post-discharge home monitoring) for Asklepios.

Pure logic, no UI and no network except `extract_discharge`.  Built for the
first days at home after a serious hospital stay (first use case: TAVI).

Design rules
------------
* The red-flag engine is DETERMINISTIC code.  No language model decides whether
  something is an emergency.
* Every numeric threshold lives in `plan["thresholds"]` together with
  `plan["thresholds_source"]`.  The built-in values are generic starting values
  taken from patient-discharge material of structural-heart programmes
  (University of Washington, Cleveland Clinic, Kettering Health TAVR sheets) and
  are labelled "to confirm with the treating cardiologist".  Anything the
  discharge letter states explicitly overrides them (`apply_letter_thresholds`).
* The module never diagnoses and never advises treatment.  It only (a) tells the
  carer when to call 112/166 or the care team, (b) keeps an organised record,
  and (c) prints that record for the clinician.

Levels:  3 EMERGENCY (call 166/112)  >  2 CALL (call the care team today/now)
         >  1 WATCH (recheck soon)   >  0 OK
"""
from __future__ import annotations

import html as _html
import io
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo("Europe/Athens")
except Exception:  # pragma: no cover
    TZ = timezone(timedelta(hours=3))

EMERGENCY, CALL, WATCH, OK = 3, 2, 1, 0
LEVEL_NAME = {3: "emergency", 2: "call", 1: "watch", 0: "ok"}
LEVEL_COLOR = {3: "#DC2626", 2: "#EA580C", 1: "#CA8A04", 0: "#059669"}
LEVEL_LABEL = {
    "el": {3: "ΕΠΕΙΓΟΝ — καλέστε 166/112", 2: "ΕΠΙΚΟΙΝΩΝΗΣΤΕ ΜΕ ΤΗΝ ΟΜΑΔΑ ΣΗΜΕΡΑ", 1: "ΠΑΡΑΚΟΛΟΥΘΗΣΗ", 0: "ΣΕ ΕΥΡΩΣΤΑ ΠΛΑΙΣΙΑ"},
    "en": {3: "EMERGENCY — call 166/112", 2: "CONTACT THE CARE TEAM TODAY", 1: "KEEP WATCHING", 0: "WITHIN EXPECTED RANGE"},
}

# ── Procedure templates ───────────────────────────────────────────────────────
# Defaults are *starting values to be confirmed by the treating team*.
# temp: 101 °F (Kettering/Cleveland) = 38.3 °C.  weight: "2–3 lb overnight or
# 5 lb in a week" (Kettering) ≈ 1.0 kg/day, 2.3 kg/7 days.  HR > 120 (Cleveland).
DEFAULT_THRESHOLDS = {
    "hr_low_call": 50, "hr_low_emergency": 40,
    "hr_high_call": 110, "hr_high_emergency": 120,   # emergency only with breathlessness/chest/dizzy symptoms
    "sbp_low_call": 100, "sbp_low_emergency": 85,
    "sbp_high_call": 180, "dbp_high_call": 110,
    "spo2_call": 93, "spo2_emergency": 90,
    "temp_call": 38.3, "temp_low_call": 35.5,
    "weight_gain_day_kg": 1.0, "weight_gain_week_kg": 2.3, "weight_loss_3d_kg": 2.0,
    "orthostatic_sbp_drop": 20,
    "bp_unstable_count_24h": 3, "bp_swing_mmhg": 40,
    "max_gap_hours": 10,
}
THRESHOLD_BOUNDS = {  # plausibility limits when a value is taken from a letter / typed
    "hr_low_call": (30, 70), "hr_low_emergency": (25, 55),
    "hr_high_call": (90, 160), "hr_high_emergency": (100, 200),
    "sbp_low_call": (70, 120), "sbp_low_emergency": (60, 100),
    "sbp_high_call": (140, 230), "dbp_high_call": (90, 130),
    "spo2_call": (85, 97), "spo2_emergency": (80, 94),
    "temp_call": (37.5, 39.5), "temp_low_call": (34.0, 36.5),
    "weight_gain_day_kg": (0.3, 3.0), "weight_gain_week_kg": (0.5, 6.0), "weight_loss_3d_kg": (0.5, 6.0),
    "orthostatic_sbp_drop": (10, 40),
    "bp_unstable_count_24h": (2, 8), "bp_swing_mmhg": (20, 80),
    "max_gap_hours": (3, 24),
}
THRESHOLD_LABELS = {  # (el, en, unit)
    "hr_low_call": ("Σφυγμοί: χαμηλό — επικοινωνία με ομάδα", "Pulse low — contact team", "bpm"),
    "hr_low_emergency": ("Σφυγμοί: χαμηλό — επείγον", "Pulse low — emergency", "bpm"),
    "hr_high_call": ("Σφυγμοί: υψηλό — επικοινωνία με ομάδα", "Pulse high — contact team", "bpm"),
    "hr_high_emergency": ("Σφυγμοί: υψηλό — επείγον", "Pulse high — emergency", "bpm"),
    "sbp_low_call": ("Συστολική: χαμηλή — επικοινωνία", "Systolic low — contact team", "mmHg"),
    "sbp_low_emergency": ("Συστολική: χαμηλή — επείγον", "Systolic low — emergency", "mmHg"),
    "sbp_high_call": ("Συστολική: υψηλή — επικοινωνία", "Systolic high — contact team", "mmHg"),
    "dbp_high_call": ("Διαστολική: υψηλή — επικοινωνία", "Diastolic high — contact team", "mmHg"),
    "spo2_call": ("SpO₂ — επικοινωνία", "SpO₂ — contact team", "%"),
    "spo2_emergency": ("SpO₂ — επείγον", "SpO₂ — emergency", "%"),
    "temp_call": ("Θερμοκρασία — επικοινωνία", "Temperature — contact team", "°C"),
    "temp_low_call": ("Χαμηλή θερμοκρασία", "Low temperature", "°C"),
    "weight_gain_day_kg": ("Αύξηση βάρους σε 1 ημέρα", "Weight gain in 1 day", "kg"),
    "weight_gain_week_kg": ("Αύξηση βάρους σε 7 ημέρες", "Weight gain in 7 days", "kg"),
    "weight_loss_3d_kg": ("Απώλεια βάρους σε 3 ημέρες", "Weight loss in 3 days", "kg"),
    "orthostatic_sbp_drop": ("Πτώση συστολικής όρθια", "Systolic drop on standing", "mmHg"),
    "bp_unstable_count_24h": ("Παρατηρήσεις εκτός ορίων ΑΠ σε 24ω", "Out-of-range BP readings in 24 h", "×"),
    "bp_swing_mmhg": ("Διακύμανση συστολικής μεταξύ μετρήσεων", "Systolic swing between readings", "mmHg"),
    "max_gap_hours": ("Μέγιστο κενό μεταξύ καταγραφών", "Longest gap between check-ins", "ώρες/h"),
}

PROCEDURES = {
    "tavi": {
        "el": "TAVI (διακαθετήρια τοποθέτηση αορτικής βαλβίδας)",
        "en": "TAVI (transcatheter aortic valve implantation)",
        "checkins_per_day": 3, "intensive_days": 3,
        "watch_extra": ["access_site", "conduction"],
    },
    "icu_general": {
        "el": "Έξοδος από ΜΕΘ / νοσηλεία για σοβαρό πρόβλημα",
        "en": "Discharge after ICU / serious hospital stay",
        "checkins_per_day": 3, "intensive_days": 3, "watch_extra": [],
    },
    "other": {"el": "Άλλη επέμβαση / νοσηλεία", "en": "Other procedure / admission",
              "checkins_per_day": 2, "intensive_days": 2, "watch_extra": []},
}

# ── Symptom catalogue ─────────────────────────────────────────────────────────
# key: (level, group, el, en, only_if_procedure_or_None)
SYMPTOMS = {
    # emergencies
    "stroke_signs":   (EMERGENCY, "neuro", "Πεσμένο πρόσωπο / αδυναμία χεριού ή ποδιού / δυσκολία στην ομιλία / απότομη αλλαγή όρασης",
                       "Face drooping / arm or leg weakness / speech difficulty / sudden vision change"),
    "syncope":        (EMERGENCY, "heart", "Λιποθυμία (έχασε τις αισθήσεις της)", "Fainted (lost consciousness)"),
    "chest_pain":     (EMERGENCY, "heart", "Πόνος ή πίεση στο στήθος που δεν περνά με ανάπαυση", "Chest pain/pressure not settling with rest"),
    "breath_rest":    (EMERGENCY, "breath", "Δύσπνοια σε ηρεμία / δεν μπορεί να πει πρόταση", "Breathless at rest / cannot speak a sentence"),
    "bleeding_heavy": (EMERGENCY, "site", "Αιμορραγία που δεν σταματά μετά από 10–20' πίεση ή αιματέμεση/μαύρα κόπρανα",
                       "Bleeding not stopping after 10–20 min pressure, or vomiting blood/black stools"),
    "limb_cold":      (EMERGENCY, "site", "Κρύο, χλωμό ή μουδιασμένο πόδι/χέρι (στο σημείο εισόδου)", "Cold, pale or numb leg/arm (access side)"),
    "fall_head":      (EMERGENCY, "fall", "Πτώση με χτύπημα στο κεφάλι (παίρνει αντιαιμοπεταλιακά)", "Fall with head strike (on antiplatelets)"),
    # call the care team
    "presyncope":     (CALL, "heart", "Λιποθυμική τάση / ζάλη όρθια", "Near-faint / dizziness on standing"),
    "palpitations":   (CALL, "heart", "Αίσθημα παλμών", "Palpitations"),
    "irregular_pulse":(CALL, "heart", "Νέος ακανόνιστος σφυγμός", "New irregular pulse"),
    "breath_worse":   (CALL, "breath", "Περισσότερη δύσπνοια από όταν βγήκε", "More breathless than at discharge"),
    "orthopnea":      (CALL, "breath", "Χρειάζεται περισσότερα μαξιλάρια / ξυπνά με δύσπνοια", "Needs more pillows / wakes breathless"),
    "leg_swelling":   (CALL, "fluid", "Νέο πρήξιμο στους αστραγάλους/πόδια", "New ankle/leg swelling"),
    "site_swelling":  (CALL, "site", "Πρήξιμο, μελανιά που μεγαλώνει ή «σβώλος» στο σημείο εισόδου", "Swelling, growing bruise or lump at access site"),
    "site_infection": (CALL, "site", "Ερυθρότητα, θερμότητα ή έκκριμα στο σημείο εισόδου", "Redness, warmth or discharge at access site"),
    "site_pain":      (CALL, "site", "Έντονος ή αυξανόμενος πόνος στο σημείο εισόδου", "Severe or increasing pain at access site"),
    "confusion":      (CALL, "neuro", "Νέα σύγχυση / υπερβολική νύστα", "New confusion / unusual drowsiness"),
    "fall":           (CALL, "fall", "Πτώση (χωρίς χτύπημα στο κεφάλι)", "Fall (no head strike)"),
    "chills":         (CALL, "infection", "Ρίγος / κρυάδες", "Chills / rigors"),
    # watch
    "cough_worse":    (WATCH, "breath", "Βήχας που χειροτερεύει / αλλαγή χρώματος πτυέλων", "Worsening cough / change in sputum colour"),
    "poor_intake":    (WATCH, "fluid", "Δεν τρώει/πίνει αρκετά, ναυτία ή εμετός", "Not eating/drinking enough, nausea or vomiting"),
    "urine_low":      (WATCH, "fluid", "Ελάχιστα ούρα σήμερα", "Very little urine today"),
    "fatigue":        (WATCH, "general", "Ασυνήθιστη κόπωση", "Unusual fatigue"),
    "poor_sleep":     (WATCH, "general", "Κακός ύπνος / άγχος", "Poor sleep / anxiety"),
}
# These symptoms upgrade a high/low pulse to an emergency.
_SERIOUS_WITH_NUMBERS = ("chest_pain", "breath_rest", "breath_worse", "presyncope", "syncope", "confusion", "stroke_signs")

SYMPTOM_GROUPS = [
    ("heart", "Καρδιά / ζάλη", "Heart / dizziness"),
    ("breath", "Αναπνοή", "Breathing"),
    ("site", "Σημείο εισόδου (βουβώνας / καρπός)", "Access site (groin / wrist)"),
    ("neuro", "Νευρολογικά", "Neurological"),
    ("fluid", "Υγρά / πρόσληψη", "Fluids / intake"),
    ("fall", "Πτώσεις", "Falls"),
    ("infection", "Λοίμωξη", "Infection"),
    ("general", "Γενικά", "General"),
]

CRITICAL_MED_WORDS = (  # missed dose → WATCH (never "double up" advice)
    "aspirin", "ασπιριν", "salospir", "clopidogrel", "plavix", "ticagrelor", "brilique",
    "prasugrel", "apixaban", "eliquis", "rivaroxaban", "xarelto", "dabigatran", "pradaxa",
    "warfarin", "sintrom", "edoxaban", "lixiana", "bisoprolol", "concor", "metoprolol",
)


NOTIFY_DEFAULT = {"emails": [], "min_level": CALL, "lang": "el", "detail": True,
                  "last_level": 0, "last_ts": None}
MAX_NOTIFY_EMAILS = 3
NOTIFY_QUIET_MIN = 15          # same-or-lower level alerts are not re-sent within this window


# ── Data helpers ──────────────────────────────────────────────────────────────
def new_plan(patient_name="", age=None, sex="", procedure="tavi", discharge_date=None) -> dict:
    proc = PROCEDURES.get(procedure, PROCEDURES["other"])
    return {
        "id": uuid.uuid4().hex[:12],
        "patient": {"name": patient_name, "age": age, "sex": sex},
        "procedure": procedure,
        "procedure_label": proc["en"],
        "discharge_date": discharge_date or datetime.now(TZ).date().isoformat(),
        "hospital_course": "",          # free text, e.g. "ICU, ventilator weaned, BP unstable"
        "baseline": {"hr_usual": None, "hr_min_known": None, "sbp_usual": None, "spo2_usual": None},
        "thresholds": dict(DEFAULT_THRESHOLDS),
        "thresholds_source": "default",  # "default" | "clinician"
        "thresholds_note": "",
        "checkins_per_day": proc["checkins_per_day"],
        "intensive_days": proc["intensive_days"],
        "meds": [],        # {"name","dose","times":[...],"critical":bool,"note":str}
        "restrictions": [],
        "followups": [],   # {"label","date","place"}
        "contacts": [],    # {"label","phone"}
        "letter_red_flags": [],
        "notify": dict(NOTIFY_DEFAULT, emails=[]),
        "created": datetime.now(TZ).isoformat(timespec="seconds"),
    }


def is_critical_med(name: str) -> bool:
    n = (name or "").lower()
    return any(w in n for w in CRITICAL_MED_WORDS)


def _f(v):
    try:
        if v is None or v == "":
            return None
        return float(str(v).replace(",", "."))
    except Exception:
        return None


def new_entry(**kw) -> dict:
    e = {
        "id": uuid.uuid4().hex[:10],
        "ts": datetime.now(TZ).isoformat(timespec="seconds"),
        "hr": None, "sbp": None, "dbp": None, "spo2": None, "temp": None, "weight": None,
        "sbp_stand": None, "dbp_stand": None,
        "symptoms": [],          # list of SYMPTOMS keys
        "meds_missed": [],       # names
        "notes": "", "by": "",
    }
    e.update(kw)
    return normalize_entry(e)


def normalize_entry(e: dict) -> dict:
    e = dict(e)
    for k in ("hr", "sbp", "dbp", "spo2", "sbp_stand", "dbp_stand"):
        v = _f(e.get(k)); e[k] = int(round(v)) if v is not None else None
    for k in ("temp", "weight"):
        v = _f(e.get(k)); e[k] = round(v, 1) if v is not None else None
    e["symptoms"] = [s for s in (e.get("symptoms") or []) if s in SYMPTOMS]
    e["meds_missed"] = list(e.get("meds_missed") or [])
    return e


def validate_entry(e: dict) -> list:
    """Typing-error guard (not clinical). Returns list of (field, message_en, message_el)."""
    bad = []
    rng = {"hr": (20, 250), "sbp": (50, 260), "dbp": (25, 160), "spo2": (50, 100),
           "temp": (32.0, 43.0), "weight": (25.0, 250.0), "sbp_stand": (50, 260), "dbp_stand": (25, 160)}
    for k, (lo, hi) in rng.items():
        v = e.get(k)
        if v is not None and not (lo <= v <= hi):
            bad.append((k, f"{k} {v} looks like a typing error (expected {lo}–{hi}).",
                        f"Η τιμή {k} = {v} μοιάζει με λάθος πληκτρολόγησης (αναμενόμενο {lo}–{hi})."))
    if e.get("sbp") and e.get("dbp") and e["dbp"] >= e["sbp"]:
        bad.append(("dbp", "Diastolic is not lower than systolic.", "Η διαστολική δεν είναι μικρότερη της συστολικής."))
    return bad


def _dt(ts):
    d = datetime.fromisoformat(ts)
    return d if d.tzinfo else d.replace(tzinfo=TZ)


def sort_entries(entries):
    return sorted(entries, key=lambda e: _dt(e["ts"]))


def day_number(plan, now=None) -> int:
    """Day 1 = discharge day."""
    now = now or datetime.now(TZ)
    d0 = datetime.fromisoformat(plan["discharge_date"]).date()
    return (now.astimezone(TZ).date() - d0).days + 1


# ── Alerts ────────────────────────────────────────────────────────────────────
@dataclass
class Alert:
    level: int
    code: str
    title: dict            # {"el","en"}
    detail: dict
    action: dict
    ts: str = ""
    values: dict = field(default_factory=dict)

    def to_dict(self):
        return {"level": self.level, "code": self.code, "title": self.title, "detail": self.detail,
                "action": self.action, "ts": self.ts, "values": self.values}


def _a(level, code, el_t, en_t, el_d, en_d, plan, ts="", values=None):
    return Alert(level, code, {"el": el_t, "en": en_t}, {"el": el_d, "en": en_d},
                 _action(level, plan), ts, values or {})


def _contact_line(plan, lang):
    cs = [c for c in plan.get("contacts", []) if c.get("phone")]
    if not cs:
        return ""
    s = "; ".join(f'{c.get("label","")}: {c["phone"]}' for c in cs[:3])
    return (" Επαφές: " if lang == "el" else " Contacts: ") + s


def _action(level, plan):
    name = (plan.get("patient") or {}).get("name") or ""
    proc = PROCEDURES.get(plan.get("procedure"), PROCEDURES["other"])
    dn = plan.get("discharge_date", "")
    if level == EMERGENCY:
        return {
            "el": (f"Καλέστε ΤΩΡΑ 166 (ΕΚΑΒ) ή 112. Μην οδηγήσετε μόνοι σας. Πείτε: «{name}, μετά από {proc['el']}, "
                   f"έξοδος από το νοσοκομείο {dn}»."),
            "en": (f"Call 166 (EKAB) or 112 NOW. Do not drive yourself. Say: “{name}, after {proc['en']}, "
                   f"discharged {dn}”."),
        }
    if level == CALL:
        return {
            "el": "Τηλεφωνήστε στην ομάδα που την παρακολουθεί (καρδιολόγος/κλινική) τώρα — όχι αύριο. Αν δεν απαντήσουν και χειροτερεύει, καλέστε 166/112." + _contact_line(plan, "el"),
            "en": "Phone her care team (cardiologist/clinic) now — not tomorrow. If they cannot be reached and she worsens, call 166/112." + _contact_line(plan, "en"),
        }
    if level == WATCH:
        return {"el": "Ξαναμετρήστε σε 30–60 λεπτά και καταγράψτε ξανά. Αν επιμένει ή χειροτερεύει, επικοινωνήστε με την ομάδα.",
                "en": "Re-measure in 30–60 minutes and log again. If it persists or worsens, contact the care team."}
    return {"el": "", "en": ""}


def evaluate_entry(plan: dict, entry: dict, history: list) -> list:
    """Alerts for ONE entry.  history = earlier entries (sorted, oldest first)."""
    th = {**DEFAULT_THRESHOLDS, **(plan.get("thresholds") or {})}
    out, ts = [], entry["ts"]
    syms = set(entry.get("symptoms") or [])
    hr, sbp, dbp = entry.get("hr"), entry.get("sbp"), entry.get("dbp")
    spo2, temp, wt = entry.get("spo2"), entry.get("temp"), entry.get("weight")
    serious = bool(syms & set(_SERIOUS_WITH_NUMBERS))

    # 1. symptoms (catalogue)
    for key in syms:
        lvl, _g, el, en = SYMPTOMS[key]
        out.append(_a(lvl, f"sym_{key}", el, en, "Καταγράφηκε από τον φροντιστή.", "Reported by the carer.", plan, ts, {"symptom": key}))

    # 2. pulse
    if hr is not None:
        if hr < th["hr_low_emergency"]:
            out.append(_a(EMERGENCY, "hr_low_emergency", f"Σφυγμοί {hr}/λεπτό — πολύ χαμηλοί", f"Pulse {hr}/min — very low",
                          f"Κάτω από το όριο {th['hr_low_emergency']}. Μετά από TAVI οι διαταραχές αγωγιμότητας είναι πιθανές τις πρώτες ημέρες.",
                          f"Below the limit of {th['hr_low_emergency']}. Conduction problems are possible in the first days after TAVI.",
                          plan, ts, {"hr": hr}))
        elif hr < th["hr_low_call"]:
            lvl = EMERGENCY if serious else CALL
            out.append(_a(lvl, "hr_low", f"Σφυγμοί {hr}/λεπτό — χαμηλοί", f"Pulse {hr}/min — low",
                          f"Κάτω από {th['hr_low_call']}" + (" και με συμπτώματα." if serious else "."),
                          f"Below {th['hr_low_call']}" + (" with symptoms." if serious else "."), plan, ts, {"hr": hr}))
        if hr > th["hr_high_emergency"]:
            lvl = EMERGENCY if (serious or "palpitations" in syms or "irregular_pulse" in syms) else CALL
            out.append(_a(lvl, "hr_high_emergency", f"Σφυγμοί {hr}/λεπτό — πολύ υψηλοί", f"Pulse {hr}/min — very high",
                          f"Πάνω από {th['hr_high_emergency']}" + (" με συμπτώματα." if lvl == EMERGENCY else " — χωρίς συμπτώματα, αλλά απαιτεί επικοινωνία."),
                          f"Above {th['hr_high_emergency']}" + (" with symptoms." if lvl == EMERGENCY else " — no symptoms reported, but needs a call."),
                          plan, ts, {"hr": hr}))
        elif hr > th["hr_high_call"]:
            out.append(_a(CALL, "hr_high", f"Σφυγμοί {hr}/λεπτό — υψηλοί", f"Pulse {hr}/min — high",
                          f"Πάνω από {th['hr_high_call']}.", f"Above {th['hr_high_call']}.", plan, ts, {"hr": hr}))

    # 3. blood pressure
    if sbp is not None:
        if sbp < th["sbp_low_emergency"]:
            lvl = EMERGENCY
            out.append(_a(lvl, "sbp_low_emergency", f"Πίεση {sbp}/{dbp or '?'} — πολύ χαμηλή", f"BP {sbp}/{dbp or '?'} — very low",
                          f"Συστολική κάτω από {th['sbp_low_emergency']}.", f"Systolic below {th['sbp_low_emergency']}.", plan, ts, {"sbp": sbp}))
        elif sbp < th["sbp_low_call"]:
            lvl = EMERGENCY if serious else CALL
            out.append(_a(lvl, "sbp_low", f"Πίεση {sbp}/{dbp or '?'} — χαμηλή", f"BP {sbp}/{dbp or '?'} — low",
                          f"Συστολική κάτω από {th['sbp_low_call']}" + (" και με συμπτώματα." if serious else "."),
                          f"Systolic below {th['sbp_low_call']}" + (" with symptoms." if serious else "."), plan, ts, {"sbp": sbp}))
        if sbp >= th["sbp_high_call"] or (dbp is not None and dbp >= th["dbp_high_call"]):
            lvl = EMERGENCY if serious else CALL
            out.append(_a(lvl, "bp_high", f"Πίεση {sbp}/{dbp or '?'} — υψηλή", f"BP {sbp}/{dbp or '?'} — high",
                          "Πάνω από το όριο" + (" και με συμπτώματα." if serious else "."),
                          "Above the limit" + (" with symptoms." if serious else "."), plan, ts, {"sbp": sbp, "dbp": dbp}))
        ss = entry.get("sbp_stand")
        if ss is not None and sbp - ss >= th["orthostatic_sbp_drop"]:
            out.append(_a(CALL, "orthostatic", f"Πτώση πίεσης όρθια: {sbp}→{ss}", f"BP drops on standing: {sbp}→{ss}",
                          f"Πτώση ≥ {th['orthostatic_sbp_drop']} mmHg — κίνδυνος πτώσης (λαμβάνει διουρητικό και αντιυπερτασικά).",
                          f"Drop ≥ {th['orthostatic_sbp_drop']} mmHg — fall risk.", plan, ts, {"sbp": sbp, "sbp_stand": ss}))
        # swing vs previous reading within 6 h
        prev = next((h for h in reversed(history) if h.get("sbp") is not None), None)
        if prev and abs(_dt(ts) - _dt(prev["ts"])) <= timedelta(hours=6) and abs(sbp - prev["sbp"]) >= th["bp_swing_mmhg"]:
            out.append(_a(WATCH, "bp_swing", f"Διακύμανση πίεσης {prev['sbp']}→{sbp}", f"BP swing {prev['sbp']}→{sbp}",
                          "Μεγάλη μεταβολή σε λίγες ώρες.", "Large change within a few hours.", plan, ts, {"sbp": sbp, "prev": prev["sbp"]}))

    # 4. oxygen
    if spo2 is not None:
        if spo2 < th["spo2_emergency"]:
            out.append(_a(EMERGENCY, "spo2_emergency", f"SpO₂ {spo2}%", f"SpO₂ {spo2}%",
                          f"Κάτω από {th['spo2_emergency']}%.", f"Below {th['spo2_emergency']}%.", plan, ts, {"spo2": spo2}))
        elif spo2 < th["spo2_call"]:
            lvl = EMERGENCY if serious else CALL
            out.append(_a(lvl, "spo2_low", f"SpO₂ {spo2}%", f"SpO₂ {spo2}%",
                          f"Κάτω από {th['spo2_call']}% (έχει ΧΑΠ/καπνίστρια: η ομάδα μπορεί να ορίσει χαμηλότερο όριο).",
                          f"Below {th['spo2_call']}% (COPD/smoker: the team may set a lower limit).", plan, ts, {"spo2": spo2}))

    # 5. temperature
    if temp is not None:
        if temp >= th["temp_call"]:
            out.append(_a(CALL, "fever", f"Πυρετός {temp}°C", f"Fever {temp}°C",
                          f"≥ {th['temp_call']}°C — μετά από επέμβαση βαλβίδας απαιτεί επικοινωνία με την ομάδα.",
                          f"≥ {th['temp_call']}°C — needs a call to the team after a valve procedure.", plan, ts, {"temp": temp}))
        elif temp <= th["temp_low_call"]:
            out.append(_a(CALL, "hypothermia", f"Χαμηλή θερμοκρασία {temp}°C", f"Low temperature {temp}°C",
                          "", "", plan, ts, {"temp": temp}))

    # 6. weight (fluid)
    if wt is not None:
        pts = [(h["ts"], h["weight"]) for h in history if h.get("weight") is not None]
        if pts:
            last_ts, last_w = pts[-1]
            if (_dt(ts) - _dt(last_ts)) <= timedelta(hours=36) and wt - last_w >= th["weight_gain_day_kg"]:
                out.append(_a(CALL, "weight_day", f"Αύξηση βάρους +{wt-last_w:.1f} kg", f"Weight +{wt-last_w:.1f} kg since last",
                              "Πιθανή κατακράτηση υγρών — ενημερώστε την ομάδα.", "Possible fluid retention — tell the team.", plan, ts, {"weight": wt, "prev": last_w}))
            wk = [w for (t, w) in pts if timedelta(0) < _dt(ts) - _dt(t) <= timedelta(days=7)]
            if wk and wt - min(wk) >= th["weight_gain_week_kg"]:
                out.append(_a(CALL, "weight_week", f"Αύξηση βάρους +{wt-min(wk):.1f} kg σε 7 ημέρες", f"Weight +{wt-min(wk):.1f} kg in 7 days",
                              "", "", plan, ts, {"weight": wt}))
            d3 = [w for (t, w) in pts if timedelta(0) < _dt(ts) - _dt(t) <= timedelta(days=3)]
            if d3 and max(d3) - wt >= th["weight_loss_3d_kg"]:
                out.append(_a(WATCH, "weight_loss", f"Απώλεια βάρους −{max(d3)-wt:.1f} kg σε 3 ημέρες", f"Weight −{max(d3)-wt:.1f} kg in 3 days",
                              "Πιθανή αφυδάτωση (λαμβάνει Lasix και Co-Diovan).", "Possible dehydration (on diuretics).", plan, ts, {"weight": wt}))

    # 7. repeated abnormal BP in 24 h  →  “unstable BP”
    if sbp is not None:
        window = [h for h in history if (_dt(ts) - _dt(h["ts"])) <= timedelta(hours=24)] + [entry]
        bad = [h for h in window if h.get("sbp") is not None and
               (h["sbp"] < th["sbp_low_call"] or h["sbp"] >= th["sbp_high_call"])]
        if len(bad) >= th["bp_unstable_count_24h"]:
            out.append(_a(CALL, "bp_unstable", f"{len(bad)} μετρήσεις πίεσης εκτός ορίων σε 24 ώρες",
                          f"{len(bad)} out-of-range BP readings in 24 h", "Επαναλαμβανόμενη ασταθής πίεση.",
                          "Repeated unstable pressure.", plan, ts, {"n": len(bad)}))

    # 8. consecutive high pulse
    if hr is not None and hr > th["hr_high_call"]:
        prev = next((h for h in reversed(history) if h.get("hr") is not None), None)
        if prev and prev["hr"] > th["hr_high_call"] and (_dt(ts) - _dt(prev["ts"])) <= timedelta(hours=6):
            out.append(_a(CALL, "hr_high_sustained", "Επίμονοι υψηλοί σφυγμοί", "Sustained high pulse",
                          "Δύο συνεχόμενες μετρήσεις πάνω από το όριο.", "Two consecutive readings above the limit.", plan, ts, {"hr": hr}))

    # 9. missed critical medication
    for m in entry.get("meds_missed") or []:
        if is_critical_med(m):
            out.append(_a(WATCH, "med_missed", f"Παραλείφθηκε δόση: {m}", f"Missed dose: {m}",
                          "Μην διπλασιάσετε τη δόση — ρωτήστε φαρμακοποιό/ομάδα τι να κάνετε.",
                          "Do not double up — ask the pharmacist/team what to do.", plan, ts, {"med": m}))
    out.sort(key=lambda a: -a.level)
    return out


def evaluate(plan: dict, entries: list, now: datetime | None = None) -> dict:
    """Current picture: alerts for the newest entry, 'overdue' state, level."""
    now = now or datetime.now(TZ)
    es = sort_entries(entries)
    alerts, last = [], (es[-1] if es else None)
    if last:
        alerts = evaluate_entry(plan, last, es[:-1])
    th = {**DEFAULT_THRESHOLDS, **(plan.get("thresholds") or {})}
    overdue = None
    hour = now.astimezone(TZ).hour
    if last:
        gap_h = (now - _dt(last["ts"])).total_seconds() / 3600
        awake = 7 <= hour < 23
        # An alerting entry must be re-checked within 1 h.
        if max([a.level for a in alerts] + [0]) >= CALL and gap_h >= 1:
            overdue = {"kind": "recheck", "gap_h": round(gap_h, 1)}
        elif awake and gap_h >= th["max_gap_hours"]:
            overdue = {"kind": "missed", "gap_h": round(gap_h, 1)}
    elif day_number(plan, now) >= 1 and 7 <= hour < 23:
        overdue = {"kind": "first", "gap_h": None}
    level = max([a.level for a in alerts] + [0])
    if overdue and level == OK:
        level = WATCH
    return {"level": level, "alerts": alerts, "last": last, "overdue": overdue, "day": day_number(plan, now)}


def alert_log(plan, entries) -> list:
    """Every alert ≥ CALL ever raised, oldest first (for the clinician summary)."""
    es = sort_entries(entries)
    rows = []
    for i, e in enumerate(es):
        for a in evaluate_entry(plan, e, es[:i]):
            if a.level >= CALL:
                rows.append(a)
    return rows


# ── Thresholds ────────────────────────────────────────────────────────────────
def apply_letter_thresholds(plan: dict, stated: dict) -> list:
    """Merge numbers the discharge letter/clinician stated explicitly. Returns list of keys applied."""
    applied = []
    for k, v in (stated or {}).items():
        if k in DEFAULT_THRESHOLDS:
            v = _f(v)
            lo, hi = THRESHOLD_BOUNDS[k]
            if v is not None and lo <= v <= hi:
                plan["thresholds"][k] = v; applied.append(k)
    if applied:
        plan["thresholds_source"] = "clinician"
    return applied


def thresholds_conflicts(th: dict) -> list:
    """Logical sanity of an edited threshold set."""
    bad = []
    if th["hr_low_emergency"] >= th["hr_low_call"]: bad.append("hr_low_emergency < hr_low_call")
    if th["hr_high_emergency"] <= th["hr_high_call"]: bad.append("hr_high_emergency > hr_high_call")
    if th["sbp_low_emergency"] >= th["sbp_low_call"]: bad.append("sbp_low_emergency < sbp_low_call")
    if th["spo2_emergency"] >= th["spo2_call"]: bad.append("spo2_emergency < spo2_call")
    return bad


def baseline_hints(plan, lang="el") -> list:
    """Warn when a stated baseline sits on the wrong side of a default threshold."""
    th, b, out = plan["thresholds"], plan.get("baseline") or {}, []
    if plan.get("thresholds_source") == "clinician":
        return out
    hm = _f(b.get("hr_min_known"))
    if hm is not None and hm < th["hr_low_call"]:
        out.append((f"Η ασθενής έχει καταγεγραμμένο ελάχιστο σφυγμό {int(hm)}/λεπτό — το προεπιλεγμένο όριο ({int(th['hr_low_call'])}) θα δώσει συχνά ψευδείς συναγερμούς ή θα παραβλέψει πραγματικό πρόβλημα. Ζητήστε από τον καρδιολόγο προσωπικό όριο."
                    if lang == "el" else
                    f"A recorded minimum pulse of {int(hm)}/min is below the default limit ({int(th['hr_low_call'])}). Ask the cardiologist for a personal limit."))
    su = _f(b.get("spo2_usual"))
    if su is not None and su < th["spo2_call"]:
        out.append(("Η συνηθισμένη SpO₂ είναι κάτω από το προεπιλεγμένο όριο — ζητήστε προσωπικό όριο από τον γιατρό."
                    if lang == "el" else "Usual SpO₂ is below the default limit — ask the doctor for a personal limit."))
    return out


# ── Series for charts ─────────────────────────────────────────────────────────
def series(entries) -> list:
    rows = []
    for e in sort_entries(entries):
        rows.append({"ts": _dt(e["ts"]).astimezone(TZ).replace(tzinfo=None), "hr": e.get("hr"), "sbp": e.get("sbp"),
                     "dbp": e.get("dbp"), "spo2": e.get("spo2"), "temp": e.get("temp"), "weight": e.get("weight")})
    return rows


def compliance(plan, entries, now=None) -> dict:
    """Check-ins done today vs planned."""
    now = now or datetime.now(TZ)
    today = now.astimezone(TZ).date()
    n = sum(1 for e in entries if _dt(e["ts"]).astimezone(TZ).date() == today)
    planned = plan.get("checkins_per_day", 2) if day_number(plan, now) <= plan.get("intensive_days", 3) + 0 else max(1, plan.get("checkins_per_day", 2) - 1)
    return {"done": n, "planned": planned}


# ── Discharge letter extraction (the only networked function) ─────────────────
_LETTER_SYSTEM = """You are a meticulous medical records transcriptionist reading a hospital discharge letter / instructions.
Copy facts exactly as written. Never infer, never add medical advice, never invent numbers.
If something is not in the document use null / empty list. Reply with ONE JSON object only."""

_LETTER_PROMPT = """Extract from this discharge document. JSON schema:
{
 "patient_name_on_document": str|null,
 "procedure": str|null,                // as written
 "admission_course": str|null,         // 1-3 sentences, as written (e.g. ICU stay, ventilation, blood-pressure problems)
 "discharge_date": "YYYY-MM-DD"|null,
 "medications": [{"name": str, "dose": str|null, "times": [str], "status": "new"|"changed"|"continued"|"stopped"|null, "note": str|null}],
 "restrictions": [str],                // driving, lifting, bathing, wound care, activity — as written
 "followups": [{"label": str, "date": "YYYY-MM-DD"|null, "place": str|null}],
 "contacts": [{"label": str, "phone": str}],
 "red_flags_stated": [str],            // warning signs the document itself lists, as written
 "thresholds_stated": {                // ONLY numbers explicitly written in the document, else omit the key
    "hr_low_call": num, "hr_high_call": num, "sbp_low_call": num, "sbp_high_call": num,
    "spo2_call": num, "temp_call": num, "weight_gain_day_kg": num, "weight_gain_week_kg": num
 }
}
Weights in lb must be converted to kg; temperatures in °F to °C (rounded to 0.1)."""


def _clean_letter(d: dict) -> dict:
    s = lambda v: (str(v).strip() if v not in (None, "") else None)
    meds = []
    for m in d.get("medications") or []:
        if isinstance(m, dict) and s(m.get("name")):
            meds.append({"name": s(m["name"]), "dose": s(m.get("dose")) or "",
                         "times": [str(x) for x in (m.get("times") or [])],
                         "status": s(m.get("status")), "note": s(m.get("note")) or "",
                         "critical": is_critical_med(m["name"])})
    fu = [{"label": s(x.get("label")) or "", "date": s(x.get("date")), "place": s(x.get("place")) or ""}
          for x in (d.get("followups") or []) if isinstance(x, dict) and s(x.get("label"))]
    ct = [{"label": s(x.get("label")) or "", "phone": s(x.get("phone")) or ""}
          for x in (d.get("contacts") or []) if isinstance(x, dict) and s(x.get("phone"))]
    th = {k: v for k, v in (d.get("thresholds_stated") or {}).items() if k in DEFAULT_THRESHOLDS}
    dd = s(d.get("discharge_date"))
    if dd and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", dd):
        dd = None
    return {"patient_name": s(d.get("patient_name_on_document")), "procedure": s(d.get("procedure")),
            "course": s(d.get("admission_course")) or "", "discharge_date": dd, "meds": meds,
            "restrictions": [str(x) for x in (d.get("restrictions") or []) if str(x).strip()],
            "followups": fu, "contacts": ct,
            "red_flags": [str(x) for x in (d.get("red_flags_stated") or []) if str(x).strip()],
            "thresholds_stated": th}


def extract_discharge(api_key, file_bytes, mime, name, heic_convert=None, downscale=None, log=None) -> dict:
    import dossier as _d
    t0 = time.time()
    try:
        block = _d._prep_file(file_bytes, mime, name, heic_convert, downscale)
        reply = _d._call_claude(api_key, _LETTER_SYSTEM, [block, {"type": "text", "text": _LETTER_PROMPT}], max_tokens=4000)
        data = _clean_letter(_d._parse_json(reply))
        if log: log("recovery_extract", ok=True, ms=(time.time() - t0) * 1000)
        return {"ok": True, "data": data, "error": None}
    except Exception as ex:  # noqa: BLE001
        if log: log("recovery_extract", ok=False, ms=(time.time() - t0) * 1000, error=str(ex)[:200])
        return {"ok": False, "data": None, "error": str(ex)}


def merge_letter(plan: dict, letter: dict) -> dict:
    """Put extracted letter data into the plan (carer reviews it afterwards)."""
    if letter.get("discharge_date"): plan["discharge_date"] = letter["discharge_date"]
    if letter.get("course"): plan["hospital_course"] = letter["course"]
    if letter.get("procedure"): plan["procedure_label"] = letter["procedure"]
    have = {m["name"].lower() for m in plan["meds"]}
    for m in letter.get("meds", []):
        if m.get("status") == "stopped" or m["name"].lower() in have:
            continue
        plan["meds"].append({k: m[k] for k in ("name", "dose", "times", "critical", "note")})
    plan["restrictions"] = list(dict.fromkeys(plan["restrictions"] + letter.get("restrictions", [])))
    plan["followups"] += [f for f in letter.get("followups", []) if f not in plan["followups"]]
    plan["contacts"] += [c for c in letter.get("contacts", []) if c not in plan["contacts"]]
    plan["letter_red_flags"] = list(dict.fromkeys(plan["letter_red_flags"] + letter.get("red_flags", [])))
    apply_letter_thresholds(plan, letter.get("thresholds_stated"))
    return plan


# ── Clinician summary (Word + HTML) ───────────────────────────────────────────
S = {
    "el": {"title": "Ημερολόγιο Παρακολούθησης στο Σπίτι", "patient": "Ασθενής", "age": "Ηλικία / Φύλο", "proc": "Επέμβαση / νοσηλεία",
           "dd": "Έξοδος από το νοσοκομείο", "course": "Πορεία νοσηλείας (όπως καταγράφηκε)", "meds": "Φαρμακευτική αγωγή",
           "th": "Όρια ειδοποίησης που χρησιμοποιήθηκαν", "th_src_c": "Όρια από τον θεράποντα / το εξιτήριο",
           "th_src_d": "Γενικά προεπιλεγμένα όρια — δεν έχουν επιβεβαιωθεί από τον θεράποντα",
           "log": "Καταγραφές", "alerts": "Ειδοποιήσεις (επικοινωνία/επείγον)", "none": "Καμία", "when": "Ημ/νία–ώρα",
           "hr": "Σφυγμοί", "bp": "ΑΠ", "spo2": "SpO₂", "temp": "Θερμ.", "wt": "Βάρος", "sym": "Συμπτώματα", "notes": "Σημειώσεις",
           "fu": "Επόμενα ραντεβού", "foot": "Τα στοιχεία καταγράφηκαν από τον φροντιστή με τον Asklepios. Δεν περιέχουν διάγνωση ή θεραπευτική σύσταση.",
           "prep": "Σύνταξη", "name": "Φάρμακο", "dose": "Δόση", "time": "Ώρα", "flag": "Ειδοποίηση", "ext": "Σύντομη εικόνα",
           "n": "Καταγραφές", "range": "Περίοδος"},
    "en": {"title": "Home Monitoring Log", "patient": "Patient", "age": "Age / Sex", "proc": "Procedure / admission",
           "dd": "Hospital discharge", "course": "Hospital course (as recorded)", "meds": "Medication",
           "th": "Alert limits used", "th_src_c": "Limits from the treating team / discharge letter",
           "th_src_d": "Generic default limits — NOT yet confirmed by the treating team",
           "log": "Readings", "alerts": "Alerts (contact / emergency)", "none": "None", "when": "Date–time",
           "hr": "Pulse", "bp": "BP", "spo2": "SpO₂", "temp": "Temp", "wt": "Weight", "sym": "Symptoms", "notes": "Notes",
           "fu": "Upcoming appointments", "foot": "Recorded by the carer using Asklepios. Contains no diagnosis or treatment recommendation.",
           "prep": "Prepared", "name": "Medication", "dose": "Dose", "time": "Time", "flag": "Alert", "ext": "At a glance",
           "n": "Entries", "range": "Period"},
}


def _fmt_ts(ts): return _dt(ts).astimezone(TZ).strftime("%d/%m %H:%M")
def _sym_txt(e, lang): return "; ".join(SYMPTOMS[k][2 if lang == "el" else 3] for k in e.get("symptoms", []))


def summary_rows(plan, entries, lang):
    rows = []
    for e in sort_entries(entries):
        bp = f'{e["sbp"]}/{e["dbp"]}' if e.get("sbp") and e.get("dbp") else (str(e["sbp"]) if e.get("sbp") else "")
        rows.append([_fmt_ts(e["ts"]), str(e.get("hr") or ""), bp, str(e.get("spo2") or ""),
                     str(e.get("temp") or ""), str(e.get("weight") or ""), _sym_txt(e, lang), e.get("notes", "")])
    return rows


def _th_rows(plan, lang):
    i = 0 if lang == "el" else 1
    return [[THRESHOLD_LABELS[k][i], f'{plan["thresholds"][k]:g} {THRESHOLD_LABELS[k][2]}'] for k in DEFAULT_THRESHOLDS]


def build_docx(plan, entries, lang="el") -> bytes:
    from docx import Document
    from docx.shared import Pt, Cm
    L = S[lang]
    doc = Document()
    for s in doc.sections:
        s.left_margin = s.right_margin = Cm(1.8); s.top_margin = s.bottom_margin = Cm(1.6)
    doc.styles["Normal"].font.name = "Calibri"; doc.styles["Normal"].font.size = Pt(10)
    doc.add_heading(L["title"], 1)
    p = plan["patient"]
    es = sort_entries(entries)

    def table(head, rows, widths=None):
        t = doc.add_table(rows=1, cols=len(head)); t.style = "Light Grid Accent 1"
        for i, h in enumerate(head): t.rows[0].cells[i].text = h
        for r in rows:
            c = t.add_row().cells
            for i, v in enumerate(r): c[i].text = str(v)
        return t

    table([L["patient"], ""], [
        [L["patient"], p.get("name", "")], [L["age"], f'{p.get("age") or ""} {p.get("sex") or ""}'.strip()],
        [L["proc"], plan.get("procedure_label", "")], [L["dd"], plan.get("discharge_date", "")],
        [L["range"], f'{_fmt_ts(es[0]["ts"])} – {_fmt_ts(es[-1]["ts"])}' if es else ""], [L["n"], len(es)]])
    if plan.get("hospital_course"):
        doc.add_heading(L["course"], 2); doc.add_paragraph(plan["hospital_course"])
    if plan.get("meds"):
        doc.add_heading(L["meds"], 2)
        table([L["name"], L["dose"], L["time"]], [[m["name"], m.get("dose", ""), ", ".join(m.get("times", []))] for m in plan["meds"]])
    doc.add_heading(L["alerts"], 2)
    al = alert_log(plan, entries)
    if al:
        table([L["when"], L["flag"]], [[_fmt_ts(a.ts), a.title[lang]] for a in al])
    else:
        doc.add_paragraph(L["none"])
    doc.add_heading(L["log"], 2)
    table([L["when"], L["hr"], L["bp"], L["spo2"], L["temp"], L["wt"], L["sym"], L["notes"]], summary_rows(plan, entries, lang))
    doc.add_heading(L["th"], 2)
    doc.add_paragraph(L["th_src_c"] if plan.get("thresholds_source") == "clinician" else L["th_src_d"])
    table(["", ""], _th_rows(plan, lang))
    if plan.get("followups"):
        doc.add_heading(L["fu"], 2)
        table(["", L["when"], ""], [[f.get("label", ""), f.get("date") or "", f.get("place", "")] for f in plan["followups"]])
    para = doc.add_paragraph(); r = para.add_run(L["foot"]); r.italic = True; r.font.size = Pt(8.5)
    bio = io.BytesIO(); doc.save(bio); return bio.getvalue()


def build_html(plan, entries, lang="el") -> str:
    L, e_ = S[lang], _html.escape
    def tbl(head, rows):
        h = "".join(f"<th>{e_(x)}</th>" for x in head)
        b = "".join("<tr>" + "".join(f"<td>{e_(str(c))}</td>" for c in r) + "</tr>" for r in rows)
        return f"<table><tr>{h}</tr>{b}</table>"
    p, es = plan["patient"], sort_entries(entries)
    al = alert_log(plan, entries)
    return f"""<!doctype html><html lang="{lang}"><meta charset="utf-8"><title>{e_(L['title'])}</title>
<style>body{{font:13px/1.45 system-ui,sans-serif;max-width:900px;margin:24px auto;padding:0 14px;color:#0A1030}}
table{{border-collapse:collapse;width:100%;margin:6px 0 16px}}th,td{{border:1px solid #cfd5ea;padding:4px 6px;text-align:left;vertical-align:top}}
th{{background:#eef1fb}}h1{{font-size:20px}}h2{{font-size:15px;margin-top:18px}}.f{{font-size:11px;color:#5a6388;font-style:italic}}</style>
<h1>{e_(L['title'])}</h1>
{tbl([L['patient'], ''], [[L['patient'], p.get('name','')], [L['age'], f"{p.get('age') or ''} {p.get('sex') or ''}"], [L['proc'], plan.get('procedure_label','')], [L['dd'], plan.get('discharge_date','')], [L['n'], len(es)]])}
<h2>{e_(L['alerts'])}</h2>{tbl([L['when'], L['flag']], [[_fmt_ts(a.ts), a.title[lang]] for a in al]) if al else '<p>'+e_(L['none'])+'</p>'}
<h2>{e_(L['log'])}</h2>{tbl([L['when'], L['hr'], L['bp'], L['spo2'], L['temp'], L['wt'], L['sym'], L['notes']], summary_rows(plan, entries, lang))}
<h2>{e_(L['th'])}</h2><p>{e_(L['th_src_c'] if plan.get('thresholds_source')=='clinician' else L['th_src_d'])}</p>{tbl(['',''], _th_rows(plan, lang))}
<p class="f">{e_(L['foot'])}</p></html>"""


# ── Caregiver e-mail alerts (Resend) ──────────────────────────────────────────
import re as _re
import urllib.request as _ur

_EMAIL_RE = _re.compile(r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$")


def parse_emails(text: str) -> tuple:
    """-> (valid_list, invalid_list); splits on newline/comma/semicolon/space, dedups, caps at MAX_NOTIFY_EMAILS."""
    good, bad = [], []
    for tok in _re.split(r"[\s,;]+", text or ""):
        tok = tok.strip()
        if not tok:
            continue
        if _EMAIL_RE.match(tok) and len(tok) <= 120:
            if tok.lower() not in [g.lower() for g in good]:
                good.append(tok)
        else:
            bad.append(tok)
    if len(good) > MAX_NOTIFY_EMAILS:
        bad += good[MAX_NOTIFY_EMAILS:]
        good = good[:MAX_NOTIFY_EMAILS]
    return good, bad


def should_notify(plan: dict, level: int, now: datetime | None = None) -> bool:
    n = plan.get("notify") or {}
    if not n.get("emails") or level < n.get("min_level", CALL) or level < CALL:
        return False
    now = now or datetime.now(TZ)
    last_ts, last_level = n.get("last_ts"), n.get("last_level", 0)
    if last_ts and level <= last_level and (now - _dt(last_ts)) < timedelta(minutes=NOTIFY_QUIET_MIN):
        return False                      # duplicate of what was just sent
    return True


def _vitals_line(e: dict, lang: str) -> str:
    bits = []
    if e.get("hr"): bits.append(f'{"Σφυγμοί" if lang=="el" else "Pulse"} {e["hr"]}')
    if e.get("sbp"): bits.append(f'{"ΑΠ" if lang=="el" else "BP"} {e["sbp"]}/{e.get("dbp") or "?"}')
    if e.get("spo2"): bits.append(f'SpO₂ {e["spo2"]}%')
    if e.get("temp"): bits.append(f'{"Θερμ." if lang=="el" else "Temp"} {e["temp"]}°C')
    if e.get("weight"): bits.append(f'{e["weight"]} kg')
    return " · ".join(bits)


def build_alert_email(plan: dict, entry: dict, alerts: list, lang: str = "el", app_url: str = "",
                      owner_email: str = "", test: bool = False) -> tuple:
    """-> (subject, text, html).  Only alerts >= CALL are listed."""
    n = plan.get("notify") or {}
    detail = n.get("detail", True)
    name = (plan.get("patient") or {}).get("name") or ("ο/η ασθενής" if lang == "el" else "the patient")
    top = max([a.level for a in alerts] + [CALL])
    shown = [a for a in alerts if a.level >= CALL] or alerts
    lbl = LEVEL_LABEL[lang][top]
    when = _dt(entry["ts"]).astimezone(TZ).strftime("%d/%m %H:%M")
    if test:
        subj = ("[ΔΟΚΙΜΗ] Asklepios — ειδοποιήσεις οικογένειας" if lang == "el" else "[TEST] Asklepios — family alerts")
    elif detail:
        subj = f"[{lbl}] {name}: {shown[0].title[lang]}" if shown else f"[{lbl}] {name}"
    else:
        subj = f"[{lbl}] Asklepios"
    lines = []
    if test:
        lines.append("Αυτό είναι δοκιμαστικό μήνυμα. Αν το βλέπετε, οι ειδοποιήσεις λειτουργούν." if lang == "el"
                     else "This is a test message. If you can see it, alerts are working.")
    elif detail:
        lines += [f"{name} — {when}", _vitals_line(entry, lang)]
        for a in shown:
            lines.append(f"• {a.title[lang]}" + (f" — {a.detail[lang]}" if a.detail[lang] else ""))
        lines += ["", (shown[0].action[lang] if shown else "")]
    else:
        lines += [("Υπάρχει νέα ειδοποίηση στην παρακολούθηση. Ανοίξτε την εφαρμογή για λεπτομέρειες." if lang == "el"
                   else "There is a new alert in home monitoring. Open the app for details."), "",
                  (("Καλέστε ΤΩΡΑ 166 (ΕΚΑΒ) ή 112." if lang == "el" else "Call 166 (EKAB) or 112 NOW.") if top == EMERGENCY
                   else ("Επικοινωνήστε σήμερα με την ομάδα που παρακολουθεί τον ασθενή." if lang == "el"
                         else "Contact the patient's care team today."))]
    if app_url:
        lines += ["", (f"Άνοιγμα: {app_url}" if lang == "el" else f"Open: {app_url}")]
    foot = ("Αυτόματο μήνυμα από τον Asklepios — δεν αποτελεί διάγνωση. Σε επείγον σύμπτωμα καλέστε 166 ή 112. Μην απαντάτε σε αυτό το email."
            if lang == "el" else
            "Automated message from Asklepios — not a diagnosis. In an emergency call 166 or 112. Do not reply to this email.")
    if owner_email:
        foot += (f" Λαμβάνετε το μήνυμα επειδή ο/η {owner_email} σας πρόσθεσε ως φροντιστή· αν δεν το αναμένατε, αγνοήστε το."
                 if lang == "el" else
                 f" You receive this because {owner_email} added you as a carer; if unexpected, ignore it.")
    text = "\n".join(l for l in lines if l is not None) + "\n\n--\n" + foot
    col = LEVEL_COLOR[top]
    e_ = _html.escape
    body = "".join(f"<p style='margin:6px 0'>{e_(l)}</p>" if l else "<br>" for l in lines)
    html = (f"<div style='font:15px/1.5 system-ui,sans-serif;max-width:560px;margin:auto'>"
            f"<div style='background:{col};color:#fff;padding:14px 18px;border-radius:12px 12px 0 0;font-weight:800;font-size:17px'>{e_(lbl)}</div>"
            f"<div style='border:1px solid #E1E5F4;border-top:0;padding:14px 18px;border-radius:0 0 12px 12px'>{body}</div>"
            f"<p style='font-size:12px;color:#5A6388'>{e_(foot)}</p></div>")
    return subj, text, html


def send_resend(api_key: str, sender: str, to: list, subject: str, text: str, html: str,
                idem_key: str = "", timeout: int = 15) -> dict:
    """POST https://api.resend.com/emails.  Never raises."""
    if not api_key or not sender or not to:
        return {"ok": False, "id": None, "error": "not_configured"}
    body = json.dumps({"from": sender, "to": to, "subject": subject, "text": text, "html": html}).encode()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "User-Agent": "asklepios/1.0"}
    if idem_key:
        headers["Idempotency-Key"] = idem_key[:256]
    try:
        req = _ur.Request("https://api.resend.com/emails", data=body, headers=headers, method="POST")
        with _ur.urlopen(req, timeout=timeout) as r:
            return {"ok": True, "id": json.loads(r.read() or b"{}").get("id"), "error": None}
    except Exception as ex:  # noqa: BLE001
        msg = str(ex)
        try:
            msg = f"{ex.code}: {ex.read().decode()[:200]}"          # urllib HTTPError
        except Exception:
            pass
        return {"ok": False, "id": None, "error": msg[:240]}


def notify_if_needed(plan: dict, entries: list, cfg: dict, now: datetime | None = None,
                     sender=None, log=None) -> dict:
    """Call right after a check-in is saved.  cfg = {api_key, from, app_url, owner}.
    Updates plan['notify'] (last_level/last_ts) when a mail went out.
    -> {"sent": bool, "reason": str, "to": [..], "error": str|None}"""
    now = now or datetime.now(TZ)
    res = evaluate(plan, entries, now)
    n = plan.setdefault("notify", dict(NOTIFY_DEFAULT, emails=[]))
    if not n.get("emails"):
        return {"sent": False, "reason": "no_recipients", "to": [], "error": None}
    if not should_notify(plan, res["level"], now):
        return {"sent": False, "reason": "not_needed", "to": [], "error": None}
    if not (cfg.get("api_key") and cfg.get("from")):
        return {"sent": False, "reason": "not_configured", "to": n["emails"], "error": None}
    entry = res["last"]
    subj, text, html = build_alert_email(plan, entry, res["alerts"], n.get("lang", "el"), cfg.get("app_url", ""), cfg.get("owner", ""))
    r = (sender or send_resend)(cfg["api_key"], cfg["from"], n["emails"], subj, text, html, idem_key=f'{plan["id"]}-{entry["id"]}')
    if log: log("recovery_notify", ok=r["ok"], level=res["level"], error=r["error"])
    if r["ok"]:
        n["last_level"], n["last_ts"] = res["level"], now.isoformat(timespec="seconds")
    return {"sent": r["ok"], "reason": "sent" if r["ok"] else "error", "to": n["emails"], "error": r["error"]}
