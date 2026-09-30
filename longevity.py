"""
longevity.py — home "Longevity check" calculations (pure functions, no UI).

Inputs come from the fingertip-camera measurements (components/pulse) and a
short questionnaire. Everything here is an ESTIMATE for wellness purposes;
the UI must label it so and point to a lab test (CPET, e.g. PNOE) for real
values.

Methods and sources
-------------------
HRmax (estimated)      211 − 0.64 × age
                       Nes BM et al., HUNT Fitness Study (Scand J Med Sci Sports
                       2013); used by NTNU CERG's HRmax calculator.
VO2max (estimated)     Heart-rate-ratio method: 15.3 × HRmax / HRrest
                       Uth N et al., Eur J Appl Physiol 2004. Simple and uses the
                       resting pulse we actually measure; accuracy is limited
                       (roughly ±15 %), so a range is always shown.
                       NOT valid on beta-blockers / rate-limiting drugs.
Reference VO2 by age   HUNT3 reference data (Loe H et al., PLoS One 2013):
                       20–29 y men 54.4, women 43.0 ml/kg/min, falling
                       ≈3.5 ml/kg/min per decade. Used for "fitness age" and
                       the 5-level category (z-score vs age/sex mean).
Heart-rate recovery    HR at the end of a 3-min step test (first 10 s after the
                       finger is on the camera) minus HR 55–65 s after stopping.
                       A drop of ≤12 bpm in the first minute is the classic
                       adverse threshold (Cole CR et al., NEJM 1999); the other
                       bands below are indicative for this home protocol.
Step protocol          ~20 cm step (a normal stair), 24 steps/min (metronome 96),
                       3 minutes — the Tecumseh/Kasch-style submaximal format.
Training zones         Karvonen (% of heart-rate reserve) from measured HRrest
                       and estimated HRmax.

TODO before marketing claims: add the HUNT non-exercise VO2peak model
(Nes et al., MSSE 2011) once the published coefficients are verified from the
paper, and validate against CPET results (PNOE) from real users.
"""
from __future__ import annotations

LEVELS = [
    # key, Greek, English, colour
    ("severe", "Σοβαρός περιορισμός", "Severe limitation", "#DC2626"),
    ("limit", "Περιορισμός", "Limitation", "#F59E0B"),
    ("neutral", "Ουδέτερο", "Neutral", "#6366F1"),
    ("good", "Καλό", "Good", "#14B8A6"),
    ("excellent", "Εξαιρετικό", "Excellent", "#059669"),
]
_LV = {k: i for i, (k, *_r) in enumerate(LEVELS)}


def level(key, lang="el"):
    k, el, en, col = LEVELS[_LV[key]]
    return {"key": k, "label": el if lang == "el" else en, "color": col, "score": _LV[key]}


def hr_max(age):
    return round(211 - 0.64 * float(age))


def vo2_uth(hr_rest, age):
    if not hr_rest or hr_rest <= 30:
        return None
    return round(15.3 * hr_max(age) / float(hr_rest), 1)


def _is_male(sex):
    return str(sex or "").strip().lower() in ("άνδρας", "ανδρας", "male", "m", "man")


def ref_vo2(age, sex):
    """HUNT3 age/sex mean VO2max (ml/kg/min), linear ≈3.5 per decade from 25 y."""
    base = 54.4 if _is_male(sex) else 43.0
    return base - 0.35 * (float(age) - 25.0)


def ref_sd(sex):
    return 8.4 if _is_male(sex) else 7.7


def fitness_age(vo2, sex):
    base = 54.4 if _is_male(sex) else 43.0
    return int(round(max(20.0, min(90.0, 25.0 + (base - float(vo2)) / 0.35))))


def vo2_level(vo2, age, sex):
    # HUNT3 volunteers were fitter than the general population, so the bands
    # are deliberately wide: "limitation" starts ~0.75 SD below their mean.
    z = (float(vo2) - ref_vo2(age, sex)) / ref_sd(sex)
    if z >= 0.5:
        return "excellent", z
    if z >= 0.0:
        return "good", z
    if z >= -0.75:
        return "neutral", z
    if z >= -1.5:
        return "limit", z
    return "severe", z


def rest_hr_level(hr):
    if hr < 60:
        return "excellent"
    if hr < 70:
        return "good"
    if hr < 80:
        return "neutral"
    if hr < 90:
        return "limit"
    return "severe"


def hrv_level(rmssd):
    if rmssd >= 50:
        return "excellent"
    if rmssd >= 35:
        return "good"
    if rmssd >= 20:
        return "neutral"
    if rmssd >= 12:
        return "limit"
    return "severe"


def hrr_level(drop):
    if drop >= 30:
        return "excellent"
    if drop >= 22:
        return "good"
    if drop >= 15:
        return "neutral"
    if drop > 12:
        return "limit"
    return "severe"


def zones(hr_rest, hr_mx):
    hrr = hr_mx - hr_rest
    cuts = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    out = []
    for i in range(5):
        out.append((i + 1, round(hr_rest + cuts[i] * hrr), round(hr_rest + cuts[i + 1] * hrr)))
    return out


def analyse(age, sex, rest=None, step=None, q=None, lang="el"):
    """Build the full result dict for the UI and the report."""
    q = q or {}
    el = lang == "el"
    res = {"age": age, "sex": sex, "pillars": [], "notes": [], "plan": []}
    rate_limited = bool(q.get("beta_blocker"))
    hrm = hr_max(age)
    res["hr_max_est"] = hrm

    hr_rest = (rest or {}).get("hr")
    if hr_rest:
        lv = rest_hr_level(hr_rest)
        res["pillars"].append({"id": "rhr", "name": "Σφυγμοί ηρεμίας" if el else "Resting pulse",
                               "value": f"{hr_rest} bpm", **level(lv, lang)})
    rmssd = (rest or {}).get("rmssd")
    if rmssd:
        lv = hrv_level(rmssd)
        res["pillars"].append({"id": "hrv", "name": "HRV (νευρικό σύστημα / στρες)" if el else "HRV (nervous system / stress)",
                               "value": f"{rmssd} ms", **level(lv, lang)})

    if hr_rest and not rate_limited:
        vo2 = vo2_uth(hr_rest, age)
        lv, z = vo2_level(vo2, age, sex)
        res["vo2"] = vo2
        res["vo2_range"] = (round(vo2 * 0.85, 1), round(vo2 * 1.15, 1))
        res["fitness_age"] = fitness_age(vo2, sex)
        res["fitness_age_range"] = (fitness_age(res["vo2_range"][1], sex), fitness_age(res["vo2_range"][0], sex))
        res["pillars"].insert(0, {"id": "vo2", "name": "Αερόβια ικανότητα (VO2max, εκτίμηση)" if el else "Aerobic capacity (VO2max, estimate)",
                                  "value": f"~{vo2} ml/kg/min", **level(lv, lang)})
        res["zones"] = zones(hr_rest, hrm)

    if step and step.get("hrr60") is not None:
        d = int(step["hrr60"])
        lv = hrr_level(d) if not rate_limited else "neutral"
        res["hrr60"] = d
        res["pillars"].append({"id": "hrr", "name": "Ικανότητα αποκατάστασης" if el else "Recovery capacity",
                               "value": f"−{d} bpm / 1′", **level(lv, lang)})

    # ── notes & plan (plain, conservative, wellness framing) ──────────────
    if rate_limited:
        res["notes"].append("Παίρνεις φάρμακο που επηρεάζει τους σφυγμούς (π.χ. β-αναστολέα): η εκτίμηση VO2max και οι ζώνες δεν ισχύουν. Για ακριβή εικόνα χρειάζεται εργομετρικό τεστ."
                            if el else "You take a medicine that affects heart rate (e.g. a beta-blocker): the VO2max estimate and zones don't apply. A lab exercise test is needed for an accurate picture.")
    if q.get("smoker"):
        res["plan"].append("🚭 " + ("Η διακοπή του καπνίσματος είναι η μεγαλύτερη βελτίωση για καρδιά, πνεύμονες και μακροζωία." if el
                                    else "Stopping smoking is the single biggest gain for heart, lungs and longevity."))
    z2 = res.get("zones", [None, None])[1] if res.get("zones") else None
    act = int(q.get("active_days") or 0)
    if z2:
        res["plan"].append("🚶 " + (f"Zone 2 (σφυγμοί {z2[1]}–{z2[2]}): 150–180 λεπτά την εβδομάδα — γρήγορο περπάτημα, ποδήλατο. Πρέπει να μπορείς να μιλάς."
                                    if el else f"Zone 2 (pulse {z2[1]}–{z2[2]}): 150–180 minutes a week — brisk walking, cycling. You should still be able to talk."))
    pv = {p["id"]: p for p in res["pillars"]}
    if "vo2" in pv and pv["vo2"]["score"] <= 2 and not rate_limited and act >= 2:
        z4 = res["zones"][3]
        res["plan"].append("⚡ " + (f"Μία φορά την εβδομάδα διαλείμματα 4×4 λεπτά στη Zone 4 ({z4[1]}–{z4[2]}), με 3 λεπτά χαλαρά ενδιάμεσα — ο πιο αποδοτικός τρόπος να ανέβει το VO2max. Μόνο αν ο γιατρός σου δεν έχει αντίρρηση."
                                    if el else f"Once a week 4×4-minute intervals in Zone 4 ({z4[1]}–{z4[2]}) with 3 easy minutes between — the most efficient way to raise VO2max. Only if your doctor has no objection."))
    if "hrr" in pv and pv["hrr"]["score"] <= 1:
        res["plan"].append("🔁 " + ("Η αποκατάσταση των σφυγμών είναι αργή: περισσότερη Zone 2, καλός ύπνος, και συζήτησέ το με τον γιατρό σου (η πτώση ≤12 παλμών το πρώτο λεπτό αξίζει έλεγχο)."
                                    if el else "Pulse recovery is slow: more Zone 2, good sleep, and mention it to your doctor (a drop of ≤12 beats in the first minute is worth checking)."))
    if "hrv" in pv and pv["hrv"]["score"] <= 1:
        res["plan"].append("😴 " + ("Χαμηλό HRV: ύπνος 7–9 ώρες, λιγότερο αλκοόλ, 5 λεπτά αργές αναπνοές (6 το λεπτό) την ημέρα."
                                    if el else "Low HRV: 7–9 hours of sleep, less alcohol, 5 minutes of slow breathing (6 per minute) a day."))
    sleep = q.get("sleep_h")
    if sleep and float(sleep) < 7:
        res["plan"].append("🛏️ " + ("Ύπνος κάτω από 7 ώρες: ο πιο γρήγορος τρόπος να βελτιωθούν αποκατάσταση και HRV." if el
                                    else "Sleep under 7 hours: the quickest lever for recovery and HRV."))
    res["plan"].append("📅 " + ("Επανάλαβε τον έλεγχο σε 8 εβδομάδες, ίδια ώρα της ημέρας, για να δεις την πρόοδο." if el
                                else "Repeat the check in 8 weeks, same time of day, to see your progress."))
    return res
