"""Run:  python -m pytest tests/test_recovery.py   (or: python tests/test_recovery.py)
Synthetic data only — no real patient information."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime, timedelta
import recovery as R

T0 = datetime(2026, 10, 20, 8, 0, tzinfo=R.TZ)

def plan(**kw):
    p = R.new_plan("Test Patient", 81, "F", "tavi", "2026-10-20")
    p.update(kw); return p

def E(h=0, **kw):
    e = R.new_entry(ts=(T0 + timedelta(hours=h)).isoformat(), **kw); return e

def codes(alerts): return {a.code for a in alerts}
def top(entries, p=None, now=None):
    p = p or plan(); return R.evaluate(p, entries, now or (T0 + timedelta(hours=entries[-1]["ts"] and 0)))

def test_normal_is_ok():
    es = [E(0, hr=72, sbp=130, dbp=75, spo2=96, temp=36.8, weight=60.0)]
    r = R.evaluate(plan(), es, T0 + timedelta(minutes=10))
    assert r["level"] == R.OK and not r["alerts"]

def test_each_emergency_symptom_is_emergency():
    for k, v in R.SYMPTOMS.items():
        if v[0] == R.EMERGENCY:
            r = R.evaluate(plan(), [E(0, symptoms=[k])], T0 + timedelta(minutes=5))
            assert r["level"] == R.EMERGENCY, k

def test_hr_thresholds():
    p = plan()
    assert R.evaluate(p, [E(0, hr=38)], T0)["level"] == R.EMERGENCY
    assert R.evaluate(p, [E(0, hr=47)], T0)["level"] == R.CALL
    assert R.evaluate(p, [E(0, hr=47, symptoms=["presyncope"])], T0)["level"] == R.EMERGENCY
    assert R.evaluate(p, [E(0, hr=125)], T0)["level"] == R.CALL            # no symptoms
    assert R.evaluate(p, [E(0, hr=125, symptoms=["breath_worse"])], T0)["level"] == R.EMERGENCY
    assert R.evaluate(p, [E(0, hr=60)], T0)["level"] == R.OK

def test_bp_thresholds():
    p = plan()
    assert R.evaluate(p, [E(0, sbp=80, dbp=50)], T0)["level"] == R.EMERGENCY
    assert R.evaluate(p, [E(0, sbp=95, dbp=60)], T0)["level"] == R.CALL
    assert R.evaluate(p, [E(0, sbp=190, dbp=100)], T0)["level"] == R.CALL
    assert R.evaluate(p, [E(0, sbp=190, dbp=100, symptoms=["stroke_signs"])], T0)["level"] == R.EMERGENCY

def test_orthostatic():
    r = R.evaluate(plan(), [E(0, sbp=130, dbp=70, sbp_stand=105)], T0)
    assert "orthostatic" in codes(r["alerts"])

def test_spo2_and_fever():
    p = plan()
    assert R.evaluate(p, [E(0, spo2=88)], T0)["level"] == R.EMERGENCY
    assert R.evaluate(p, [E(0, spo2=92)], T0)["level"] == R.CALL
    assert R.evaluate(p, [E(0, temp=38.4)], T0)["level"] == R.CALL
    assert R.evaluate(p, [E(0, temp=38.0)], T0)["level"] == R.OK

def test_weight_rules():
    p = plan()
    es = [E(0, weight=60.0), E(24, weight=61.2)]
    assert "weight_day" in codes(R.evaluate(p, es, T0 + timedelta(hours=25))["alerts"])
    es = [E(0, weight=60.0), E(24, weight=60.6), E(48, weight=61.1), E(72, weight=62.4)]
    assert "weight_week" in codes(R.evaluate(p, es, T0 + timedelta(hours=73))["alerts"])
    es = [E(0, weight=62.0), E(24, weight=60.5)]
    assert "weight_loss" not in codes(R.evaluate(p, es, T0 + timedelta(hours=25))["alerts"])
    es = [E(0, weight=62.0), E(24, weight=60.5), E(48, weight=59.8)]
    assert "weight_loss" in codes(R.evaluate(p, es, T0 + timedelta(hours=49))["alerts"])

def test_bp_unstable_pattern_and_swing():
    es = [E(0, sbp=98, dbp=60), E(4, sbp=135, dbp=80), E(8, sbp=96, dbp=58), E(12, sbp=94, dbp=55)]
    c = codes(R.evaluate(plan(), es, T0 + timedelta(hours=12, minutes=5))["alerts"])
    assert "bp_unstable" in c
    es = [E(0, sbp=140, dbp=80), E(3, sbp=98, dbp=60)]
    assert "bp_swing" in codes(R.evaluate(plan(), es, T0 + timedelta(hours=3))["alerts"])

def test_missed_and_recheck():
    p = plan()
    es = [E(0, hr=70, sbp=130, dbp=80)]
    r = R.evaluate(p, es, T0 + timedelta(hours=11))        # 19:00, gap 11h
    assert r["overdue"]["kind"] == "missed" and r["level"] == R.WATCH
    es = [E(0, hr=47)]
    r = R.evaluate(p, es, T0 + timedelta(minutes=90))
    assert r["overdue"]["kind"] == "recheck"
    # night: no 'missed' alert
    r = R.evaluate(p, [E(0, hr=70)], T0 + timedelta(hours=16))   # 00:00
    assert r["overdue"] is None

def test_missed_critical_med():
    es = [E(0, meds_missed=["Salospir (aspirin) 100mg"])]
    assert "med_missed" in codes(R.evaluate(plan(), es, T0)["alerts"])
    es = [E(0, meds_missed=["Magnesium"])]
    assert "med_missed" not in codes(R.evaluate(plan(), es, T0)["alerts"])

def test_letter_thresholds_override_and_bounds():
    p = plan()
    ap = R.apply_letter_thresholds(p, {"hr_low_call": 45, "sbp_low_call": 5, "bogus": 1})
    assert ap == ["hr_low_call"] and p["thresholds"]["hr_low_call"] == 45 and p["thresholds_source"] == "clinician"
    assert p["thresholds"]["sbp_low_call"] == 100
    assert R.evaluate(p, [E(0, hr=47)], T0)["level"] == R.OK

def test_baseline_hint():
    p = plan(); p["baseline"]["hr_min_known"] = 36
    assert R.baseline_hints(p, "en")
    p["thresholds_source"] = "clinician"
    assert not R.baseline_hints(p, "en")

def test_validate_entry():
    assert R.validate_entry(R.new_entry(hr=700))
    assert R.validate_entry(R.new_entry(sbp=100, dbp=120))
    assert not R.validate_entry(R.new_entry(hr=70, sbp=120, dbp=80))

def test_clean_letter_and_merge():
    raw = {"patient_name_on_document": "X", "discharge_date": "2026-10-22", "medications": [
        {"name": "Aspirin", "dose": "100mg", "times": ["noon"], "status": "continued"},
        {"name": "Old drug", "status": "stopped"}],
        "thresholds_stated": {"hr_low_call": "48", "evil": 1}, "contacts": [{"label": "Cardiology", "phone": "210"}, {"label": "x"}]}
    L = R._clean_letter(raw); p = R.merge_letter(plan(), L)
    assert p["discharge_date"] == "2026-10-22" and [m["name"] for m in p["meds"]] == ["Aspirin"]
    assert p["meds"][0]["critical"] and p["thresholds"]["hr_low_call"] == 48 and len(p["contacts"]) == 1

def test_exports_build():
    es = [E(0, hr=47, sbp=95, dbp=60, symptoms=["presyncope"], notes="n"), E(5, hr=70, sbp=128, dbp=76)]
    for lang in ("el", "en"):
        assert R.build_docx(plan(), es, lang)[:2] == b"PK"
        h = R.build_html(plan(), es, lang)
        assert "47" in h and "<script" not in h
    assert len(R.alert_log(plan(), es)) >= 2

def test_html_escapes():
    es = [E(0, notes="<script>alert(1)</script>")]
    assert "<script>alert" not in R.build_html(plan(), es, "en")

if __name__ == "__main__":
    n = 0
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); n += 1; print("ok", k)
    print(n, "passed")
