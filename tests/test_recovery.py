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

# ── e-mail alerts ─────────────────────────────────────────────────────────────
def mail_plan(**kw):
    p = plan(); p["patient"]["name"] = "Test Patient"
    p["notify"].update({"emails": ["a@example.com"], **kw}); return p

CFG = {"api_key": "k", "from": "Asklepios <x@example.com>", "app_url": "https://app.example", "owner": "me@example.com"}

def fake_sender(box, ok=True):
    def f(api_key, sender, to, subject, text, html, idem_key="", timeout=15):
        box.append({"to": to, "subject": subject, "text": text, "html": html, "idem": idem_key})
        return {"ok": ok, "id": "id1" if ok else None, "error": None if ok else "boom"}
    return f

def test_parse_emails():
    g, b = R.parse_emails("a@example.com, B@example.com\nbad@, a@example.com; c@example.com d@example.com")
    assert g == ["a@example.com", "B@example.com", "c@example.com"] and "bad@" in b and "d@example.com" in b
    assert R.parse_emails("") == ([], [])

def test_should_notify_levels_and_quiet_window():
    p = mail_plan()
    assert not R.should_notify(p, R.WATCH, T0) and R.should_notify(p, R.CALL, T0) and R.should_notify(p, R.EMERGENCY, T0)
    p2 = mail_plan(min_level=R.EMERGENCY)
    assert not R.should_notify(p2, R.CALL, T0) and R.should_notify(p2, R.EMERGENCY, T0)
    p["notify"].update({"last_level": R.CALL, "last_ts": T0.isoformat()})
    assert not R.should_notify(p, R.CALL, T0 + timedelta(minutes=5))          # duplicate
    assert R.should_notify(p, R.EMERGENCY, T0 + timedelta(minutes=5))         # escalation
    assert R.should_notify(p, R.CALL, T0 + timedelta(minutes=16))             # quiet window over
    assert not R.should_notify(plan(), R.EMERGENCY, T0)                        # no recipients

def test_notify_if_needed_flow():
    p, box = mail_plan(), []
    es = [E(0, hr=70, sbp=130, dbp=80)]
    r = R.notify_if_needed(p, es, CFG, T0, fake_sender(box))
    assert r["reason"] == "not_needed" and not box
    es = [E(0, hr=47, sbp=95, dbp=60, symptoms=["presyncope"])]
    r = R.notify_if_needed(p, es, CFG, T0 + timedelta(minutes=1), fake_sender(box))
    assert r["sent"] and len(box) == 1 and box[0]["to"] == ["a@example.com"] and box[0]["idem"].startswith(p["id"])
    assert p["notify"]["last_level"] == R.EMERGENCY
    r = R.notify_if_needed(p, es, CFG, T0 + timedelta(minutes=2), fake_sender(box))     # duplicate suppressed
    assert r["reason"] == "not_needed" and len(box) == 1
    p2 = mail_plan()
    assert R.notify_if_needed(p2, es, {**CFG, "api_key": ""}, T0, fake_sender(box))["reason"] == "not_configured"
    assert R.notify_if_needed(plan(), es, CFG, T0, fake_sender(box))["reason"] == "no_recipients"
    r = R.notify_if_needed(mail_plan(), es, CFG, T0, fake_sender([], ok=False))
    assert r["reason"] == "error" and not r["sent"]

def test_email_content_detail_modes():
    p = mail_plan(); es = [E(0, hr=44, sbp=92, dbp=55, symptoms=["presyncope"], notes="<b>x</b>")]
    al = R.evaluate_entry(p, es[0], [])
    subj, text, html = R.build_alert_email(p, es[0], al, "en", "https://app.example", "me@example.com")
    assert "Test Patient" in subj and "166" in text and "Pulse 44" in text and "https://app.example" in text
    assert "me@example.com" in text and "<b>x</b>" not in html
    p["notify"]["detail"] = False
    subj, text, html = R.build_alert_email(p, es[0], al, "en")
    assert "Test Patient" not in subj + text + html and "44" not in text
    subj, text, html = R.build_alert_email(p, es[0], [], "el", test=True)
    assert "ΔΟΚΙΜΗ" in subj

def test_send_resend_mocked():
    import urllib.request as ur, io, json as js
    seen = {}
    class Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): pass
    def ok(req, timeout=0):
        seen["h"] = dict(req.header_items()); seen["b"] = js.loads(req.data); return Resp(b'{"id":"abc"}')
    old = ur.urlopen; ur.urlopen = ok
    try:
        r = R.send_resend("key", "A <a@x.com>", ["b@x.com"], "s", "t", "<p>t</p>", idem_key="k1")
        assert r == {"ok": True, "id": "abc", "error": None}
        assert seen["h"]["Authorization"] == "Bearer key" and seen["h"]["Idempotency-key"] == "k1" and seen["b"]["to"] == ["b@x.com"]
        def bad(req, timeout=0): raise OSError("net down")
        ur.urlopen = bad
        r = R.send_resend("key", "A <a@x.com>", ["b@x.com"], "s", "t", "h")
        assert not r["ok"] and "net down" in r["error"]
    finally:
        ur.urlopen = old
    assert R.send_resend("", "a", ["b"], "s", "t", "h")["error"] == "not_configured"

if __name__ == "__main__":
    n = 0
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); n += 1; print("ok", k)
    print(n, "passed")
