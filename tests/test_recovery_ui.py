"""UI scenarios through Streamlit's AppTest. Run: python tests/test_recovery_ui.py"""
import os, sys, warnings, logging
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
HERE = os.path.dirname(os.path.abspath(__file__))
from streamlit.testing.v1 import AppTest

def banner(at):
    return " ".join(m.value for m in at.markdown if "rc-banner" in m.value)

def submit(at):
    b = [b for b in at.button if "Save & check" in b.label][0]; b.click(); return at.run()

def fresh():
    at = AppTest.from_file(os.path.join(HERE, "ui_harness.py"), default_timeout=30).run()
    at.button(key="rc_new_tavi").click().run(); assert not at.exception, at.exception; return at

def test_emergency_flow():
    at = fresh()
    at.number_input[0].set_value(38)                # pulse
    at.checkbox(key="rc_s_presyncope").check()
    at = submit(at); assert not at.exception, at.exception
    b = banner(at)
    assert "#DC2626" in b and "tel:166" in b and "tel:112" in b, b[:300]

def test_ok_flow_and_followup_entries():
    at = fresh()
    at.number_input[0].set_value(72); at.number_input[1].set_value(128); at.number_input[2].set_value(76)
    at = submit(at); assert not at.exception, at.exception
    assert "#059669" in banner(at)
    assert len(at.session_state["_rec"]["entries"][at.session_state["_rec"]["active"]]) == 1

def test_typing_error_blocked():
    at = fresh()
    at.number_input[0].set_value(240); at.number_input[1].set_value(50); at.number_input[2].set_value(90)  # dbp>sbp
    at = submit(at)
    assert any("Diastolic" in e.value for e in at.error), [e.value for e in at.error]
    assert not at.session_state["_rec"]["entries"][at.session_state["_rec"]["active"]]

def test_nothing_entered():
    at = fresh(); at = submit(at)
    assert any("Nothing" in w.value for w in at.warning)

def test_tabs_export_and_plan_render():
    at = fresh()
    for h, s in ((72, 128), (48, 96)):
        at.number_input[0].set_value(h); at.number_input[1].set_value(s); at.number_input[2].set_value(70)
        at = submit(at)
    assert not at.exception, at.exception
    st = at.session_state["_rec"]; p = next(x for x in st["plans"] if x["id"] == st["active"])
    assert p["thresholds_source"] == "default"
    assert any("NOT been confirmed" in w.value for w in at.warning)      # default-threshold warning
    plan_save = [b for b in at.button if "Save plan" in b.label][0]; plan_save.click(); at.run()
    assert not at.exception, at.exception

if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("ok", k)
