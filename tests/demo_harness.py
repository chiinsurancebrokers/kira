"""Visual demo with SYNTHETIC data only:  streamlit run tests/demo_harness.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime, timedelta
import streamlit as st
from types import SimpleNamespace as NS
import recovery as R, recovery_ui
st.set_page_config(page_title="Recovery demo", layout="centered")
if "_rec" not in st.session_state:
    now = datetime.now(R.TZ)
    p = R.new_plan("Demo Patient", 81, "F", "tavi", (now - timedelta(days=1)).date().isoformat())
    p["hospital_course"] = "ICU stay after the procedure; ventilator weaned; blood pressure unstable (synthetic demo)."
    p["baseline"]["hr_min_known"] = 36
    p["meds"] = [{"name": "Salospir 100 mg", "dose": "100 mg", "times": ["noon"], "critical": True, "note": ""},
                 {"name": "Concor", "dose": "1/4 x 10 mg", "times": ["morning"], "critical": True, "note": ""}]
    p["contacts"] = [{"label": "Cardiology clinic", "phone": "210 0000000"}]
    es = []
    def add(h, **k): es.append(R.new_entry(ts=(now - timedelta(hours=h)).isoformat(timespec="seconds"), **k))
    add(30, hr=70, sbp=128, dbp=76, spo2=95, temp=36.7, weight=60.2)
    add(24, hr=66, sbp=118, dbp=70, spo2=95, weight=60.0)
    add(14, hr=58, sbp=104, dbp=62, spo2=94, temp=36.9)
    add(8, hr=54, sbp=96, dbp=58, spo2=94, symptoms=["fatigue"])
    add(1.2, hr=44, sbp=92, dbp=55, spo2=93, symptoms=["presyncope"], notes="Dizzy when standing up")
    st.session_state["_rec"] = {"plans": [p], "entries": {p["id"]: es}, "active": p["id"], "loaded": True,
                                "persist_ok": False, "just_saved": None, "flash": None}
st.session_state.setdefault("lang", "en")
def header(*a, **k): st.title(a[1])
deps = NS(lang=st.session_state["lang"], email="", supabase=lambda: None, fernet=lambda: None, api_key=lambda: "",
          log=lambda *a, **k: None, rate_gate=lambda n: True, heic=None, downscale=None, header=header,
          profile=lambda: {}, dossier_meds=None)
recovery_ui.render(deps)
