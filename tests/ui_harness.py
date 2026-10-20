import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
from types import SimpleNamespace as NS
import recovery_ui
def header(*a, **k): st.title(a[1])
deps = NS(lang=st.session_state.get("lang", "en"), email="", supabase=lambda: None, fernet=lambda: None,
          api_key=lambda: "", log=lambda *a, **k: None, rate_gate=lambda n: True, heic=None, downscale=None,
          header=header, profile=lambda: {"name": "Test Patient", "age": 81, "sex": "F"},
          dossier_meds=lambda: [{"drug": "Concor", "dose": "1/4 x 10 mg", "time": "morning"}])
recovery_ui.render(deps)
