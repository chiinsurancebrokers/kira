"""Fingertip-camera pulse component (components/pulse): declared in its own
module so Streamlit can register it under a stable module name."""
import os
import streamlit.components.v1 as _components

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "components", "pulse")
pulse_component = _components.declare_component("asklepios_pulse", path=_DIR)
