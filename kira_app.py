"""
KIRA — AI Nurse
Bilingual AI health assistant for the Greek market.
Standalone Streamlit app · Real data only · No placeholders.
"""

import streamlit as st
import json
import io
import urllib.request
import urllib.parse
from datetime import datetime, date

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kira · AI Nurse",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── STYLING ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #F0F4FF 0%, #F8F0FF 100%);
}
[data-testid="stSidebar"] { display: none; }

.kira-hero {
    background: linear-gradient(135deg, #2D3FE7 0%, #7B2FE0 100%);
    border-radius: 20px;
    padding: 48px 40px;
    color: white;
    text-align: center;
    margin-bottom: 32px;
}
.kira-hero h1 { font-size: 52px; font-weight: 700; margin: 0; letter-spacing: -1px; }
.kira-hero p  { font-size: 18px; opacity: 0.85; margin: 12px 0 0; }
.kira-tagline { font-size: 13px; opacity: 0.65; margin-top: 8px; letter-spacing: 2px; text-transform: uppercase; }

.card {
    background: white;
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 20px;
    box-shadow: 0 2px 12px rgba(45,63,231,0.07);
    border: 1px solid rgba(45,63,231,0.08);
}
.card h3 { font-size: 16px; font-weight: 600; margin: 0 0 16px; color: #1A1A2E; }

.vital-badge {
    background: #F4F6FF;
    border: 1px solid #E0E5FF;
    border-radius: 12px;
    padding: 14px 18px;
    min-width: 120px;
    text-align: center;
    flex: 1;
}
.vital-badge.green { background: #EDFBF0; border-color: #A3E6B5; }
.vital-badge.yellow { background: #FFFBEB; border-color: #FCD34D; }
.vital-badge.red { background: #FEF2F2; border-color: #FCA5A5; }
.vital-badge .vb-value { font-size: 22px; font-weight: 700; color: #1A1A2E; }
.vital-badge .vb-label { font-size: 11px; color: #6B7280; margin-top: 2px; }
.vital-badge .vb-unit  { font-size: 10px; color: #9CA3AF; }

.pill { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }
.pill-green  { background: #DCFCE7; color: #15803D; }
.pill-yellow { background: #FEF9C3; color: #A16207; }
.pill-red    { background: #FEE2E2; color: #B91C1C; }
.pill-blue   { background: #DBEAFE; color: #1D4ED8; }

.disclaimer {
    background: #FFFBEB;
    border: 1px solid #FCD34D;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 13px;
    color: #92400E;
    margin: 12px 0;
}
.disclaimer-red {
    background: #FEF2F2;
    border: 1px solid #FCA5A5;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 13px;
    color: #991B1B;
    margin: 12px 0;
}

.emergency {
    background: linear-gradient(90deg, #DC2626, #B91C1C);
    color: white; border-radius: 10px; padding: 16px 20px;
    font-weight: 600; font-size: 14px; margin: 12px 0;
}

.kira-stepper {
    display: flex; align-items: center; justify-content: center;
    gap: 0; margin: 0 0 28px; padding: 16px 0 0;
}
.kira-step {
    display: flex; flex-direction: column; align-items: center; gap: 6px;
    flex: 1; max-width: 120px;
}
.kira-step-circle {
    width: 32px; height: 32px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; font-weight: 700; border: 2px solid #E0E5FF;
    background: white; color: #CBD5E1; position: relative; z-index: 1;
}
.kira-step.done   .kira-step-circle { background: #7B2FE0; border-color: #7B2FE0; color: white; }
.kira-step.active .kira-step-circle { background: #2D3FE7; border-color: #2D3FE7; color: white; box-shadow: 0 0 0 4px rgba(45,63,231,.15); }
.kira-step-label { font-size: 10px; color: #94A3B8; text-align: center; letter-spacing: .02em; }
.kira-step.done   .kira-step-label  { color: #7B2FE0; }
.kira-step.active .kira-step-label  { color: #2D3FE7; font-weight: 600; }
.kira-step-line {
    flex: 1; height: 2px; background: #E0E5FF; margin-bottom: 18px;
}
.kira-step-line.done { background: #7B2FE0; }

.chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 16px; }
.chip {
    padding: 6px 14px; border-radius: 20px; font-size: 13px; cursor: pointer;
    border: 1.5px solid #C4B5FD; color: #5B21B6; background: #F5F3FF;
    transition: all .15s; user-select: none;
}
.chip.selected { background: #7B2FE0; border-color: #7B2FE0; color: white; }

.wellness-wrap {
    display: flex; align-items: center; gap: 20px;
    background: linear-gradient(135deg,#2D3FE7,#7B2FE0);
    border-radius: 16px; padding: 20px 24px; margin-bottom: 20px; color: white;
}
.wellness-score { font-size: 48px; font-weight: 800; letter-spacing: -2px; }
.wellness-label { font-size: 12px; opacity: .7; text-transform: uppercase; letter-spacing: 1.5px; }
.wellness-desc  { font-size: 15px; opacity: .9; margin-top: 4px; }

.red-flags-urgent {
    background: linear-gradient(90deg,#DC2626,#B91C1C);
    color: white; border-radius: 12px; padding: 16px 20px; margin: 12px 0;
    animation: pulse-bg 2s ease-in-out infinite;
}
@keyframes pulse-bg { 0%,100%{opacity:1} 50%{opacity:.85} }

/* Mobile */
@media (max-width: 768px) {
    .kira-hero h1 { font-size: 32px !important; }
    .kira-hero { padding: 28px 20px !important; }
    .stChatMessage { font-size: 14px !important; }
    [data-testid="stChatMessageContent"] { max-width: 100% !important; overflow-wrap: break-word !important; }
    .main .block-container { padding-bottom: 120px !important; }
    .stButton button { white-space: normal !important; min-height: 44px !important; }
}
[data-testid="stMarkdownContainer"] { overflow-wrap: break-word !important; word-break: break-word !important; }
</style>
""", unsafe_allow_html=True)

# ── KEYS ──────────────────────────────────────────────────────────────────────
def _key(name, fallback=""):
    for k in [name, name.lower(), name.upper()]:
        v = st.secrets.get(k, "")
        if v:
            return v
    return fallback

def get_claude_key():  return _key("Claude_API_Key")
def get_openai_key():  return _key("OPENAI_API_KEY", "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")
def get_ncbi_key():    return _key("NCBI_API_KEY", "5bcced38b6d0cbb9998281811cbf56c9ac09")

# ── NCBI HELPERS ──────────────────────────────────────────────────────────────
def pubmed_search(query, n=3):
    try:
        p = urllib.parse.urlencode({"db":"pubmed","term":query,"retmax":n,"retmode":"json","api_key":get_ncbi_key()})
        with urllib.request.urlopen(f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?{p}", timeout=8) as r:
            ids = json.loads(r.read()).get("esearchresult",{}).get("idlist",[])
        if not ids: return []
        p2 = urllib.parse.urlencode({"db":"pubmed","id":",".join(ids),"retmode":"json","api_key":get_ncbi_key()})
        with urllib.request.urlopen(f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?{p2}", timeout=8) as r:
            res = json.loads(r.read()).get("result",{})
        out = []
        for pmid in ids:
            a = res.get(pmid,{})
            out.append({
                "pmid": pmid, "title": a.get("title","—"),
                "authors": ", ".join(x.get("name","") for x in a.get("authors",[])[:2]),
                "journal": a.get("source",""), "date": a.get("pubdate",""),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            })
        return out
    except: return []

def rxnorm_interactions(names):
    try:
        cuis = []
        for name in names:
            p = urllib.parse.urlencode({"name": name.split()[0]})
            with urllib.request.urlopen(f"https://rxnav.nlm.nih.gov/REST/rxcui.json?{p}", timeout=6) as r:
                ids = json.loads(r.read()).get("idGroup",{}).get("rxnormId",[])
                if ids: cuis.append(ids[0])
        if len(cuis) < 2: return None
        p2 = urllib.parse.urlencode({"rxcuis": " ".join(cuis)})
        with urllib.request.urlopen(f"https://rxnav.nlm.nih.gov/REST/interaction/list.json?{p2}", timeout=8) as r:
            data = json.loads(r.read())
        pairs = data.get("fullInteractionTypeGroup",[])
        if not pairs: return "\u2705 RxNorm: No known interactions found."
        lines = []
        for g in pairs:
            src = g.get("sourceName","")
            for t2 in g.get("fullInteractionType",[]):
                for pair in t2.get("interactionPair",[]):
                    sev  = pair.get("severity","")
                    desc = pair.get("description","")
                    drugs = " + ".join(c.get("minConceptItem",{}).get("name","") for c in pair.get("interactionConcept",[]))
                    lines.append(f"- **{drugs}** [{sev}] \u2014 {desc} *({src})*")
        return "\n".join(lines) if lines else "\u2705 RxNorm: No known interactions found."
    except: return None

# ── GPT-4o ────────────────────────────────────────────────────────────────────
def gpt4o(prompt, system="", max_tokens=900):
    try:
        oai = get_openai_key()
        if not oai: return None
        body = json.dumps({
            "model": "gpt-4o",
            "max_tokens": max_tokens,
            "messages": [{"role":"system","content":system},{"role":"user","content":prompt}] if system
                        else [{"role":"user","content":prompt}],
        }).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions", data=body,
            headers={"Content-Type":"application/json","Authorization":f"Bearer {oai}"}
        )
        with urllib.request.urlopen(req, timeout=25) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]
    except Exception as e:
        return f"GPT-4o unavailable: {e}"

# ── CLAUDE ────────────────────────────────────────────────────────────────────
def claude(messages, system="", max_tokens=1200, timeout=60):
    """Call Claude via raw HTTP."""
    key = get_claude_key()
    if not key:
        return "\u26a0\ufe0f Claude API key not set."
    body = json.dumps({
        "model": "claude-sonnet-4-6",
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        return data["content"][0]["text"]
    except urllib.error.URLError as e:
        if "timed out" in str(e).lower() or "timeout" in str(e).lower():
            return "\u26a0\ufe0f Request timed out. Please try again."
        return f"\u26a0\ufe0f Claude error: {e}"
    except Exception as e:
        return f"\u26a0\ufe0f Claude error: {e}"

# ── SESSION STATE ─────────────────────────────────────────────────────────────
defaults = {
    "lang": "el",
    "screen": "home",
    "profile": {},
    "vitals": {},
    "vitals_analysis": "",
    "triage_chat": [],
    "triage_ready": False,
    "report": "",
    "report_pubmed": [],
    "report_gpt": "",
    "medications": [],
    "med_inputs": [],
    "symptom_chips": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── TRANSLATIONS ──────────────────────────────────────────────────────────────
T = {
    "el": {
        "title": "Kira",
        "subtitle": "\u039f AI \u039d\u03bf\u03c3\u03b7\u03bb\u03b5\u03c5\u03c4\u03ae\u03c2 \u03c3\u03bf\u03c5",
        "tagline": "\u0388\u03b3\u03ba\u03c5\u03c1\u03b7 \u03b9\u03b1\u03c4\u03c1\u03b9\u03ba\u03ae \u03c0\u03bb\u03b7\u03c1\u03bf\u03c6\u03cc\u03c1\u03b7\u03c3\u03b7 \u00b7 \u03a0\u03ac\u03bd\u03c4\u03b1 \u03b4\u03af\u03c0\u03bb\u03b1 \u03c3\u03bf\u03c5",
        "start": "\u039e\u03b5\u03ba\u03af\u03bd\u03b1 \u0395\u03ba\u03c4\u03af\u03bc\u03b7\u03c3\u03b7",
        "disclaimer_main": "\u26a0\ufe0f \u0397 Kira \u03c0\u03b1\u03c1\u03ad\u03c7\u03b5\u03b9 \u03c0\u03bb\u03b7\u03c1\u03bf\u03c6\u03bf\u03c1\u03af\u03b5\u03c2 \u03c5\u03b3\u03b5\u03af\u03b1\u03c2 \u03b1\u03c0\u03bf\u03ba\u03bb\u03b5\u03b9\u03c3\u03c4\u03b9\u03ba\u03ac \u03b3\u03b9\u03b1 \u03b5\u03bd\u03b7\u03bc\u03b5\u03c1\u03c9\u03c4\u03b9\u03ba\u03bf\u03cd\u03c2 \u03c3\u03ba\u03bf\u03c0\u03bf\u03cd\u03c2. \u0394\u03b5\u03bd \u03b1\u03bd\u03c4\u03b9\u03ba\u03b1\u03b8\u03b9\u03c3\u03c4\u03ac \u03b9\u03b1\u03c4\u03c1\u03b9\u03ba\u03ae \u03b4\u03b9\u03ac\u03b3\u03bd\u03c9\u03c3\u03b7 \u03ae \u03b8\u03b5\u03c1\u03b1\u03c0\u03b5\u03af\u03b1. \u03a3\u03b5 \u03b5\u03c0\u03b5\u03af\u03b3\u03bf\u03c5\u03c3\u03b1 \u03b1\u03bd\u03ac\u03b3\u03ba\u03b7 \u03ba\u03b1\u03bb\u03ad\u03c3\u03c4\u03b5 **166** (\u0395\u039a\u0391\u0392) \u03ae **112**.",
        "emergency": "\U0001f6a8 \u03a3\u0395 \u0395\u03a0\u0395\u0399\u0393\u039f\u03a5\u03a3\u0391 \u0391\u039d\u0391\u0393\u039a\u0397: \u039a\u0391\u039b\u0395\u03a3\u03a4\u0395 166 (\u0395\u039a\u0391\u0392) \u03ae 112",
        "name": "\u038c\u03bd\u03bf\u03bc\u03b1", "age": "\u0397\u03bb\u03b9\u03ba\u03af\u03b1", "sex": "\u03a6\u03cd\u03bb\u03bf",
        "male": "\u0386\u03bd\u03b4\u03c1\u03b1\u03c2", "female": "\u0393\u03c5\u03bd\u03b1\u03af\u03ba\u03b1", "other": "\u0386\u03bb\u03bb\u03bf",
        "history": "\u0399\u03b1\u03c4\u03c1\u03b9\u03ba\u03cc \u03b9\u03c3\u03c4\u03bf\u03c1\u03b9\u03ba\u03cc (\u03c0\u03c1\u03bf\u03b7\u03b3\u03bf\u03cd\u03bc\u03b5\u03bd\u03b5\u03c2 \u03c0\u03b1\u03b8\u03ae\u03c3\u03b5\u03b9\u03c2, \u03c7\u03b5\u03b9\u03c1\u03bf\u03c5\u03c1\u03b3\u03b5\u03af\u03b1)",
        "allergies": "\u0391\u03bb\u03bb\u03b5\u03c1\u03b3\u03af\u03b5\u03c2",
        "meds": "\u03a4\u03c1\u03ad\u03c7\u03bf\u03bd\u03c4\u03b1 \u03c6\u03ac\u03c1\u03bc\u03b1\u03ba\u03b1 / \u03c3\u03c5\u03bc\u03c0\u03bb\u03b7\u03c1\u03ce\u03bc\u03b1\u03c4\u03b1",
        "next": "\u0395\u03c0\u03cc\u03bc\u03b5\u03bd\u03bf \u2192",
        "back": "\u2190 \u03a0\u03af\u03c3\u03c9",
        "vitals_title": "\u0396\u03c9\u03c4\u03b9\u03ba\u03ad\u03c2 \u0395\u03bd\u03b4\u03b5\u03af\u03be\u03b5\u03b9\u03c2",
        "vitals_sub": "\u0395\u03b9\u03c3\u03ac\u03b3\u03b5\u03c4\u03b5 \u03c4\u03b9\u03c2 \u03bc\u03b5\u03c4\u03c1\u03ae\u03c3\u03b5\u03b9\u03c2 \u03c3\u03b1\u03c2.",
        "hr": "\u039a\u03b1\u03c1\u03b4\u03b9\u03b1\u03ba\u03cc\u03c2 \u03a1\u03c5\u03b8\u03bc\u03cc\u03c2 (bpm)",
        "bp_sys": "\u0391\u03c1\u03c4\u03b7\u03c1\u03b9\u03b1\u03ba\u03ae \u03a0\u03af\u03b5\u03c3\u03b7 \u2014 \u03a3\u03c5\u03c3\u03c4\u03bf\u03bb\u03b9\u03ba\u03ae (mmHg)",
        "bp_dia": "\u0391\u03c1\u03c4\u03b7\u03c1\u03b9\u03b1\u03ba\u03ae \u03a0\u03af\u03b5\u03c3\u03b7 \u2014 \u0394\u03b9\u03b1\u03c3\u03c4\u03bf\u03bb\u03b9\u03ba\u03ae (mmHg)",
        "br": "\u0391\u03bd\u03b1\u03c0\u03bd\u03b5\u03c5\u03c3\u03c4\u03b9\u03ba\u03cc\u03c2 \u03a1\u03c5\u03b8\u03bc\u03cc\u03c2 (/min)",
        "spo2": "SpO2 (%)",
        "temp": "\u0398\u03b5\u03c1\u03bc\u03bf\u03ba\u03c1\u03b1\u03c3\u03af\u03b1 (\u00b0C)",
        "weight": "\u0392\u03ac\u03c1\u03bf\u03c2 (kg)",
        "height": "\u038e\u03c8\u03bf\u03c2 (cm)",
        "analyse_vitals": "\u0391\u03bd\u03ac\u03bb\u03c5\u03c3\u03b7 \u0396\u03c9\u03c4\u03b9\u03ba\u03ce\u03bd",
        "triage_title": "\u0395\u03ba\u03c4\u03af\u03bc\u03b7\u03c3\u03b7 \u03a3\u03c5\u03bc\u03c0\u03c4\u03c9\u03bc\u03ac\u03c4\u03c9\u03bd",
        "triage_sub": "\u03a0\u03b5\u03c1\u03b9\u03b3\u03c1\u03ac\u03c8\u03c4\u03b5 \u03c4\u03b1 \u03c3\u03c5\u03bc\u03c0\u03c4\u03ce\u03bc\u03b1\u03c4\u03ac \u03c3\u03b1\u03c2. \u0397 Kira \u03b8\u03b1 \u03c3\u03b1\u03c2 \u03ba\u03ac\u03bd\u03b5\u03b9 \u03ba\u03b1\u03c4\u03b5\u03c5\u03b8\u03c5\u03bd\u03cc\u03bc\u03b5\u03bd\u03b5\u03c2 \u03b5\u03c1\u03c9\u03c4\u03ae\u03c3\u03b5\u03b9\u03c2.",
        "triage_placeholder": "\u03a0.\u03c7. \u0388\u03c7\u03c9 \u03c0\u03bf\u03bd\u03bf\u03ba\u03ad\u03c6\u03b1\u03bb\u03bf \u03c4\u03c1\u03b9\u03ce\u03bd \u03b7\u03bc\u03b5\u03c1\u03ce\u03bd \u03bc\u03b5 \u03bd\u03b1\u03c5\u03c4\u03af\u03b1...",
        "generate_report": "\u0394\u03b7\u03bc\u03b9\u03bf\u03c5\u03c1\u03b3\u03af\u03b1 \u03a0\u03bb\u03ae\u03c1\u03bf\u03c5\u03c2 \u0391\u03bd\u03b1\u03c6\u03bf\u03c1\u03ac\u03c2",
        "report_title": "\u039b\u03b5\u03c0\u03c4\u03bf\u03bc\u03b5\u03c1\u03ae\u03c2 \u0395\u03ba\u03c4\u03af\u03bc\u03b7\u03c3\u03b7 \u03a5\u03b3\u03b5\u03af\u03b1\u03c2",
        "second_opinion": "\u0394\u03b5\u03cd\u03c4\u03b5\u03c1\u03b7 \u0393\u03bd\u03ce\u03bc\u03b7 GPT-4o",
        "pubmed": "\u0395\u03c0\u03b9\u03c3\u03c4\u03b7\u03bc\u03bf\u03bd\u03b9\u03ba\u03ad\u03c2 \u0391\u03bd\u03b1\u03c6\u03bf\u03c1\u03ad\u03c2 PubMed",
        "skip_vitals": "\u03a0\u03b1\u03c1\u03ac\u03bb\u03b5\u03b9\u03c8\u03b7 (\u03c7\u03c9\u03c1\u03af\u03c2 \u03bc\u03b5\u03c4\u03c1\u03ae\u03c3\u03b5\u03b9\u03c2)",
    },
    "en": {
        "title": "Kira",
        "subtitle": "Your AI Nurse",
        "tagline": "Evidence-based health guidance · Always by your side",
        "start": "Start Assessment",
        "disclaimer_main": "\u26a0\ufe0f Kira provides health information for informational purposes only. It does not replace medical diagnosis or treatment. In an emergency call **166** (EKAB) or **112**.",
        "emergency": "\U0001f6a8 EMERGENCY: CALL 166 (EKAB) or 112",
        "name": "Name", "age": "Age", "sex": "Biological Sex",
        "male": "Male", "female": "Female", "other": "Other",
        "history": "Medical history (conditions, surgeries)",
        "allergies": "Allergies",
        "meds": "Current medications / supplements",
        "next": "Next \u2192",
        "back": "\u2190 Back",
        "vitals_title": "Your Vitals",
        "vitals_sub": "Enter your measurements.",
        "hr": "Heart Rate (bpm)",
        "bp_sys": "Blood Pressure \u2014 Systolic (mmHg)",
        "bp_dia": "Blood Pressure \u2014 Diastolic (mmHg)",
        "br": "Breathing Rate (/min)",
        "spo2": "SpO2 (%)",
        "temp": "Temperature (\u00b0C)",
        "weight": "Weight (kg)",
        "height": "Height (cm)",
        "analyse_vitals": "Analyse Vitals",
        "triage_title": "Symptom Assessment",
        "triage_sub": "Describe your symptoms. Kira will ask targeted follow-up questions.",
        "triage_placeholder": "E.g. I have had a headache for three days with nausea...",
        "generate_report": "Generate Full Clinical Report",
        "report_title": "Detailed Health Assessment",
        "second_opinion": "GPT-4o Second Opinion",
        "pubmed": "PubMed Evidence",
        "skip_vitals": "Skip (no measurements)",
    }
}

def t(key): return T[st.session_state.lang].get(key, key)


def render_stepper(current):
    steps_el = ["1 Στοιχεία","2 Ζωτικές","3 Συμπτώματα","4 Αναφορά"]
    steps_en = ["1 Profile","2 Vitals","3 Symptoms","4 Report"]
    steps = steps_el if st.session_state.lang=="el" else steps_en
    order = ["intake","vitals","triage","report"]
    cur_i = order.index(current) if current in order else 0
    html = '<div class="kira-stepper">'
    for i, label in enumerate(steps):
        cls = "done" if i < cur_i else ("active" if i == cur_i else "")
        icon = "✓" if i < cur_i else str(i+1)
        html += f'<div class="kira-step {cls}"><div class="kira-step-circle">{icon}</div><div class="kira-step-label">{label}</div></div>'
        if i < len(steps)-1:
            line_cls = "done" if i < cur_i else ""
            html += f'<div class="kira-step-line {line_cls}"></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

def classify_vitals(v):
    status = {}
    hr = v.get("hr")
    if hr:
        if 60<=hr<=100: status["hr"]="green"
        elif 50<=hr<=110: status["hr"]="yellow"
        else: status["hr"]="red"
    sys = v.get("bp_sys"); dia = v.get("bp_dia")
    if sys and dia:
        if sys<120 and dia<80: status["bp"]="green"
        elif sys<130: status["bp"]="yellow"
        elif sys<140 or dia<90: status["bp"]="yellow"
        else: status["bp"]="red"
    br = v.get("br")
    if br:
        if 12<=br<=20: status["br"]="green"
        elif 10<=br<=24: status["br"]="yellow"
        else: status["br"]="red"
    spo2 = v.get("spo2")
    if spo2:
        if spo2>=95: status["spo2"]="green"
        elif spo2>=90: status["spo2"]="yellow"
        else: status["spo2"]="red"
    temp = v.get("temp")
    if temp:
        if 36.1<=temp<=37.2: status["temp"]="green"
        elif 37.3<=temp<=38.0: status["temp"]="yellow"
        else: status["temp"]="red"
    w=v.get("weight"); h=v.get("height")
    if w and h:
        bmi=w/((h/100)**2); v["bmi"]=round(bmi,1)
        if 18.5<=bmi<=24.9: status["bmi"]="green"
        elif 25<=bmi<=29.9: status["bmi"]="yellow"
        else: status["bmi"]="red"
    return status

KIRA_SYSTEM_EL = """Είσαι η Kira — AI νοσηλευτής για Έλληνες χρήστες. Είσαι κλινικά ακριβής, άμεση και υποστηρικτική.
Ρόλος: Τριάζ συμπτωμάτων (μία ερώτηση κάθε φορά), ερμηνεία ζωτικών, φάρμακα, ελληνικό σύστημα υγείας (ΕΟΠΥΥ, ΕΟΔΥ, ΕΟΦ).
Κανόνες: Πάντα συστήνεις επαγγελματία. Κόκκινες σημαίες → 166/112. Όταν έχεις αρκετά: "Έχω αρκετά στοιχεία — μπορούμε να δημιουργήσουμε πλήρη αναφορά." Μία ερώτηση κάθε φορά."""

KIRA_SYSTEM_EN = """You are Kira — an AI nurse for users in Greece. Clinically accurate, direct, supportive.
Role: Symptom triage (one question at a time), vitals interpretation, medications, Greek health system (EOPYY, EODY, EOF).
Rules: Always recommend a professional. Red flags → 166/112. When ready: "I have enough information — we can generate a full clinical report." One question at a time."""

def kira_system(): return KIRA_SYSTEM_EL if st.session_state.lang=="el" else KIRA_SYSTEM_EN

def generate_html_report(profile, vitals, report_text, pubmed_refs, lang="el"):
    import re as _re, html as _html
    name=_html.escape(str(profile.get("name","—"))); age=str(profile.get("age","—"))
    sex=_html.escape(str(profile.get("sex",""))); hx=_html.escape(str(profile.get("history","") or "—"))
    allg=_html.escape(str(profile.get("allergies","") or "—")); meds=_html.escape(str(profile.get("meds_raw","") or "—"))
    ts=datetime.now().strftime("%d %B %Y  %H:%M")
    VLABELS={"hr":("Καρδιακός Ρυθμός","bpm"),"bp_sys":("ΑΠ Συστολική","mmHg"),"bp_dia":("ΑΠ Διαστολική","mmHg"),"br":("Αναπνευστικός Ρυθμός","/min"),"spo2":("SpO2","%"),"temp":("Θερμοκρασία","°C"),"weight":("Βάρος","kg"),"height":("Ύψος","cm"),"bmi":("ΔΜΣ","kg/m²"),"hrv":("HRV","ms"),"stress":("Δείκτης Στρες","/100")}
    vitals_rows="".join(f"<tr><td>{VLABELS.get(k,(k,''))[0]}</td><td><strong>{_html.escape(str(val))}</strong> {VLABELS.get(k,(k,''))[1]}</td></tr>" for k,val in (vitals or {}).items())
    vitals_sec=f"<h2>Ζωτικές Ενδείξεις</h2><table class='vitals'><thead><tr><th>Παράμετρος</th><th>Τιμή</th></tr></thead><tbody>{vitals_rows}</tbody></table>" if vitals_rows else ""
    def md2h(text):
        out=[]
        for line in text.splitlines():
            l=line.strip()
            if not l: out.append("<br>"); continue
            if l.startswith("## ") or l.startswith("# "): out.append(f"<h2>{_html.escape(l.lstrip('#').strip())}</h2>")
            elif l.startswith(("- ","* ","• ")): out.append(f"<li>{_re.sub(r'\*\*(.*?)\*\*',r'<strong>\1</strong>',_html.escape(l[2:]))}</li>")
            else: out.append(f"<p>{_re.sub(r'\*\*(.*?)\*\*',r'<strong>\1</strong>',_html.escape(l))}</p>")
        r="\n".join(out)
        return _re.sub(r"(<li>.*?</li>\n)+",lambda m:"<ul>"+m.group(0)+"</ul>",r,flags=_re.DOTALL)
    refs_html=""
    if pubmed_refs:
        refs_html="<h2>Βιβλιογραφία</h2><ol>"+"".join(f'<li>{_html.escape(a.get("title","—"))} — {_html.escape(a.get("authors",""))}. <em>{_html.escape(a.get("journal",""))}</em>, {_html.escape(a.get("date",""))}. <a href="{_html.escape(a.get("url",""))}">{_html.escape(a.get("url",""))}</a></li>' for a in pubmed_refs)+"</ol>"
    html_out=f"""<!DOCTYPE html><html lang="{lang}"><head><meta charset="UTF-8"><title>Kira Report — {name}</title>
<style>*{{box-sizing:border-box;margin:0;padding:0}}body{{font-family:'Inter',sans-serif;font-size:13px;color:#1A1A2E;max-width:820px;margin:0 auto;padding:32px 40px}}
.hdr{{display:flex;justify-content:space-between;align-items:center;border-bottom:3px solid #2D3FE7;padding-bottom:14px;margin-bottom:20px}}
.hdr-logo{{font-size:22px;font-weight:800;color:#2D3FE7}}.hdr-date{{font-size:11px;color:#6B7280;text-align:right}}
.patient{{background:linear-gradient(135deg,#2D3FE7,#7B2FE0);color:white;border-radius:12px;padding:18px 22px;margin-bottom:20px}}
.patient-name{{font-size:20px;font-weight:700;margin-bottom:4px}}.patient-meta{{font-size:12px;opacity:.8}}.patient-detail{{font-size:11px;opacity:.75;margin-top:10px;line-height:1.8}}
h2{{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#7B2FE0;border-bottom:1px solid #E0E5FF;padding-bottom:5px;margin:20px 0 10px}}
p{{margin:4px 0;line-height:1.65}}ul{{margin:6px 0 6px 18px}}li{{margin:3px 0;line-height:1.6}}
table.vitals{{width:100%;border-collapse:collapse;margin:10px 0;font-size:12px}}
table.vitals thead tr{{background:#2D3FE7;color:white}}table.vitals th,table.vitals td{{padding:7px 12px;text-align:left;border:1px solid #E0E5FF}}
table.vitals tbody tr:nth-child(even){{background:#F8FAFF}}
.emergency{{background:#DC2626;color:white;border-radius:8px;padding:12px 16px;font-weight:700;margin:16px 0}}
.disclaimer{{background:#FFFBEB;border:1px solid #FCD34D;border-radius:8px;padding:10px 14px;font-size:11px;color:#92400E;margin:12px 0}}
.hint{{text-align:center;margin:24px 0 0;font-size:12px;color:#94A3B8;border-top:1px dashed #E0E5FF;padding-top:14px}}
@media print{{body{{padding:16px}}.patient,.emergency{{-webkit-print-color-adjust:exact;print-color-adjust:exact}}@page{{margin:15mm}}}}</style></head><body>
<div class="hdr"><div class="hdr-logo">🩺 Kira AI Nurse</div><div class="hdr-date">Κλινική Εκτίμηση<br>{ts}</div></div>
<div class="patient"><div class="patient-name">{name}</div><div class="patient-meta">{age} ετών · {sex}</div>
<div class="patient-detail"><strong>Ιστορικό:</strong> {hx}<br><strong>Αλλεργίες:</strong> {allg}<br><strong>Φάρμακα:</strong> {meds}</div></div>
{vitals_sec}<h2>Κλινική Αξιολόγηση</h2>{md2h(report_text or "")}{refs_html}
<div class="emergency">🚨 ΣΕ ΕΠΕΙΓΟΥΣΑ ΑΝΑΓΚΗ: ΚΑΛΕΣΤΕ 166 (ΕΚΑΒ) ή 112</div>
<div class="disclaimer">⚠️ AI-generated. Δεν αποτελεί ιατρική διάγνωση. Απαιτείται επίσκεψη σε επαγγελματία υγείας.</div>
<div class="hint">💡 Ctrl+P → Save as PDF</div></body></html>"""
    return html_out.encode("utf-8")

def render_home():
    c1,c2=st.columns([6,1])
    with c2:
        if st.button("🇬🇧 EN" if st.session_state.lang=="el" else "🇬🇷 ΕΛ"):
            st.session_state.lang="en" if st.session_state.lang=="el" else "el"; st.rerun()
    st.markdown(f'''<div class="kira-hero"><div style="font-size:64px;margin-bottom:8px">🩺</div><h1>{t("title")}</h1><p>{t("subtitle")}</p><div class="kira-tagline">{t("tagline")}</div></div>''',unsafe_allow_html=True)
    st.markdown(f'<div class="disclaimer">{t("disclaimer_main")}</div>',unsafe_allow_html=True)
    col1,col2,col3=st.columns([1,2,1])
    with col2:
        if st.button(t("start"),type="primary",use_container_width=True):
            st.session_state.screen="intake"; st.rerun()
    st.markdown("---")
    f1,f2,f3=st.columns(3)
    with f1: st.markdown('<div class="card"><div style="font-size:32px">🔬</div><h3 style="margin-top:12px">PubMed Evidence</h3><p style="font-size:13px;color:#6B7280">Κάθε εκτίμηση υποστηρίζεται από επιστημονική βιβλιογραφία NCBI.</p></div>',unsafe_allow_html=True)
    with f2: st.markdown('<div class="card"><div style="font-size:32px">🤖</div><h3 style="margin-top:12px">Dual AI Engine</h3><p style="font-size:13px;color:#6B7280">Claude Sonnet + GPT-4o για διπλή κλινική γνώμη.</p></div>',unsafe_allow_html=True)
    with f3: st.markdown('<div class="card"><div style="font-size:32px">🇬🇷</div><h3 style="margin-top:12px">Ελληνικό Πλαίσιο</h3><p style="font-size:13px;color:#6B7280">ΕΟΠΥΥ, ΕΟΔΥ, ΕΟΦ — ελληνικό σύστημα υγείας.</p></div>',unsafe_allow_html=True)
    st.markdown(f'<div class="emergency">{t("emergency")}</div>',unsafe_allow_html=True)

def render_intake():
    render_stepper("intake")
    st.markdown(f"## 👤 {t('name')} & Ιστορικό")
    c1,c2,c3=st.columns([2,1,1])
    with c1: name=st.text_input(t("name"),value=st.session_state.profile.get("name",""),placeholder="Χριστόφορος")
    with c2: age=st.number_input(t("age"),min_value=1,max_value=120,value=st.session_state.profile.get("age",40))
    with c3: sex=st.selectbox(t("sex"),[t("male"),t("female"),t("other")])
    history=st.text_area(t("history"),value=st.session_state.profile.get("history",""),height=90,placeholder="Π.χ. Υπέρταση, Τ2 Διαβήτης")
    allergies=st.text_input(t("allergies"),value=st.session_state.profile.get("allergies",""),placeholder="Π.χ. Πενικιλλίνη")
    st.markdown("**"+t("meds")+"**")
    if not st.session_state.med_inputs:
        prev=st.session_state.profile.get("meds_raw","")
        st.session_state.med_inputs=[m.strip() for m in prev.split(",") if m.strip()] or [""]
    for mi,med_val in enumerate(st.session_state.med_inputs):
        mc1,mc2=st.columns([5,1])
        with mc1: st.session_state.med_inputs[mi]=st.text_input(f"Φάρμακο {mi+1}",value=med_val,key=f"med_field_{mi}",label_visibility="collapsed",placeholder="Π.χ. Metformin 500mg" if mi==0 else "")
        with mc2:
            if st.button("✕",key=f"del_med_{mi}"): st.session_state.med_inputs.pop(mi); st.rerun()
    if st.button("＋ "+("Προσθήκη" if st.session_state.lang=="el" else "Add med")): st.session_state.med_inputs.append(""); st.rerun()
    meds_raw=", ".join(m for m in st.session_state.med_inputs if m.strip())
    col_b,col_n=st.columns([1,3])
    with col_b:
        if st.button(t("back")): st.session_state.screen="home"; st.rerun()
    with col_n:
        if st.button(t("next"),type="primary",use_container_width=True):
            if name:
                st.session_state.profile={"name":name,"age":age,"sex":sex,"history":history,"allergies":allergies,"meds_raw":meds_raw}
                st.session_state.medications=[{"name":m.strip(),"freq":"","notes":""} for m in meds_raw.split(",") if m.strip()] if meds_raw else []
                if st.session_state.get("_from_facescan") and st.session_state.vitals:
                    st.session_state.screen="triage"
                else:
                    st.session_state.screen="vitals"
                st.rerun()
            else:
                st.warning("Παρακαλώ εισάγετε το όνομά σας." if st.session_state.lang=="el" else "Please enter your name.")

def render_vitals():
    render_stepper("vitals")
    p=st.session_state.profile
    st.markdown(f"## 📊 {t('vitals_title')} — {p.get('name','')}")
    facescan_url=st.secrets.get("FACESCAN_URL","https://kiraainurse.netlify.app")
    kira_url=st.secrets.get("KIRA_URL","https://kiraainurse.streamlit.app")
    if facescan_url:
        scan_link=f"{facescan_url}?kira_url={urllib.parse.quote(kira_url)}"
        st.markdown(f'''<div style="background:linear-gradient(135deg,#2D3FE7,#7B2FE0);border-radius:16px;padding:24px;text-align:center;margin-bottom:20px;color:white">
            <div style="font-size:36px">📷</div><div style="font-size:18px;font-weight:700;margin:8px 0">Σάρωση Προσώπου</div>
            <div style="font-size:13px;opacity:0.8;margin-bottom:16px">Μέτρηση HR, BP, HRV, stress σε 30 δευτερόλεπτα</div>
            <a href="{scan_link}" target="_blank" style="background:white;color:#2D3FE7;padding:12px 24px;border-radius:8px;font-weight:700;text-decoration:none;font-size:14px">Έναρξη Σάρωσης →</a></div>''',unsafe_allow_html=True)
        st.divider()
    st.markdown("**Χειροκίνητη Εισαγωγή**" if st.session_state.lang=="el" else "**Manual Entry**")
    v=st.session_state.vitals
    c1,c2,c3=st.columns(3)
    with c1:
        hr=st.number_input(t("hr"),min_value=0,max_value=300,value=int(v.get("hr",0)) or None,placeholder="76")
        spo2=st.number_input(t("spo2"),min_value=0,max_value=100,value=int(v.get("spo2",0)) or None,placeholder="98")
        temp=st.number_input(t("temp"),min_value=0.0,max_value=45.0,value=float(v.get("temp",0.0)) or None,placeholder="36.6",format="%.1f")
    with c2:
        bp_s=st.number_input(t("bp_sys"),min_value=0,max_value=300,value=int(v.get("bp_sys",0)) or None,placeholder="120")
        bp_d=st.number_input(t("bp_dia"),min_value=0,max_value=200,value=int(v.get("bp_dia",0)) or None,placeholder="80")
        br=st.number_input(t("br"),min_value=0,max_value=60,value=int(v.get("br",0)) or None,placeholder="15")
    with c3:
        weight=st.number_input(t("weight"),min_value=0.0,max_value=300.0,value=float(v.get("weight",0.0)) or None,placeholder="75",format="%.1f")
        height=st.number_input(t("height"),min_value=0,max_value=250,value=int(v.get("height",0)) or None,placeholder="175")
    col_b,col_s,col_n=st.columns([1,1,2])
    with col_b:
        if st.button(t("back")): st.session_state.screen="intake"; st.rerun()
    with col_s:
        if st.button(t("skip_vitals")): st.session_state.vitals={}; st.session_state.screen="triage"; st.rerun()
    with col_n:
        if st.button(t("analyse_vitals"),type="primary",use_container_width=True):
            vd={}
            if hr: vd["hr"]=hr
            if bp_s: vd["bp_sys"]=bp_s
            if bp_d: vd["bp_dia"]=bp_d
            if br: vd["br"]=br
            if spo2: vd["spo2"]=spo2
            if temp: vd["temp"]=temp
            if weight: vd["weight"]=weight
            if height: vd["height"]=height
            for extra in ["hrv","stress","cardio"]:
                if extra in st.session_state.vitals: vd[extra]=st.session_state.vitals[extra]
            st.session_state.vitals=vd; classify_vitals(vd)
            if vd:
                with st.spinner("Ανάλυση..."):
                    vtext="\n".join(f"- {k}: {val}" for k,val in vd.items())
                    pp=p.get
                    st.session_state.vitals_analysis=claude([{"role":"user","content":f"Patient: {pp('name')}, {pp('age')}yo {pp('sex')}, Hx: {pp('history','none')}, Meds: {pp('meds_raw','none')}\n\nVitals:\n{vtext}\n\nInterpret. Categorise each. Flag urgent findings. Be direct."}],system=kira_system(),max_tokens=1200)
            st.session_state.screen="triage"; st.rerun()

def render_vitals_summary():
    v=st.session_state.vitals
    if not v: return
    status=classify_vitals(v)
    LABELS={"hr":("❤️","Heart Rate","bpm"),"bp":("🩸","Blood Pressure","mmHg"),"br":("🌬️","Breathing","/min"),"spo2":("💧","SpO2","%"),"temp":("🌡️","Temp","°C"),"bmi":("⚖️","BMI","kg/m²")}
    badges=[]
    if "hr" in v: badges.append(("hr",v["hr"],"bpm",status.get("hr","green")))
    if "bp_sys" in v and "bp_dia" in v: badges.append(("bp",f"{v['bp_sys']}/{v['bp_dia']}","mmHg",status.get("bp","green")))
    if "br" in v: badges.append(("br",v["br"],"/min",status.get("br","green")))
    if "spo2" in v: badges.append(("spo2",v["spo2"],"%",status.get("spo2","green")))
    if "temp" in v: badges.append(("temp",v["temp"],"°C",status.get("temp","green")))
    if "bmi" in v: badges.append(("bmi",v["bmi"],"kg/m²",status.get("bmi","green")))
    if not badges: return
    cols=st.columns(len(badges))
    for i,(key,val,unit,col) in enumerate(badges):
        icon,label,_=LABELS.get(key,("","",""))
        with cols[i]:
            bg={"green":"#EDFBF0","yellow":"#FFFBEB","red":"#FEF2F2"}.get(col,"#F4F6FF")
            brd={"green":"#A3E6B5","yellow":"#FCD34D","red":"#FCA5A5"}.get(col,"#E0E5FF")
            st.markdown(f'<div style="background:{bg};border:1px solid {brd};border-radius:12px;padding:12px;text-align:center"><div style="font-size:18px">{icon}</div><div style="font-size:20px;font-weight:700">{val}</div><div style="font-size:10px;color:#6B7280">{unit}</div><div style="font-size:11px;color:#374151">{label}</div></div>',unsafe_allow_html=True)
    if st.session_state.vitals_analysis:
        with st.expander("📋 Ανάλυση ζωτικών" if st.session_state.lang=="el" else "📋 Vitals analysis"):
            st.markdown(st.session_state.vitals_analysis)

def render_triage():
    render_stepper("triage")
    p=st.session_state.profile
    st.markdown(f"## 🩺 {t('triage_title')} — {p.get('name','')}")
    render_vitals_summary()
    st.markdown(f'<div class="disclaimer">{t("disclaimer_main")}</div>',unsafe_allow_html=True)
    CHIPS_EL=["Πονοκέφαλος","Πυρετός","Βήχας","Δύσπνοια","Ναυτία","Πόνος στήθους","Κοιλιακός πόνος","Ζάλη","Κόπωση","Πόνος πλάτης","Διάρροια","Αιματοχεσία","Άλλο"]
    CHIPS_EN=["Headache","Fever","Cough","Shortness of breath","Nausea","Chest pain","Abdominal pain","Dizziness","Fatigue","Back pain","Diarrhoea","Blood in stool","Other"]
    chips=CHIPS_EL if st.session_state.lang=="el" else CHIPS_EN
    st.caption("Γρήγορη επιλογή:" if st.session_state.lang=="el" else "Quick select:")
    chip_row1=chips[:7]; chip_row2=chips[7:]
    cr1=st.columns(len(chip_row1))
    for ci,chip in enumerate(chip_row1):
        with cr1[ci]:
            sel=chip in st.session_state.symptom_chips
            if st.button(("✓ " if sel else "")+chip,key=f"chip_{ci}",use_container_width=True):
                if chip in st.session_state.symptom_chips: st.session_state.symptom_chips.remove(chip)
                else: st.session_state.symptom_chips.append(chip)
                st.rerun()
    cr2=st.columns(len(chip_row2))
    for ci,chip in enumerate(chip_row2):
        with cr2[ci]:
            sel=chip in st.session_state.symptom_chips
            if st.button(("✓ " if sel else "")+chip,key=f"chip2_{ci}",use_container_width=True):
                if chip in st.session_state.symptom_chips: st.session_state.symptom_chips.remove(chip)
                else: st.session_state.symptom_chips.append(chip)
                st.rerun()
    if st.session_state.symptom_chips:
        if st.button("➤ "+("Αποστολή επιλεγμένων" if st.session_state.lang=="el" else "Send selected"),type="primary"):
            msg=("Κύρια συμπτώματα: " if st.session_state.lang=="el" else "Main symptoms: ")+", ".join(st.session_state.symptom_chips)
            st.session_state.triage_chat.append({"role":"user","content":msg}); st.session_state.symptom_chips=[]; st.rerun()
    st.divider()
    for msg in st.session_state.triage_chat:
        with st.chat_message(msg["role"], avatar="🩺" if msg["role"]=="assistant" else None):
            st.markdown(msg["content"])
    ready_phrases=["έχω αρκετά στοιχεία","μπορούμε να δημιουργήσουμε","i have enough information","we can generate","full clinical report","πλήρη αναφορά"]
    last_kira=next((m["content"].lower() for m in reversed(st.session_state.triage_chat) if m["role"]=="assistant"),"")
    triage_ready=any(ph in last_kira for ph in ready_phrases)
    user_input=st.chat_input(t("triage_placeholder"),key="triage_input")
    if user_input:
        st.session_state.triage_chat.append({"role":"user","content":user_input})
        with st.spinner("Kira..."):
            pp=p.get
            profile_ctx=f"Patient: {pp('name')}, {pp('age')}yo {pp('sex')}, Hx: {pp('history','none')}, Allergies: {pp('allergies','none')}, Meds: {pp('meds_raw','none')}"
            vitals_ctx="Vitals: "+", ".join(f"{k}={val}" for k,val in st.session_state.vitals.items()) if st.session_state.vitals else "Vitals: not provided"
            system_ctx=kira_system()+f"\n\n{profile_ctx}\n{vitals_ctx}"
            reply=claude([{"role":m["role"],"content":m["content"]} for m in st.session_state.triage_chat],system=system_ctx,max_tokens=1500)
            if reply and reply.strip() and reply.strip()[-1] not in ".!?»)": reply=reply.rstrip()+" ..."
        st.session_state.triage_chat.append({"role":"assistant","content":reply}); st.rerun()
    col_b,col_r=st.columns([1,2])
    with col_b:
        if st.button(t("back")): st.session_state.screen="vitals"; st.rerun()
    with col_r:
        enabled=triage_ready or len(st.session_state.triage_chat)>=6
        if st.button(t("generate_report"),type="primary",use_container_width=True,disabled=not enabled):
            st.session_state.screen="report"; st.rerun()
    if not enabled:
        st.caption("Συνεχίστε — η Kira θα σας ειδοποιήσει όταν έχει αρκετά." if st.session_state.lang=="el" else "Continue — Kira will let you know when she has enough.")

def render_report():
    render_stepper("report")
    p=st.session_state.profile; lang=st.session_state.lang
    st.markdown(f"## 📋 {t('report_title')}"); st.caption(f"{p.get('name','')}, {p.get('age')}y · {datetime.now().strftime('%d %b %Y %H:%M')}")
    render_vitals_summary()
    if not st.session_state.report:
        conversation="\n".join(f"{'Patient' if m['role']=='user' else 'Kira'}: {m['content']}" for m in st.session_state.triage_chat)
        vitals_text="\n".join(f"- {k}: {v}" for k,v in st.session_state.vitals.items()) if st.session_state.vitals else "Not provided"
        vitals_analysis=st.session_state.vitals_analysis or "Not available"
        last_user=next((m["content"] for m in reversed(st.session_state.triage_chat) if m["role"]=="user"),"")
        search_query=last_user[:80]+" diagnosis management" if last_user else "symptom assessment management"
        with st.spinner("🔬 PubMed..." if lang=="el" else "🔬 Searching PubMed..."):
            refs=pubmed_search(search_query,n=3); st.session_state.report_pubmed=refs
        pubmed_ctx="\n".join(f"- {a['title']} ({a['journal']}, {a['date']}) {a['url']}" for a in refs) if refs else "None found."
        pp=p.get
        report_prompt=f"""Generate a concise clinical assessment for:
PATIENT: {pp('name')}, {pp('age')}yo {pp('sex')}
HISTORY: {pp('history','none')} | ALLERGIES: {pp('allergies','none')} | MEDS: {pp('meds_raw','none')}
VITALS: {vitals_text}
VITALS ANALYSIS: {vitals_analysis}
CONSULTATION: {conversation}
PUBMED: {pubmed_ctx}
Write: 1.CHIEF COMPLAINT 2.HISTORY 3.ASSESSMENT (primary dx + 2-3 differentials with %) 4.TREATMENT PLAN 5.RED FLAGS 6.PUBMED CITATIONS
Language: {"Greek" if lang=="el" else "English"}. Be direct. End with AI disclaimer."""
        with st.spinner("Δημιουργία αναφοράς..." if lang=="el" else "Generating report..."):
            result=claude([{"role":"user","content":report_prompt}],system=kira_system(),max_tokens=2000,timeout=120)
            if result.startswith("⚠️"):
                st.error(result)
                if st.button("🔄 Retry"): st.rerun()
                return
            st.session_state.report=result
    if not st.session_state.report:
        if st.button("🔄 "+("Δοκιμή ξανά" if lang=="el" else "Retry"),type="primary"): st.rerun()
        return
    st.markdown('<div class="card">',unsafe_allow_html=True)
    st.markdown(st.session_state.report)
    st.markdown('</div>',unsafe_allow_html=True)
    if st.session_state.report_pubmed:
        with st.expander(f"🔬 {t('pubmed')} ({len(st.session_state.report_pubmed)})"):
            for a in st.session_state.report_pubmed:
                st.markdown(f"**[{a['title']}]({a['url']})**  \n*{a['authors']} — {a['journal']}, {a['date']}*")
    if get_openai_key():
        with st.expander(f"🤖 {t('second_opinion')}"):
            if not st.session_state.report_gpt:
                if st.button("Get GPT-4o Second Opinion",type="secondary"):
                    with st.spinner("GPT-4o reviewing..."):
                        st.session_state.report_gpt=gpt4o(prompt=f"Patient: {p.get('name')}, {p.get('age')}yo\n\nClaude report:\n{st.session_state.report}\n\nAgree? Additions or corrections?",system=kira_system(),max_tokens=900)
                    st.rerun()
            else: st.markdown(st.session_state.report_gpt)
    if len(st.session_state.medications)>=2:
        with st.expander("💊 RxNorm" + (" — Έλεγχος Αλληλεπιδράσεων" if lang=="el" else " — Interactions")):
            with st.spinner("RxNorm..."): rxr=rxnorm_interactions([m["name"] for m in st.session_state.medications])
            if rxr: st.markdown(rxr)
    v=st.session_state.vitals
    if v.get("hr") or v.get("bp_sys"):
        status_map=classify_vitals(dict(v))
        reds=sum(1 for s in status_map.values() if s=="red"); yellows=sum(1 for s in status_map.values() if s=="yellow")
        wellness=max(20,100-reds*20-yellows*8)
        wcolor="#10B981" if wellness>=75 else "#F59E0B" if wellness>=50 else "#EF4444"
        wlabel=("Εξαιρετικό" if wellness>=85 else "Καλό" if wellness>=70 else "Μέτριο" if wellness>=50 else "Χρήζει Προσοχής") if lang=="el" else ("Excellent" if wellness>=85 else "Good" if wellness>=70 else "Moderate" if wellness>=50 else "Needs Attention")
        st.markdown(f'''<div class="wellness-wrap"><div><div class="wellness-score" style="color:{wcolor}">{wellness}</div><div class="wellness-label">Wellness Score</div></div><div style="flex:1"><div class="wellness-desc">{wlabel}</div><div style="background:rgba(255,255,255,.2);border-radius:99px;height:8px;margin-top:10px"><div style="background:{wcolor};width:{wellness}%;height:8px;border-radius:99px"></div></div></div></div>''',unsafe_allow_html=True)
    urgent_kw=["chest pain","πόνος στήθους","stroke","εγκεφαλικό","anaphylaxis","αναφυλαξία","166","112","emergency","επείγον","unconscious","αναίσθητος"]
    if any(kw in st.session_state.report.lower() for kw in urgent_kw):
        st.markdown('<div class="red-flags-urgent">🚨 Η αναφορά περιέχει <b>επείγουσες ενδείξεις</b>. Καλέστε <b>166</b> ή <b>112</b> αμέσως αν ισχύουν.</div>',unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="emergency">{t("emergency")}</div>',unsafe_allow_html=True)
    st.markdown('<div class="disclaimer-red">AI-generated. Δεν αντικαθιστά ιατρική γνώμη.</div>',unsafe_allow_html=True)
    fname=f"kira_report_{p.get('name','patient')}_{datetime.now().strftime('%Y%m%d')}"
    c1,c2,c3,c4=st.columns(4)
    with c1:
        if st.button("← "+("Νέα Αξιολόγηση" if lang=="el" else "New Assessment"),use_container_width=True):
            for k,vv in defaults.items(): st.session_state[k]=vv
            st.rerun()
    with c2:
        st.download_button("📄 TXT",data=st.session_state.report,file_name=fname+".txt",mime="text/plain",use_container_width=True)
    with c3:
        st.download_button("📄 PDF/HTML",data=generate_html_report(st.session_state.profile,st.session_state.vitals,st.session_state.report,st.session_state.report_pubmed,lang=lang),file_name=fname+".html",mime="text/html",use_container_width=True,help="Open in browser → Ctrl+P → Save as PDF")
    with c4:
        msg=f"Kira AI Nurse\nΑσθενής: {p.get('name','')} {p.get('age','')}y\n\n"
        if v.get("hr"): msg+=f"HR:{v['hr']}bpm "
        if v.get("bp_sys"): msg+=f"BP:{v['bp_sys']}/{v.get('bp_dia','?')}mmHg\n"
        msg+="\n---\nAI-generated. kiraainurse.streamlit.app"
        wa_url="https://wa.me/?text="+urllib.parse.quote(msg)
        st.markdown(f'<a href="{wa_url}" target="_blank" style="display:block;text-align:center;padding:8px;border-radius:8px;text-decoration:none;font-weight:600;font-size:13px;color:white;background:#25D366">WhatsApp</a>',unsafe_allow_html=True)

# ── FACESCAN INTERCEPTION ─────────────────────────────────────────────────────
try:
    _raw=st.query_params.get("facescan","")
    if _raw:
        _scanned=json.loads(urllib.parse.unquote(_raw))
        if _scanned and isinstance(_scanned,dict):
            st.session_state.vitals=_scanned
            st.session_state["_from_facescan"]=True
            if st.session_state.profile.get("name"): st.session_state.screen="triage"
            else: st.session_state.screen="intake"; st.session_state["_fs_banner"]=True
            st.query_params.clear(); st.rerun()
except Exception: pass

# ── ROUTER ────────────────────────────────────────────────────────────────────
screen=st.session_state.screen
if st.session_state.pop("_fs_banner",False):
    st.success("✅ Δεδομένα σάρωσης φορτώθηκαν!" if st.session_state.lang=="el" else "✅ Face scan data loaded!")
if   screen=="home":   render_home()
elif screen=="intake": render_intake()
elif screen=="vitals": render_vitals()
elif screen=="triage": render_triage()
elif screen=="report": render_report()
else: render_home()
