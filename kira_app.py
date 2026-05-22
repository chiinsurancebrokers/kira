"""
KIRA — AI Nurse
Bilingual AI health assistant for the Greek market.
Standalone Streamlit app · Real data only · No placeholders.
"""

import streamlit as st
import json
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

/* Hero */
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

/* Cards */
.card {
    background: white;
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 20px;
    box-shadow: 0 2px 12px rgba(45,63,231,0.07);
    border: 1px solid rgba(45,63,231,0.08);
}
.card-purple {
    background: linear-gradient(135deg, #2D3FE7 0%, #7B2FE0 100%);
    color: white;
}
.card h3 { font-size: 16px; font-weight: 600; margin: 0 0 16px; color: #1A1A2E; }
.card-purple h3 { color: white; }

/* Vital badges */
.vital-grid { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 8px; }
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

/* Status pills */
.pill { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }
.pill-green  { background: #DCFCE7; color: #15803D; }
.pill-yellow { background: #FEF9C3; color: #A16207; }
.pill-red    { background: #FEE2E2; color: #B91C1C; }
.pill-blue   { background: #DBEAFE; color: #1D4ED8; }

/* Disclaimer */
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

/* Step indicator */
.step-bar {
    display: flex; gap: 8px; margin-bottom: 28px; justify-content: center;
}
.step {
    width: 32px; height: 6px; border-radius: 3px;
    background: #E0E5FF;
}
.step.active { background: #2D3FE7; }
.step.done   { background: #7B2FE0; }

/* Chat */
.chat-user {
    background: #2D3FE7; color: white;
    padding: 12px 16px; border-radius: 12px 12px 2px 12px;
    margin: 8px 0; max-width: 80%; margin-left: auto;
    font-size: 14px;
}
.chat-kira {
    background: white; border: 1px solid #E0E5FF;
    padding: 12px 16px; border-radius: 2px 12px 12px 12px;
    margin: 8px 0; max-width: 85%;
    font-size: 14px;
}

/* Report */
.report-section { margin: 16px 0; }
.report-header {
    font-size: 11px; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; color: #6B7280;
    border-bottom: 1px solid #E5E7EB;
    padding-bottom: 8px; margin-bottom: 12px;
}
.diff-dx {
    display: flex; align-items: center; gap: 12px;
    padding: 10px 0; border-bottom: 1px solid #F3F4F6;
}
.diff-dx-bar { height: 6px; border-radius: 3px; background: #2D3FE7; }
.diff-dx-pct { font-weight: 700; color: #2D3FE7; min-width: 38px; }

/* Section nav */
.section-nav {
    display: flex; gap: 8px; margin-bottom: 24px; flex-wrap: wrap;
}
.nav-btn {
    padding: 8px 16px; border-radius: 20px; font-size: 13px; font-weight: 500;
    cursor: pointer; border: 1.5px solid #2D3FE7; color: #2D3FE7;
    background: white; transition: all 0.15s;
}
.nav-btn.active { background: #2D3FE7; color: white; }

/* Emergency banner */
.emergency {
    background: linear-gradient(90deg, #DC2626, #B91C1C);
    color: white; border-radius: 10px; padding: 16px 20px;
    font-weight: 600; font-size: 14px; margin: 12px 0;
}

/* Lang toggle */
.lang-toggle { text-align: right; margin-bottom: -8px; }
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
def pubmed_search(query, n=4):
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
        if not pairs: return "✅ RxNorm: No known interactions found."
        lines = []
        for g in pairs:
            src = g.get("sourceName","")
            for t in g.get("fullInteractionType",[]):
                for pair in t.get("interactionPair",[]):
                    sev  = pair.get("severity","")
                    desc = pair.get("description","")
                    drugs = " + ".join(c.get("minConceptItem",{}).get("name","") for c in pair.get("interactionConcept",[]))
                    lines.append(f"- **{drugs}** [{sev}] — {desc} *({src})*")
        return "\n".join(lines) if lines else "✅ RxNorm: No known interactions found."
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
    """Call Claude via raw HTTP — no anthropic package needed."""
    key = get_claude_key()
    if not key:
        return "⚠️ Claude API key not set."
    body = json.dumps({
        "model": "claude-sonnet-4-20250514",
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
            return "⚠️ Claude error: Request timed out — the report was too long. Try again or reduce the conversation length."
        return f"⚠️ Claude error: {e}"
    except Exception as e:
        return f"⚠️ Claude error: {e}"

# ── SESSION STATE ─────────────────────────────────────────────────────────────
defaults = {
    "lang": "el",           # el | en
    "screen": "home",       # home | intake | vitals | triage | report
    "profile": {},          # user profile dict
    "vitals": {},           # vitals dict
    "vitals_analysis": "",  # Claude's vitals interpretation
    "triage_chat": [],      # [{role, content}]
    "triage_ready": False,  # enough info collected for report
    "report": "",           # full clinical report text
    "report_pubmed": [],    # pubmed refs used
    "report_gpt": "",       # GPT-4o second opinion
    "medications": [],      # [{name, freq, notes}]
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── TRANSLATIONS ──────────────────────────────────────────────────────────────
T = {
    "el": {
        "title": "Kira",
        "subtitle": "Ο AI Νοσηλευτής σου",
        "tagline": "Έγκυρη ιατρική πληροφόρηση · Πάντα δίπλα σου",
        "start": "Ξεκίνα Εκτίμηση",
        "disclaimer_main": "⚠️ Η Kira παρέχει πληροφορίες υγείας αποκλειστικά για ενημερωτικούς σκοπούς. Δεν αντικαθιστά ιατρική διάγνωση ή θεραπεία. Σε επείγουσα ανάγκη καλέστε **166** (ΕΚΑΒ) ή **112**.",
        "emergency": "🚨 ΣΕ ΕΠΕΙΓΟΥΣΑ ΑΝΑΓΚΗ: ΚΑΛΕΣΤΕ 166 (ΕΚΑΒ) ή 112",
        "name": "Όνομα", "age": "Ηλικία", "sex": "Φύλο",
        "male": "Άνδρας", "female": "Γυναίκα", "other": "Άλλο",
        "history": "Ιατρικό ιστορικό (προηγούμενες παθήσεις, χειρουργεία)",
        "allergies": "Αλλεργίες",
        "meds": "Τρέχοντα φάρμακα / συμπληρώματα",
        "next": "Επόμενο →",
        "back": "← Πίσω",
        "vitals_title": "Ζωτικές Ενδείξεις",
        "vitals_sub": "Εισάγετε τις μετρήσεις σας. Χρησιμοποιήστε πιστοποιημένη συσκευή για ακριβή αποτελέσματα.",
        "hr": "Καρδιακός Ρυθμός (bpm)",
        "bp_sys": "Αρτηριακή Πίεση — Συστολική (mmHg)",
        "bp_dia": "Αρτηριακή Πίεση — Διαστολική (mmHg)",
        "br": "Αναπνευστικός Ρυθμός (/min)",
        "spo2": "SpO2 (%)",
        "temp": "Θερμοκρασία (°C)",
        "weight": "Βάρος (kg)",
        "height": "Ύψος (cm)",
        "analyse_vitals": "Ανάλυση Ζωτικών",
        "triage_title": "Εκτίμηση Συμπτωμάτων",
        "triage_sub": "Περιγράψτε τα συμπτώματά σας. Η Kira θα σας κάνει κατευθυνόμενες ερωτήσεις.",
        "triage_placeholder": "Π.χ. Έχω πονοκέφαλο τριών ημερών με ναυτία...",
        "generate_report": "Δημιουργία Πλήρους Αναφοράς",
        "report_title": "Λεπτομερής Εκτίμηση Υγείας",
        "second_opinion": "Δεύτερη Γνώμη GPT-4o",
        "pubmed": "Επιστημονικές Αναφορές PubMed",
        "green_label": "✅ Ενθαρρυντικά",
        "yellow_label": "🟡 Παρακολούθηση",
        "red_label": "🔴 Χρήζει Προσοχής",
        "skip_vitals": "Παράλειψη (χωρίς μετρήσεις)",
        "face_scan_soon": "📷 Σάρωση Προσώπου — Σύντομα",
        "face_scan_note": "Η αυτόματη ανάγνωση ζωτικών μέσω κάμερας θα είναι διαθέσιμη στην επόμενη έκδοση.",
    },
    "en": {
        "title": "Kira",
        "subtitle": "Your AI Nurse",
        "tagline": "Evidence-based health guidance · Always by your side",
        "start": "Start Assessment",
        "disclaimer_main": "⚠️ Kira provides health information for informational purposes only. It does not replace medical diagnosis or treatment. In an emergency call **166** (EKAB) or **112**.",
        "emergency": "🚨 EMERGENCY: CALL 166 (EKAB) or 112",
        "name": "Name", "age": "Age", "sex": "Biological Sex",
        "male": "Male", "female": "Female", "other": "Other",
        "history": "Medical history (conditions, surgeries)",
        "allergies": "Allergies",
        "meds": "Current medications / supplements",
        "next": "Next →",
        "back": "← Back",
        "vitals_title": "Your Vitals",
        "vitals_sub": "Enter your measurements. Use a certified device for accurate readings.",
        "hr": "Heart Rate (bpm)",
        "bp_sys": "Blood Pressure — Systolic (mmHg)",
        "bp_dia": "Blood Pressure — Diastolic (mmHg)",
        "br": "Breathing Rate (/min)",
        "spo2": "SpO2 (%)",
        "temp": "Temperature (°C)",
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
        "green_label": "✅ Reassuring",
        "yellow_label": "🟡 Worth Watching",
        "red_label": "🔴 Needs Attention",
        "skip_vitals": "Skip (no measurements)",
        "face_scan_soon": "📷 Face Scan — Coming Soon",
        "face_scan_note": "Automatic vital sign detection via camera will be available in the next version.",
    }
}

def t(key): return T[st.session_state.lang].get(key, key)

# ── VITAL INTERPRETATION ──────────────────────────────────────────────────────
def classify_vitals(v):
    """Returns {metric: 'green'|'yellow'|'red'} for known vitals."""
    status = {}
    hr = v.get("hr")
    if hr:
        if 60 <= hr <= 100: status["hr"] = "green"
        elif 50 <= hr <= 110: status["hr"] = "yellow"
        else: status["hr"] = "red"
    sys = v.get("bp_sys"); dia = v.get("bp_dia")
    if sys and dia:
        if sys < 120 and dia < 80: status["bp"] = "green"
        elif sys < 130 and dia < 80: status["bp"] = "yellow"
        elif sys < 140 or dia < 90: status["bp"] = "yellow"
        else: status["bp"] = "red"
    br = v.get("br")
    if br:
        if 12 <= br <= 20: status["br"] = "green"
        elif 10 <= br <= 24: status["br"] = "yellow"
        else: status["br"] = "red"
    spo2 = v.get("spo2")
    if spo2:
        if spo2 >= 95: status["spo2"] = "green"
        elif spo2 >= 90: status["spo2"] = "yellow"
        else: status["spo2"] = "red"
    temp = v.get("temp")
    if temp:
        if 36.1 <= temp <= 37.2: status["temp"] = "green"
        elif 37.3 <= temp <= 38.0: status["temp"] = "yellow"
        else: status["temp"] = "red"
    w = v.get("weight"); h = v.get("height")
    if w and h:
        bmi = w / ((h/100)**2)
        v["bmi"] = round(bmi, 1)
        if 18.5 <= bmi <= 24.9: status["bmi"] = "green"
        elif 25 <= bmi <= 29.9: status["bmi"] = "yellow"
        else: status["bmi"] = "red"
    return status

KIRA_SYSTEM_EL = """Είσαι η Kira — AI νοσηλευτής για Έλληνες χρήστες. Είσαι κλινικά ακριβής, άμεση και υποστηρικτική.

Ρόλος:
- Τριάζ συμπτωμάτων: Κάνεις στοχευμένες ερωτήσεις (μία κάθε φορά), αξιολογείς και δίνεις δομημένη διαφορική.
- Ζωτικές ενδείξεις: Ερμηνεύεις τιμές, εντοπίζεις μοτίβα, προτείνεις δράση.
- Φάρμακα: Εξηγείς αλληλεπιδράσεις, χρόνο λήψης, παρενέργειες.
- Ελληνικό σύστημα υγείας: Αναφέρεις ΕΟΠΥΥ, ΕΟΔΥ, ΕΟΦ όπου σχετικό.

Κανόνες:
- Πάντα συστήνεις επαγγελματία για διάγνωση/θεραπεία.
- Κόκκινες σημαίες (chest pain, stroke, anaphylaxis) → άμεση παραπομπή σε 166/112.
- Απαντάς στη γλώσσα του χρήστη (Ελληνικά ή Αγγλικά).
- Μία ερώτηση κάθε φορά — ποτέ λίστα ερωτήσεων.
- Όταν έχεις αρκετές πληροφορίες, γράψε: "Έχω αρκετά στοιχεία — μπορούμε να δημιουργήσουμε πλήρη αναφορά."
- Μην αρνείσαι ιατρικές ερωτήσεις — δίνεις την καλύτερη τεκμηριωμένη απάντηση σημειώνοντας τους περιορισμούς."""

KIRA_SYSTEM_EN = """You are Kira — an AI nurse for users in Greece. You are clinically accurate, direct, and supportive.

Role:
- Symptom triage: Ask targeted questions (one at a time), assess, give structured differentials.
- Vitals: Interpret readings, spot patterns, recommend action.
- Medications: Explain interactions, timing, side-effects.
- Greek health system: Reference EOPYY, EODY, EOF where relevant.

Rules:
- Always recommend a healthcare professional for diagnosis/treatment.
- Red flags (chest pain, stroke signs, anaphylaxis) → immediate referral to 166/112.
- Respond in the user's language (Greek or English).
- One question at a time — never a list of questions.
- When you have enough information, write: "I have enough information — we can generate a full clinical report."
- Never refuse a medical question — give the best evidence-based answer while noting limitations."""

def kira_system(): return KIRA_SYSTEM_EL if st.session_state.lang == "el" else KIRA_SYSTEM_EN

# ─────────────────────────────────────────────────────────────────────────────
# SCREENS
# ─────────────────────────────────────────────────────────────────────────────

def render_home():
    # Lang toggle
    c1, c2 = st.columns([6, 1])
    with c2:
        if st.button("🇬🇧 EN" if st.session_state.lang == "el" else "🇬🇷 ΕΛ"):
            st.session_state.lang = "en" if st.session_state.lang == "el" else "el"
            st.rerun()

    st.markdown(f"""
    <div class="kira-hero">
        <div style="font-size:64px;margin-bottom:8px">🩺</div>
        <h1>{t('title')}</h1>
        <p>{t('subtitle')}</p>
        <div class="kira-tagline">{t('tagline')}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="disclaimer">{t("disclaimer_main")}</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button(t("start"), type="primary", use_container_width=True):
            st.session_state.screen = "intake"
            st.rerun()

    st.markdown("---")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""<div class="card">
            <div style="font-size:32px">🔬</div>
            <h3 style="margin-top:12px">PubMed Evidence</h3>
            <p style="font-size:13px;color:#6B7280">Κάθε εκτίμηση υποστηρίζεται από επιστημονική βιβλιογραφία από το NCBI.</p>
        </div>""", unsafe_allow_html=True)
    with f2:
        st.markdown("""<div class="card">
            <div style="font-size:32px">🤖</div>
            <h3 style="margin-top:12px">Dual AI Engine</h3>
            <p style="font-size:13px;color:#6B7280">Claude Sonnet + GPT-4o για διπλή κλινική γνώμη σε κάθε εκτίμηση.</p>
        </div>""", unsafe_allow_html=True)
    with f3:
        st.markdown("""<div class="card">
            <div style="font-size:32px">🇬🇷</div>
            <h3 style="margin-top:12px">Ελληνικό Πλαίσιο</h3>
            <p style="font-size:13px;color:#6B7280">ΕΟΠΥΥ, ΕΟΔΥ, ΕΟΦ — προσαρμοσμένο στο ελληνικό σύστημα υγείας.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown(f'<div class="emergency">{t("emergency")}</div>', unsafe_allow_html=True)


def render_intake():
    st.markdown(f"## 👤 {t('name')} & Ιστορικό")
    st.caption("Βήμα 1 από 3 — Προσωπικά στοιχεία")

    c1, c2, c3 = st.columns([2,1,1])
    with c1:
        name = st.text_input(t("name"), value=st.session_state.profile.get("name",""), placeholder="Χριστόφορος")
    with c2:
        age = st.number_input(t("age"), min_value=1, max_value=120, value=st.session_state.profile.get("age", 40))
    with c3:
        sex = st.selectbox(t("sex"), [t("male"), t("female"), t("other")],
                           index=[t("male"),t("female"),t("other")].index(
                               st.session_state.profile.get("sex", t("male"))))

    history   = st.text_area(t("history"),   value=st.session_state.profile.get("history",""),   height=90, placeholder="Π.χ. Υπέρταση, Τ2 Διαβήτης, Χολοκυστεκτομή 2019")
    allergies = st.text_input(t("allergies"), value=st.session_state.profile.get("allergies",""), placeholder="Π.χ. Πενικιλλίνη")
    meds_raw  = st.text_area(t("meds"),       value=st.session_state.profile.get("meds_raw",""),  height=70, placeholder="Π.χ. Metformin 500mg 2x/day, Creatine 5g/day")

    col_b, col_n = st.columns([1,3])
    with col_b:
        if st.button(t("back")):
            st.session_state.screen = "home"; st.rerun()
    with col_n:
        if st.button(t("next"), type="primary", use_container_width=True):
            if name:
                st.session_state.profile = {
                    "name": name, "age": age, "sex": sex,
                    "history": history, "allergies": allergies, "meds_raw": meds_raw,
                }
                # Parse meds into list
                st.session_state.medications = [
                    {"name": m.strip(), "freq": "", "notes": ""}
                    for m in meds_raw.split(",") if m.strip()
                ] if meds_raw else []
                st.session_state.screen = "vitals"
                st.rerun()
            else:
                st.warning("Παρακαλώ εισάγετε το όνομά σας." if st.session_state.lang=="el" else "Please enter your name.")


def render_vitals():
    p = st.session_state.profile
    st.markdown(f"## 📊 {t('vitals_title')} — {p.get('name','')}")
    st.caption(t("vitals_sub"))

    # ── Check for incoming face scan results via URL params ───────────────
    try:
        params = st.query_params
        facescan_raw = params.get("facescan", "")
        if facescan_raw and not st.session_state.vitals:
            import urllib.parse as _up
            scanned = json.loads(_up.unquote(facescan_raw))
            if scanned:
                st.session_state.vitals = scanned
                st.query_params.clear()
                st.success("✅ Τα δεδομένα από τη σάρωση προσώπου φορτώθηκαν!" if st.session_state.lang=="el"
                           else "✅ Face scan data loaded!")
    except Exception:
        pass

    # ── Face scan card ────────────────────────────────────────────────────
    shenai_key = st.secrets.get("SHENAI_API_KEY", "")
    facescan_url = st.secrets.get("FACESCAN_URL", "https://kiraainurse.netlify.app")
    kira_url_default = "https://kiraainurse.streamlit.app"

    if facescan_url:
        import urllib.parse as _up
        kira_url = st.secrets.get("KIRA_URL", kira_url_default)
        scan_link = f"{facescan_url}?kira_url={_up.quote(kira_url)}"
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#2D3FE7,#7B2FE0);border-radius:16px;padding:24px;text-align:center;margin-bottom:20px;color:white">
            <div style="font-size:36px">📷</div>
            <div style="font-size:18px;font-weight:700;margin:8px 0">Σάρωση Προσώπου — Shen.AI</div>
            <div style="font-size:13px;opacity:0.8;margin-bottom:16px">Μέτρηση καρδιακού ρυθμού, πίεσης, HRV, stress index σε 60 δευτερόλεπτα</div>
            <a href="{scan_link}" target="_blank" style="background:white;color:#2D3FE7;padding:12px 24px;border-radius:8px;font-weight:700;text-decoration:none;font-size:14px">
                Έναρξη Σάρωσης →
            </a>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Μετά τη σάρωση θα επιστρέψετε αυτόματα εδώ με τα αποτελέσματα.")
        st.divider()



    # ── Manual vitals input ───────────────────────────────────────────────
    st.markdown(f"**Χειροκίνητη Εισαγωγή Μετρήσεων**" if st.session_state.lang=="el" else "**Manual Vitals Entry**")

    # Pre-populate from face scan if available
    v = st.session_state.vitals

    c1, c2, c3 = st.columns(3)
    with c1:
        hr    = st.number_input(t("hr"),   min_value=0, max_value=300, value=int(v.get("hr",0)) or None,    placeholder="76")
        spo2  = st.number_input(t("spo2"), min_value=0, max_value=100, value=int(v.get("spo2",0)) or None,  placeholder="98")
        temp  = st.number_input(t("temp"), min_value=0.0, max_value=45.0, value=float(v.get("temp",0.0)) or None, placeholder="36.6", format="%.1f")
    with c2:
        bp_s  = st.number_input(t("bp_sys"), min_value=0, max_value=300, value=int(v.get("bp_sys",0)) or None, placeholder="120")
        bp_d  = st.number_input(t("bp_dia"), min_value=0, max_value=200, value=int(v.get("bp_dia",0)) or None, placeholder="80")
        br    = st.number_input(t("br"),   min_value=0, max_value=60,  value=int(v.get("br",0)) or None,    placeholder="15")
    with c3:
        weight = st.number_input(t("weight"), min_value=0.0, max_value=300.0, value=float(v.get("weight",0.0)) or None, placeholder="75", format="%.1f")
        height = st.number_input(t("height"), min_value=0, max_value=250, value=int(v.get("height",0)) or None, placeholder="175")

    col_b, col_s, col_n = st.columns([1,1,2])
    with col_b:
        if st.button(t("back")):
            st.session_state.screen = "intake"; st.rerun()
    with col_s:
        if st.button(t("skip_vitals")):
            st.session_state.vitals = {}
            st.session_state.screen = "triage"; st.rerun()
    with col_n:
        if st.button(t("analyse_vitals"), type="primary", use_container_width=True):
            vd = {}
            if hr:     vd["hr"] = hr
            if bp_s:   vd["bp_sys"] = bp_s
            if bp_d:   vd["bp_dia"] = bp_d
            if br:     vd["br"] = br
            if spo2:   vd["spo2"] = spo2
            if temp:   vd["temp"] = temp
            if weight: vd["weight"] = weight
            if height: vd["height"] = height
            # Preserve Shen.AI-only fields (HRV, stress, cardio) if present
            for extra in ["hrv","stress","cardio"]:
                if extra in st.session_state.vitals: vd[extra] = st.session_state.vitals[extra]
            st.session_state.vitals = vd
            classify_vitals(vd)

            if vd:
                with st.spinner("Ανάλυση ζωτικών..." if st.session_state.lang=="el" else "Analysing vitals..."):
                    vtext = "\n".join(f"- {k}: {val}" for k, val in vd.items())
                    profile_text = f"{p.get('name')}, {p.get('age')}yo {p.get('sex')}, Hx: {p.get('history','none')}, Meds: {p.get('meds_raw','none')}"
                    prompt = f"""Patient: {profile_text}

Vitals:
{vtext}

Interpret these vitals. Categorise each as normal/borderline/concerning. Note patterns. Flag anything requiring urgent attention. Be direct and specific."""
                    st.session_state.vitals_analysis = claude(
                        [{"role":"user","content": prompt}], system=kira_system(), max_tokens=800
                    )
            st.session_state.screen = "triage"
            st.rerun()


def render_vitals_summary():
    """Compact vitals display used in triage & report screens."""
    v = st.session_state.vitals
    if not v: return
    status = classify_vitals(v)

    LABELS = {
        "hr":("❤️","Heart Rate","bpm"), "bp":("🩸","Blood Pressure","mmHg"),
        "br":("🌬️","Breathing","/min"), "spo2":("💧","SpO2","%"),
        "temp":("🌡️","Temp","°C"), "bmi":("⚖️","BMI","kg/m²"),
    }

    badges = []
    if "hr"   in v: badges.append(("hr",   v["hr"],   "bpm", status.get("hr","green")))
    if "bp_sys" in v and "bp_dia" in v:
        badges.append(("bp", f"{v['bp_sys']}/{v['bp_dia']}", "mmHg", status.get("bp","green")))
    if "br"   in v: badges.append(("br",   v["br"],   "/min", status.get("br","green")))
    if "spo2" in v: badges.append(("spo2", v["spo2"], "%",    status.get("spo2","green")))
    if "temp" in v: badges.append(("temp", v["temp"], "°C",   status.get("temp","green")))
    if "bmi"  in v: badges.append(("bmi",  v["bmi"],  "kg/m²",status.get("bmi","green")))

    cols = st.columns(len(badges)) if badges else []
    for i, (key, val, unit, col) in enumerate(badges):
        icon, label, _ = LABELS.get(key, ("","",""))
        with cols[i]:
            bg = {"green":"#EDFBF0","yellow":"#FFFBEB","red":"#FEF2F2"}.get(col,"#F4F6FF")
            brd = {"green":"#A3E6B5","yellow":"#FCD34D","red":"#FCA5A5"}.get(col,"#E0E5FF")
            st.markdown(f"""
            <div style="background:{bg};border:1px solid {brd};border-radius:12px;padding:12px;text-align:center">
                <div style="font-size:18px">{icon}</div>
                <div style="font-size:20px;font-weight:700">{val}</div>
                <div style="font-size:10px;color:#6B7280">{unit}</div>
                <div style="font-size:11px;color:#374151">{label}</div>
            </div>""", unsafe_allow_html=True)

    if st.session_state.vitals_analysis:
        with st.expander("📋 Ανάλυση ζωτικών" if st.session_state.lang=="el" else "📋 Vitals analysis"):
            st.markdown(st.session_state.vitals_analysis)


def render_triage():
    p = st.session_state.profile
    st.markdown(f"## 🩺 {t('triage_title')} — {p.get('name','')}")
    st.caption(t("triage_sub"))

    render_vitals_summary()

    st.markdown(f'<div class="disclaimer">{t("disclaimer_main")}</div>', unsafe_allow_html=True)

    # Chat display
    for msg in st.session_state.triage_chat:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-kira">🩺 {msg["content"]}</div>', unsafe_allow_html=True)

    # Check if Kira says she has enough info
    ready_phrases = [
        "έχω αρκετά στοιχεία", "μπορούμε να δημιουργήσουμε",
        "i have enough information", "we can generate",
        "full clinical report", "πλήρη αναφορά",
    ]
    last_kira = next((m["content"].lower() for m in reversed(st.session_state.triage_chat) if m["role"]=="assistant"), "")
    triage_ready = any(ph in last_kira for ph in ready_phrases)

    # Input
    user_input = st.chat_input(t("triage_placeholder"), key="triage_input")
    if user_input:
        st.session_state.triage_chat.append({"role":"user","content":user_input})
        with st.spinner("Kira..."):
            # Build context
            profile_ctx = f"Patient: {p.get('name')}, {p.get('age')}yo {p.get('sex')}, Hx: {p.get('history','none')}, Allergies: {p.get('allergies','none')}, Meds: {p.get('meds_raw','none')}"
            vitals_ctx  = "Vitals: " + ", ".join(f"{k}={val}" for k,val in st.session_state.vitals.items()) if st.session_state.vitals else "Vitals: not provided"
            system_ctx  = kira_system() + f"\n\n{profile_ctx}\n{vitals_ctx}"
            reply = claude(
                [{"role":m["role"],"content":m["content"]} for m in st.session_state.triage_chat],
                system=system_ctx, max_tokens=600,
            )
        st.session_state.triage_chat.append({"role":"assistant","content":reply})
        st.rerun()

    # Action buttons
    col_b, col_r = st.columns([1,2])
    with col_b:
        if st.button(t("back")):
            st.session_state.screen = "vitals"; st.rerun()
    with col_r:
        label = t("generate_report")
        enabled = triage_ready or len(st.session_state.triage_chat) >= 6
        if st.button(label, type="primary", use_container_width=True, disabled=not enabled):
            st.session_state.screen = "report"
            st.rerun()

    if not enabled:
        st.caption("Συνεχίστε τη συνομιλία — η Kira θα σας ειδοποιήσει όταν έχει αρκετά στοιχεία." if st.session_state.lang=="el"
                   else "Continue the conversation — Kira will let you know when she has enough information.")


def render_report():
    p = st.session_state.profile
    lang = st.session_state.lang

    st.markdown(f"## 📋 {t('report_title')}")
    st.caption(f"{p.get('name','')}, {p.get('age')}y · {datetime.now().strftime('%d %b %Y %H:%M')}")

    render_vitals_summary()

    # ── Generate report if not already done ──────────────────────────────────
    if not st.session_state.report:
        conversation = "\n".join(
            f"{'Patient' if m['role']=='user' else 'Kira'}: {m['content']}"
            for m in st.session_state.triage_chat
        )
        vitals_text = "\n".join(f"- {k}: {v}" for k,v in st.session_state.vitals.items()) if st.session_state.vitals else "Not provided"
        vitals_analysis = st.session_state.vitals_analysis or "Not available"

        # PubMed search — extract key symptom from conversation
        last_user = next((m["content"] for m in reversed(st.session_state.triage_chat) if m["role"]=="user"), "")
        search_query = last_user[:80] + " diagnosis management" if last_user else "symptom assessment management"
        with st.spinner("🔬 Αναζήτηση PubMed..." if lang=="el" else "🔬 Searching PubMed evidence..."):
            refs = pubmed_search(search_query, n=3)   # n=3 → faster
            st.session_state.report_pubmed = refs

        pubmed_ctx = "\n".join(f"- {a['title']} ({a['journal']}, {a['date']}) {a['url']}" for a in refs) if refs else "None found."

        report_prompt = f"""Generate a concise clinical assessment report for:

PATIENT: {p.get('name')}, {p.get('age')}yo {p.get('sex')}
MEDICAL HISTORY: {p.get('history','none')}
ALLERGIES: {p.get('allergies','none')}
MEDICATIONS: {p.get('meds_raw','none')}

VITALS:
{vitals_text}

VITALS INTERPRETATION: {vitals_analysis}

CLINICAL CONSULTATION:
{conversation}

PUBMED REFERENCES:
{pubmed_ctx}

Write a structured report with these sections (keep each section concise):
1. CHIEF COMPLAINT
2. HISTORY OF PRESENT ILLNESS
3. ASSESSMENT — Primary diagnosis with reasoning + top 2-3 differentials with % probability
4. TREATMENT PLAN — Immediate actions, medications to discuss with doctor, lifestyle, follow-up
5. RED FLAGS — symptoms requiring emergency care
6. PUBMED CITATIONS — cite 1-2 references

Language: {"Greek (Ελληνικά)" if lang=="el" else "English"}
Be direct and clinical. End with AI disclaimer."""

        with st.spinner("🩺 Δημιουργία κλινικής αναφοράς — παρακαλώ περιμένετε (30-60\")..." if lang=="el"
                        else "🩺 Generating clinical report — please wait (30-60s)..."):
            result = claude(
                [{"role":"user","content":report_prompt}],
                system=kira_system(),
                max_tokens=1500,   # down from 2000 → faster, avoids timeout
                timeout=120,       # 2 minutes for the report call
            )
            if result.startswith("⚠️"):
                st.error(result)
                st.info("Tip: Try clicking 'Generate Full Report' again — network timeouts are usually transient." if lang=="en"
                        else "Συμβουλή: Κάντε κλικ ξανά στο 'Δημιουργία Αναφοράς' — τα timeouts είναι συνήθως προσωρινά.")
            else:
                st.session_state.report = result

    # ── Display report ────────────────────────────────────────────────────────
    if not st.session_state.report:
        st.warning("Η αναφορά δεν δημιουργήθηκε ακόμα. Πατήστε 'Δημιουργία Αναφοράς' ξανά." if lang=="el"
                   else "Report not generated yet. Click 'Generate Full Report' again.")
        if st.button("🔄 " + ("Δοκιμή ξανά" if lang=="el" else "Retry"), type="primary"):
            st.rerun()
        return
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(st.session_state.report)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── PubMed references ─────────────────────────────────────────────────────
    if st.session_state.report_pubmed:
        with st.expander(f"🔬 {t('pubmed')} ({len(st.session_state.report_pubmed)})"):
            for a in st.session_state.report_pubmed:
                st.markdown(f"**[{a['title']}]({a['url']})**  \n*{a['authors']} — {a['journal']}, {a['date']}*")

    # ── GPT-4o second opinion ─────────────────────────────────────────────────
    if get_openai_key():
        with st.expander(f"🤖 {t('second_opinion')}"):
            if not st.session_state.report_gpt:
                if st.button("Get GPT-4o Second Opinion", type="secondary"):
                    with st.spinner("GPT-4o reviewing..."):
                        st.session_state.report_gpt = gpt4o(
                            prompt=f"Patient: {p.get('name')}, {p.get('age')}yo {p.get('sex')}\n\nClaude's clinical report:\n{st.session_state.report}\n\nDo you agree with this assessment? Provide additions, corrections, or alternative considerations. Be specific.",
                            system=kira_system(), max_tokens=900,
                        )
                    st.rerun()
            else:
                st.markdown(st.session_state.report_gpt)

    # ── RxNorm medication check ───────────────────────────────────────────────
    if len(st.session_state.medications) >= 2:
        with st.expander("💊 RxNorm — Έλεγχος Αλληλεπιδράσεων" if lang=="el" else "💊 RxNorm — Interaction Check"):
            with st.spinner("Querying RxNorm..."):
                rxr = rxnorm_interactions([m["name"] for m in st.session_state.medications])
            if rxr: st.markdown(rxr)

    # ── Emergency banner always ───────────────────────────────────────────────
    st.markdown(f'<div class="emergency">{t("emergency")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="disclaimer-red">AI-generated report. Η Kira δεν παρέχει ιατρική διάγνωση. Πάντα να συμβουλεύεστε επαγγελματία υγείας.</div>', unsafe_allow_html=True)

    # ── Actions ───────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Νέα Αξιολόγηση" if lang=="el" else "← New Assessment"):
            for k, v in defaults.items():
                st.session_state[k] = v
            st.rerun()
    with col2:
        st.download_button(
            "📥 Λήψη Αναφοράς (.txt)" if lang=="el" else "📥 Download Report (.txt)",
            data=st.session_state.report,
            file_name=f"kira_report_{p.get('name','patient')}_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True,
        )


# ── ROUTER ────────────────────────────────────────────────────────────────────
screen = st.session_state.screen

if   screen == "home":   render_home()
elif screen == "intake": render_intake()
elif screen == "vitals": render_vitals()
elif screen == "triage": render_triage()
elif screen == "report": render_report()
else: render_home()
