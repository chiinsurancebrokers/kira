"""
ASKLEPIOS — AI Nurse
Bilingual AI health assistant for the Greek market.
Standalone Streamlit app · Real data only · No placeholders.
"""

import streamlit as st
import os
import json
import io
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
import io as _io, base64 as _b64
import hmac, hashlib, time, unicodedata

# "Stay signed in" via a browser cookie (persists login across reloads / new tabs,
# e.g. when returning from the external face scan). Degrades gracefully if missing.
try:
    import extra_streamlit_components as stx
    _STX_OK = True
except Exception:
    _STX_OK = False

# HEIC support for iPhone photos
try:
    import pillow_heif as _heif
    from PIL import Image as _Image
    _heif.register_heif_opener()
    HEIC_OK = True
except ImportError:
    HEIC_OK = False

# ── SAFE SECRETS / ENV ACCESS ─────────────────────────────────────────────────
def _secret(name, default=""):
    """Read a config value from st.secrets, falling back to os.environ, then default.
    Safe on platforms (e.g. Railway) where no secrets.toml exists — accessing
    st.secrets there raises StreamlitSecretNotFoundError even with a default."""
    try:
        v = st.secrets.get(name, None)
        if v not in (None, ""):
            return v
    except Exception:
        pass
    v = os.environ.get(name, "")
    return v if v != "" else default

# ── PHOTO SCANNER FUNCTIONS ───────────────────────────────────────────────────
HUMAN_SCAN_PROMPTS = {
    "eye":    "Examine the eye carefully. Describe: sclera colour (white/red/yellow), pupil symmetry, conjunctiva, any discharge (colour, quantity), eyelid swelling, corneal clarity, third eyelid. Flag any urgent findings.",
    "skin":   "Examine the skin lesion or rash. Describe: colour, size (estimate), borders (regular/irregular), texture (flat/raised/scaly), distribution pattern, any ulceration, satellite lesions. Note ABCDE criteria if applicable (Asymmetry, Border, Colour, Diameter, Evolution).",
    "wound":  "Examine the wound. Describe: type (laceration/abrasion/puncture/burn), dimensions (estimate), depth, wound edges, signs of infection (redness/swelling/warmth/pus/odour), presence of foreign bodies, tissue viability.",
    "throat": "Examine the mouth and throat. Describe: tonsil size and appearance, pharyngeal wall, any exudate or white patches, uvula position, tongue coating, gum condition, mucosal lesions, petechiae on palate.",
    "nails":  "Examine the nails. Describe: colour (pale/yellow/blue/brown/white), shape (clubbing/koilonychia/normal), surface (ridges/pitting/onycholysis), subungual changes, surrounding skin.",
    "body":   "Describe the visible body area. Note: skin colour, visible swelling, asymmetry, rashes, bruising, oedema, muscle wasting, posture, any visible masses or lesions.",
}

def convert_heic_human(img_bytes):
    if not HEIC_OK: raise RuntimeError("pillow-heif not installed")
    img = _Image.open(_io.BytesIO(img_bytes))
    buf = _io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=92)
    return buf.getvalue(), "image/jpeg"

def florence2_human(image_b64, scan_type, api_key):
    workspace = _secret("ROBOFLOW_WORKSPACE","chriss-workspace-zk0ng")
    workflow  = _secret("ROBOFLOW_WORKFLOW","florence2-base-demo")
    url = f"https://serverless.roboflow.com/{workspace}/workflows/{workflow}"
    task_prompt = HUMAN_SCAN_PROMPTS.get(scan_type, HUMAN_SCAN_PROMPTS["skin"])
    body = json.dumps({
        "api_key": api_key,
        "inputs": {"image":{"type":"base64","value":image_b64},"task_prompt":task_prompt}
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            result = json.loads(r.read())
        outputs = result.get("outputs",[])
        if outputs:
            for key in ["output","caption","text","result","description"]:
                if key in outputs[0] and outputs[0][key]:
                    return {"ok":True,"description":str(outputs[0][key])}
        return {"ok":True,"description":str(result)}
    except Exception as e:
        return {"ok":False,"error":str(e)}

def claude_vision_human(image_b64, image_type, prompt, system=""):
    key = get_claude_key()
    if not key: return "⚠️ API key not set."
    body = json.dumps({
        "model":"claude-sonnet-4-6","max_tokens":3000,"system":system,
        "messages":[{"role":"user","content":[
            {"type":"image","source":{"type":"base64","media_type":image_type,"data":image_b64}},
            {"type":"text","text":prompt}
        ]}]
    }).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages",data=body,
        headers={"x-api-key":key,"anthropic-version":"2023-06-01","content-type":"application/json"})
    try:
        with urllib.request.urlopen(req,timeout=60) as r:
            return json.loads(r.read())["content"][0]["text"]
    except Exception as e: return f"⚠️ {e}"

def transcribe_audio(audio_bytes, lang="el", mime="audio/webm", filename="recording.webm"):
    """Transcribe a short voice recording → text. Groq Whisper large-v3 primary
    (fast, free tier, Greek-capable), OpenAI Whisper-1 fallback.
    
    Input expected from st.audio_input → WebM/Opus, small (~1MB per minute).
    Both APIs use multipart/form-data. We build it manually with urllib to
    avoid adding requests as a dep.
    
    Privacy: audio goes to the chosen STT API for processing, NEVER stored
    on our side. Only the resulting transcript text enters session state."""
    import uuid as _uuid
    boundary = f"----asklepios{_uuid.uuid4().hex}"
    
    def _multipart(parts):
        """parts: list of (name, value, filename_or_None, content_type_or_None)"""
        body = bytearray()
        for name, value, fn, ct in parts:
            body += f"--{boundary}\r\n".encode()
            if fn:
                body += f'Content-Disposition: form-data; name="{name}"; filename="{fn}"\r\n'.encode()
                body += f"Content-Type: {ct or 'application/octet-stream'}\r\n\r\n".encode()
                body += value
                body += b"\r\n"
            else:
                body += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
                body += str(value).encode()
                body += b"\r\n"
        body += f"--{boundary}--\r\n".encode()
        return bytes(body)
    
    # Try Groq Whisper large-v3 first
    groq_key = get_groq_key()
    if groq_key:
        try:
            body = _multipart([
                ("file",  audio_bytes, filename, mime),
                ("model", "whisper-large-v3", None, None),
                ("language", lang, None, None),
                ("response_format", "text", None, None),
            ])
            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                data=body,
                headers={
                    "Authorization": f"Bearer {groq_key}",
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                },
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                txt = r.read().decode("utf-8", errors="replace").strip()
            if txt:
                return txt.strip('"'), "groq"
        except Exception:
            pass
    
    # Fallback: OpenAI Whisper-1
    openai_key = get_openai_key()
    if openai_key:
        try:
            body = _multipart([
                ("file",  audio_bytes, filename, mime),
                ("model", "whisper-1", None, None),
                ("language", lang, None, None),
                ("response_format", "text", None, None),
            ])
            req = urllib.request.Request(
                "https://api.openai.com/v1/audio/transcriptions",
                data=body,
                headers={
                    "Authorization": f"Bearer {openai_key}",
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                },
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                txt = r.read().decode("utf-8", errors="replace").strip()
            if txt:
                return txt.strip('"'), "openai"
        except Exception as e:
            return f"⚠️ Σφάλμα μεταγραφής: {e}", None
    
    return "⚠️ Καμία STT API key δεν είναι ρυθμισμένη (Groq ή OpenAI).", None


def claude_analyze_lab(file_bytes, mime_type, profile, conversation, lang, file_name=""):
    """Analyze lab results (PDF or image) via Claude with native document support.
    
    Lab results in Greece are typically PDFs from labs (e.g. Biocheck, Affidea) or
    phone photos of paper printouts. Claude's PDF support handles both text-based
    and image-based PDFs internally (built-in OCR). Result is interpreted WITHIN
    the conversation context — not as a standalone report — so findings tie back
    to the user's reported symptoms.
    
    Privacy: file is sent to Claude API for processing, NEVER stored on our side.
    """
    key = get_claude_key()
    if not key:
        return "⚠️ Claude API key not set."
    
    file_b64 = _b64.b64encode(file_bytes).decode()
    
    # Clinical context from the ongoing assessment
    convo_txt = "\n".join(
        f"{'Ασθενής' if m['role']=='user' else 'Asklepios'}: {m['content'][:400]}"
        for m in (conversation or [])[-6:]
    ) if conversation else ("Δεν έχει καταγραφεί συνομιλία ακόμη." if lang=="el" else "No conversation yet.")
    
    age = profile.get("age", "?")
    sex = profile.get("sex", "")
    history = profile.get("history", "") or "—"
    meds = profile.get("meds_raw", "") or "—"
    # Special-population flags affect reference ranges + drug warnings
    flags = []
    if profile.get("pregnancy"):
        flags.append("ΕΓΚΥΟΣ" if lang=="el" else "PREGNANT")
    if profile.get("for_whom") == "other":
        flags.append("Caregiver-mode" if lang=="el" else "Caregiver-mode")
    try:
        _aint = int(age)
        if _aint < 18:
            flags.append(f"ΠΑΙΔΙΑΤΡΙΚΟΣ {_aint}" if lang=="el" else f"PEDIATRIC {_aint}")
    except (TypeError, ValueError):
        pass
    flags_line = (" | ".join(flags)) if flags else ("—" if lang=="el" else "—")
    
    if lang == "el":
        system = ("Είσαι έμπειρος ιατρός νοσηλευτής που ερμηνεύει εργαστηριακές εξετάσεις "
                  "στα Ελληνικά. Είσαι ακριβής, σαφής, και κάνεις το κλινικό συμπέρασμα ΜΕΣΑ "
                  "στο πλαίσιο των συμπτωμάτων και του ιστορικού. ΔΕΝ κάνεις τελική διάγνωση — "
                  "επισημαίνεις ευρήματα και τι μπορεί να σημαίνουν.")
        prompt = f"""ΚΛΙΝΙΚΟ ΠΛΑΙΣΙΟ:
Ασθενής: {age} ετών, {sex}
Ιστορικό: {history}
Φάρμακα: {meds}

Συνομιλία μέχρι τώρα:
{convo_txt}

---

ΕΡΓΑΣΤΗΡΙΑΚΕΣ ΕΞΕΤΑΣΕΙΣ (επισυνάπτεται PDF/εικόνα):
Ανάλυσε τα αποτελέσματα σε αυτές τις ενότητες:

**1. ΕΥΡΗΜΑΤΑ ΕΚΤΟΣ ΟΡΙΩΝ**
Πίνακας ή λίστα με τους δείκτες που είναι ψηλά ή χαμηλά, με την τιμή, τα όρια αναφοράς, την κατεύθυνση (↑/↓). Αν όλα είναι εντός ορίων, πες το ξεκάθαρα.

**2. ΕΡΜΗΝΕΙΑ**
Τι μπορεί να σημαίνει αυτή η εικόνα κλινικά. Σύντομα, σε απλή γλώσσα.

**3. ΣΧΕΣΗ ΜΕ ΣΥΜΠΤΩΜΑΤΑ**
Συμβατά με όσα περιγράφει ο ασθενής στη συνομιλία; Υποστηρίζουν την τρέχουσα εκτίμηση ή την αλλάζουν;

**4. ΕΠΟΜΕΝΑ ΒΗΜΑΤΑ**
Τι θα ρωτούσε ο γιατρός. Επιπλέον εξετάσεις που ίσως χρειάζονται. Πότε είναι επείγον.

ΣΗΜΑΝΤΙΚΟ: ΜΗΝ κάνεις τελική διάγνωση. Πάντα συστήνεις επίσκεψη σε ιατρό για ερμηνεία.
Αναφέρε ΜΟΝΟ τα ευρήματα που πραγματικά βλέπεις στο έγγραφο — μην εφεύρεις δείκτες."""
    else:
        system = ("You are an expert clinical reviewer interpreting lab results. Be precise, "
                  "clear, and tie findings to the patient's reported symptoms. Do NOT make a "
                  "final diagnosis — surface findings and what they may indicate.")
        prompt = f"""CLINICAL CONTEXT:
Patient: {age} yo {sex}
History: {history}
Medications: {meds}

Conversation so far:
{convo_txt}

---

LAB RESULTS (PDF/image attached):
Analyse in these sections:

**1. OUT-OF-RANGE FINDINGS**
Table or list of indicators that are high or low, with value, reference range, direction (↑/↓). If all within range, say so clearly.

**2. INTERPRETATION**
What this clinical picture may indicate. Brief, plain language.

**3. RELATION TO SYMPTOMS**
Consistent with what the patient describes? Supports or changes the current assessment?

**4. NEXT STEPS**
What the doctor would ask. Additional tests possibly needed. When this is urgent.

IMPORTANT: Do NOT make a final diagnosis. Always recommend seeing a doctor for interpretation.
Only findings you actually see in the document — don't invent indicators."""
    
    # Build content block based on file type
    if mime_type == "application/pdf":
        content_block = {
            "type": "document",
            "source": {"type":"base64", "media_type":"application/pdf", "data":file_b64}
        }
    else:
        content_block = {
            "type": "image",
            "source": {"type":"base64", "media_type":mime_type, "data":file_b64}
        }
    
    body = json.dumps({
        "model": "claude-sonnet-4-6",
        "max_tokens": 3000,
        "system": system,
        "messages": [{
            "role": "user",
            "content": [content_block, {"type":"text","text":prompt}]
        }]
    }).encode()
    
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return json.loads(r.read())["content"][0]["text"]
    except Exception as e:
        return f"⚠️ Σφάλμα ανάλυσης: {e}"

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Asklepios · AI Nurse",
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
/* Markdown tables — clean column alignment. Auto layout (no fixed widths) so each
 * table sizes naturally: differential-diagnosis (3 cols) and treatment plans (2 cols)
 * both render correctly. Previously had table-layout:fixed + nth-child(2):64px which
 * was meant for the diagnosis %-column but accidentally squashed every 2-col table
 * into one-letter-per-line on mobile. */
[data-testid="stMarkdownContainer"] table {
    width: 100%; border-collapse: collapse;
    font-size: 12.5px; margin: 12px 0;
}
[data-testid="stMarkdownContainer"] thead th { background: #F4F6FF; font-weight: 600; }
[data-testid="stMarkdownContainer"] th,
[data-testid="stMarkdownContainer"] td {
    border: 1px solid #E0E5FF; padding: 7px 9px;
    text-align: left; vertical-align: top;
    word-break: normal !important; overflow-wrap: break-word !important; hyphens: none;
}

/* ── SOFT-MODERN PASS ──────────────────────────────────────────────────────
 * Rounds and softens the native Streamlit widgets (buttons, text/number
 * inputs, selectboxes, expanders) so they match the calmer, more rounded
 * card language used elsewhere (doc-header, vital cards, bottom nav).
 * Deliberately scoped to native widgets only — none of the existing custom
 * .class components (kira-stepper, pill, vital-badge, etc.) are touched. */
.stButton button, .stDownloadButton button, .stFormSubmitButton button {
    border-radius: 14px !important;
    font-weight: 700 !important;
    transition: transform .12s ease, box-shadow .12s ease;
}
.stButton button:active { transform: scale(0.98); }
.stButton button[kind="primary"] {
    box-shadow: 0 4px 14px rgba(45,63,231,0.22) !important;
    border: none !important;
}
.stButton button[kind="secondary"] {
    border: 1.5px solid #E0E5FF !important;
    box-shadow: 0 1px 4px rgba(15,23,42,0.04) !important;
}
div[data-baseweb="input"], div[data-baseweb="textarea"], div[data-baseweb="select"] {
    border-radius: 14px !important;
}
.stTextInput input, .stNumberInput input, .stTextArea textarea {
    border-radius: 14px !important;
    background: #F7F8FC !important;
    border: 1.5px solid transparent !important;
}
.stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
    border-color: #2D3FE7 !important;
    box-shadow: 0 0 0 3px rgba(45,63,231,0.12) !important;
}
[data-testid="stExpander"] {
    border-radius: 16px !important;
    border: 1px solid #E5E7EB !important;
    box-shadow: 0 1px 4px rgba(15,23,42,0.03) !important;
    overflow: hidden;
}
[data-testid="stExpander"] summary {
    font-weight: 700 !important;
}
/* Radio / checkbox accent — keep the same blue instead of Streamlit's default */
.stRadio [data-baseweb="radio"] div:first-child, .stCheckbox [data-baseweb="checkbox"] div:first-child {
    border-color: #2D3FE7 !important;
}
</style>
""", unsafe_allow_html=True)

# ── KEYS ──────────────────────────────────────────────────────────────────────
def _key(name, fallback=""):
    for k in [name, name.lower(), name.upper()]:
        v = _secret(k, "")
        if v:
            return v
    return fallback

def get_claude_key():  return _key("Claude_API_Key")
def get_openai_key():  return _key("OPENAI_API_KEY")
def get_groq_key():    return _key("GROQ_API_KEY")
def get_ncbi_key():    return _key("NCBI_API_KEY")
def get_a2e_key():     return _key("A2E_API_KEY")
def get_admin_password(): return _key("ADMIN_PASSWORD")

def render_physio_card(condition_hint, lang="el", refs_override=None):
    """Render a Physiotherapy & Rehabilitation evidence card.
    By default sources refs from st.session_state.report_physio_refs (set once
    the final report's PEDro-equivalent PubMed/MEDLINE search completes — see
    pedro_pillar_search). Pass refs_override to render with a different list,
    e.g. the live triage-time cache used when this card is surfaced proactively
    mid-conversation (before any report exists). No API key required either way."""
    title = "🏃 Φυσιοθεραπεία & Αποκατάσταση" if lang == "el" else "🏃 Physiotherapy & Rehabilitation"
    sub   = ("Σχετική βιβλιογραφία φυσικοθεραπείας (RCT/συστηματικές ανασκοπήσεις) — "
             "συζήτησε με φυσιοθεραπευτή πριν ξεκινήσεις οποιαδήποτε άσκηση."
             if lang == "el" else
             "Relevant physiotherapy evidence (RCTs / systematic reviews) — "
             "discuss with a physiotherapist before starting any exercise.")
    refs = refs_override if refs_override is not None else (st.session_state.get("report_physio_refs") or [])

    import html as _html_p
    if refs:
        items_html = "".join(
            f'<div class="physio-ref">'
            f'<a href="{_html_p.escape(r.get("url","") or "")}" target="_blank" '
            f'class="physio-ref-title">{_html_p.escape((r.get("title","—") or "")[:160])}</a>'
            f'<div class="physio-ref-meta">{_html_p.escape(r.get("journal","") or "")}'
            f'{(" · " + _html_p.escape(r.get("date","")[:4])) if r.get("date") else ""}</div>'
            f'</div>'
            for r in refs
        )
    else:
        no_res = ("Δεν βρέθηκε σχετική βιβλιογραφία φυσικοθεραπείας για αυτήν την πάθηση — "
                  "ζήτησε αξιολόγηση από φυσιοθεραπευτή." if lang == "el" else
                  "No relevant physiotherapy literature found for this condition — "
                  "ask a physiotherapist for a direct assessment.")
        items_html = f'<div class="physio-empty">{no_res}</div>'

    credit = ("Αναζήτηση μέσω PubMed/MEDLINE — εύρος αντίστοιχο PEDro (φυσιοθεραπευτικά RCT/MeSH)."
              if lang == "el" else
              "Searched via PubMed/MEDLINE — PEDro-equivalent scope (physiotherapy RCT/MeSH).")

    st.markdown(f"""
<style>
.physio-card {{
  background: white; border: 1px solid #A7F3D0; border-radius: 22px;
  padding: 20px 22px; margin: 16px 0;
  font-family: 'Inter', system-ui, sans-serif;
  box-shadow: 0 2px 8px rgba(5,150,105,0.07);
}}
.physio-title {{
  font-size: 15px; font-weight: 800; color: #065F46; margin-bottom: 4px;
}}
.physio-sub {{
  font-size: 12px; color: #6B7280; margin-bottom: 14px; line-height: 1.5;
}}
.physio-ref {{
  border: 1px solid #ECFDF5; border-radius: 10px; padding: 11px 13px;
  margin-bottom: 8px; background: #F0FDF4;
}}
.physio-ref-title {{
  font-size: 13.5px; font-weight: 700; color: #065F46; text-decoration: none;
  display: block; margin-bottom: 4px; line-height: 1.4;
}}
.physio-ref-title:hover {{ text-decoration: underline; }}
.physio-ref-meta {{ font-size: 11.5px; color: #6B7280; }}
.physio-empty {{ font-size: 13px; color: #9CA3AF; padding: 10px 0; }}
.physio-credit {{
  font-size: 11px; color: #9CA3AF; margin-top: 14px; padding-top: 10px;
  border-top: 1px dashed #E5E7EB; text-align: right;
}}
</style>
<div class="physio-card">
  <div class="physio-title">{title}</div>
  <div class="physio-sub">{sub}</div>
  {items_html}
  <div class="physio-credit">{credit}</div>
</div>
""", unsafe_allow_html=True)


# MENTAL HEALTH RESOURCES (psychology support section)
# Hardcoded trusted Greek resources — no third-party API available for Greek
# psychology directories. Links are official or well-established organisations.
PSYCHOLOGY_RESOURCES_EL = [
    ("🧠", "Ψυχολόγος ΕΣΥ / Ψυχιατρικές Κλινικές",
     "Δωρεάν πρόσβαση μέσω παραπομπής γιατρού ΕΣΥ.",
     "https://www.moh.gov.gr"),
    ("📞", "Γραμμή Ψυχολογικής Υποστήριξης (10306)",
     "Δωρεάν 24ωρη γραμμή ψυχολογικής υποστήριξης — ΕΚΕΠΥ.",
     "tel:10306"),
    ("💬", "Γραμμή Παρέμβασης Αυτοκτονίας (1018)",
     "Κέντρο Πρόληψης ΚΕΘΕΑ — 24ωρη γραμμή κρίσης.",
     "tel:1018"),
    ("🏥", "Ψυχιατρικό Νοσοκομείο Αττικής",
     "Επείγοντα ψυχιατρικά περιστατικά.",
     "https://www.psyhat.gr"),
    ("🌐", "ΕΤΗΕΑ — Ελληνική Εταιρεία Κλινικής Ψυχολογίας",
     "Μητρώο αδειοδοτημένων ψυχολόγων.",
     "https://www.etheaclinicalpsy.gr"),
    ("🌿", "MindHub Greece",
     "Online ψυχολογική υποστήριξη από αδειοδοτημένους ψυχολόγους.",
     "https://www.mindhub.gr"),
]
PSYCHOLOGY_RESOURCES_EN = [
    ("🧠", "NHS-equivalent (ESY) Psychiatry",
     "Free access via GP referral through the national health system.",
     "https://www.moh.gov.gr"),
    ("📞", "Psychological Support Line (10306)",
     "Free 24h psychological support line — EΚΕΠΥ.",
     "tel:10306"),
    ("💬", "Suicide Prevention Line (1018)",
     "KETHEA crisis centre — 24h line.",
     "tel:1018"),
    ("🏥", "Attica Psychiatric Hospital",
     "Psychiatric emergencies in the Attica region.",
     "https://www.psyhat.gr"),
    ("🌐", "ETHEΑ — Greek Clinical Psychology Society",
     "Registry of licensed clinical psychologists.",
     "https://www.etheaclinicalpsy.gr"),
    ("🌿", "MindHub Greece",
     "Online psychological support from licensed psychologists.",
     "https://www.mindhub.gr"),
]

def render_psychology_card(lang="el", refs_override=None):
    """Render a Mental Health & Psychology support card.
    Shows curated Greek resources: helplines, directories, organisations, plus
    a peer-reviewed PubMed/MEDLINE research section. By default the research
    section sources from st.session_state.report_psych_refs (set once the final
    report's search completes). Pass refs_override to use a different list —
    e.g. the live triage-time cache used when this card is surfaced proactively
    mid-conversation, before any report exists."""
    title = "🧠 Ψυχολογική Υποστήριξη & Ψυχική Υγεία" if lang == "el" else "🧠 Psychological Support & Mental Health"
    sub   = ("Επίσημες υπηρεσίες, γραμμές κρίσης και αδειοδοτημένοι ψυχολόγοι — η αξιολόγηση γίνεται πάντα από επαγγελματία."
             if lang == "el" else
             "Official services, crisis lines and licensed psychologists — assessment is always done by a professional.")
    resources = PSYCHOLOGY_RESOURCES_EL if lang == "el" else PSYCHOLOGY_RESOURCES_EN

    cards_html = ""
    for icon, label, desc, url in resources:
        is_tel = url.startswith("tel:")
        num    = url.replace("tel:", "")
        if is_tel:
            link_html = (f'<a href="{url}" style="display:inline-block;background:#6366F1;color:white;'
                         f'padding:5px 14px;border-radius:8px;font-size:13px;font-weight:700;'
                         f'text-decoration:none;margin-top:6px">📞 {num}</a>')
        else:
            link_html = (f'<a href="{url}" target="_blank" style="display:inline-block;background:#EEF2FF;'
                         f'color:#4338CA;padding:5px 14px;border-radius:8px;font-size:12px;font-weight:700;'
                         f'text-decoration:none;margin-top:6px">↗ Άνοιγμα</a>' if lang == "el" else
                         f'<a href="{url}" target="_blank" style="display:inline-block;background:#EEF2FF;'
                         f'color:#4338CA;padding:5px 14px;border-radius:8px;font-size:12px;font-weight:700;'
                         f'text-decoration:none;margin-top:6px">↗ Open</a>')
        cards_html += (
            f'<div class="psych-item">'
            f'<div class="psych-icon">{icon}</div>'
            f'<div class="psych-body">'
            f'<div class="psych-label">{label}</div>'
            f'<div class="psych-desc">{desc}</div>'
            f'{link_html}'
            f'</div></div>'
        )

    # ── Related research (PubMed/MEDLINE, peer-reviewed) — additive, below the
    # crisis/directory resources above. Those resources are unaffected; this
    # only adds literature.
    import html as _html_q
    _psych_refs = refs_override if refs_override is not None else (st.session_state.get("report_psych_refs") or [])
    research_lbl = "📚 Σχετική Έρευνα (PubMed)" if lang == "el" else "📚 Related Research (PubMed)"
    if _psych_refs:
        research_items = "".join(
            f'<div class="psych-ref">'
            f'<a href="{_html_q.escape(r.get("url","") or "")}" target="_blank" '
            f'class="psych-ref-title">{_html_q.escape((r.get("title","—") or "")[:160])}</a>'
            f'<div class="psych-ref-meta">{_html_q.escape(r.get("journal","") or "")}'
            f'{(" · " + _html_q.escape(r.get("date","")[:4])) if r.get("date") else ""}</div>'
            f'</div>'
            for r in _psych_refs
        )
        research_html = (
            f'<div class="psych-research"><div class="psych-research-lbl">{research_lbl}</div>'
            f'{research_items}</div>'
        )
    else:
        research_html = ""

    st.markdown(f"""
<style>
.psych-card {{
  background: white; border: 1px solid #C7D2FE; border-radius: 22px;
  padding: 20px 22px; margin: 16px 0;
  font-family: 'Inter', system-ui, sans-serif;
  box-shadow: 0 2px 8px rgba(99,102,241,0.07);
}}
.psych-title {{ font-size: 15px; font-weight: 800; color: #3730A3; margin-bottom: 4px; }}
.psych-sub {{ font-size: 12px; color: #6B7280; margin-bottom: 14px; line-height: 1.5; }}
.psych-item {{
  display: flex; gap: 12px; align-items: flex-start;
  border-bottom: 1px solid #F3F4F6; padding: 12px 0;
}}
.psych-item:last-child {{ border-bottom: none; padding-bottom: 0; }}
.psych-icon {{ font-size: 22px; flex-shrink: 0; padding-top: 2px; }}
.psych-body {{ flex: 1; min-width: 0; }}
.psych-label {{ font-size: 13.5px; font-weight: 700; color: #1F2937; margin-bottom: 3px; }}
.psych-desc {{ font-size: 12.5px; color: #4B5563; line-height: 1.5; }}
.psych-research {{ margin-top: 16px; padding-top: 14px; border-top: 1px dashed #E5E7EB; }}
.psych-research-lbl {{
  font-size: 11px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
  color: #6B7280; margin-bottom: 10px;
}}
.psych-ref {{
  border: 1px solid #EEF2FF; border-radius: 10px; padding: 11px 13px;
  margin-bottom: 8px; background: #F5F7FF;
}}
.psych-ref-title {{
  font-size: 13.5px; font-weight: 700; color: #3730A3; text-decoration: none;
  display: block; margin-bottom: 4px; line-height: 1.4;
}}
.psych-ref-title:hover {{ text-decoration: underline; }}
.psych-ref-meta {{ font-size: 11.5px; color: #6B7280; }}
</style>
<div class="psych-card">
  <div class="psych-title">{title}</div>
  <div class="psych-sub">{sub}</div>
  {cards_html}
  {research_html}
</div>
""", unsafe_allow_html=True)
# Graceful degradation: if SUPABASE_URL / SUPABASE_ANON_KEY are not set (or the
# supabase package is missing), auth stays OFF and the whole app is open — so the
# demo keeps working. Set the secrets to switch the gate on automatically.
def _supabase_client():
    url = _secret("SUPABASE_URL", "")
    key = _secret("SUPABASE_ANON_KEY", "")
    if not url or not key:
        return None
    try:
        from supabase import create_client
        return create_client(url, key)
    except Exception:
        return None

def auth_enabled():
    return _supabase_client() is not None

def is_logged_in():
    return bool(st.session_state.get("auth_user"))

# ── PERSISTENT LOGIN (HMAC-signed cookie — cannot be forged) ──────────────────
CM = None  # CookieManager instance, created once per run in the router
COOKIE_NAME = "ak_session"

def _cookie_secret():
    return (_secret("AUTH_COOKIE_SECRET","") or _secret("SUPABASE_ANON_KEY","")
            or "asklepios-dev-cookie-secret")

def _make_token(email, days=14):
    exp = int(time.time()) + days*86400
    body = f"{email}|{exp}"
    sig = hmac.new(_cookie_secret().encode(), body.encode(), hashlib.sha256).hexdigest()[:32]
    return _b64.urlsafe_b64encode(f"{body}|{sig}".encode()).decode()

def _read_token(tok):
    try:
        raw = _b64.urlsafe_b64decode(str(tok).encode()).decode()
        email, exp, sig = raw.rsplit("|", 2)
        if int(exp) < time.time():
            return None
        good = hmac.new(_cookie_secret().encode(), f"{email}|{exp}".encode(), hashlib.sha256).hexdigest()[:32]
        if hmac.compare_digest(sig, good):
            return email
    except Exception:
        return None
    return None

def _save_login_cookie(email):
    cm = globals().get("CM")
    if not cm:
        return
    try:
        cm.set(COOKIE_NAME, _make_token(email), key="ak_set_auth",
               expires_at=datetime.now()+timedelta(days=14))
    except Exception:
        pass

def _clear_login_cookie():
    cm = globals().get("CM")
    if not cm:
        return
    try:
        cm.delete(COOKIE_NAME, key="ak_del_auth")
    except Exception:
        pass

# ── IN-PROGRESS PROFILE DRAFT (server-side, encrypted) ────────────────────────
# Returning from the external face scan opens a NEW browser tab → a fresh Streamlit
# session, so the profile held in session_state is gone and intake gets re-asked.
# We persist the profile server-side in Supabase, keyed by the user's email, and
# ENCRYPT it (Fernet symmetric, key derived from the app secret) so the stored row
# is ciphertext — readable only by the app, not by anyone who can see the DB.
try:
    from cryptography.fernet import Fernet
    _ENC_OK = True
except Exception:
    _ENC_OK = False

def _fernet():
    # 32-byte urlsafe key derived from the app secret (AUTH_COOKIE_SECRET / anon key)
    key = _b64.urlsafe_b64encode(hashlib.sha256(_cookie_secret().encode()).digest())
    return Fernet(key)

def save_draft(email, payload):
    sb = _supabase_client()
    if not sb or not email or not _ENC_OK:
        return
    try:
        blob = _fernet().encrypt(
            json.dumps(payload, ensure_ascii=False).encode()
        ).decode()
        sb.table("drafts").upsert({"user_email": email, "data": blob}, on_conflict="user_email").execute()
    except Exception:
        pass

def load_draft(email):
    sb = _supabase_client()
    if not sb or not email or not _ENC_OK:
        return None
    try:
        res = sb.table("drafts").select("data").eq("user_email", email).limit(1).execute()
        rows = res.data or []
        if rows and rows[0].get("data"):
            dec = _fernet().decrypt(rows[0]["data"].encode()).decode()
            return json.loads(dec)
    except Exception:
        return None
    return None

def delete_draft(email):
    sb = _supabase_client()
    if not sb or not email:
        return
    try:
        sb.table("drafts").delete().eq("user_email", email).execute()
    except Exception:
        pass

def _save_session_for_external_nav():
    """Save the current assessment to Supabase right before the user clicks an
    EXTERNAL navigation (face scan in a new tab). Called from the render that
    shows the scan link, so by click time the draft is in the DB and the new tab
    can restore it. Single-use, deleted immediately after restore."""
    if not (auth_enabled() and is_logged_in() and st.session_state.profile.get("name")):
        return
    payload = {
        "profile":         st.session_state.profile,
        "lang":            st.session_state.lang,
        "triage_chat":     st.session_state.triage_chat,
        "medications":     st.session_state.medications,
        "vitals_analysis": st.session_state.vitals_analysis,
    }
    save_draft(st.session_state.get("auth_user", ""), payload)


def send_otp(email):
    sb = _supabase_client()
    if not sb: return False, "Auth not configured."
    try:
        sb.auth.sign_in_with_otp({"email": email})
        return True, ""
    except Exception as e:
        return False, str(e)

def verify_otp(email, token):
    sb = _supabase_client()
    if not sb: return False, "Auth not configured."
    token = str(token).strip()
    last_err = "invalid"
    # New users (or with "Confirm email" on) get a 'signup' token; returning users get 'email'.
    for otp_type in ("email", "signup"):
        try:
            res = sb.auth.verify_otp({"email": email, "token": token, "type": otp_type})
            if getattr(res, "user", None):
                st.session_state["auth_user"] = email
                st.session_state["_hero_seen"] = True  # hero already seen — don't show again after login
                return True, ""
        except Exception as e:
            last_err = str(e)
    return False, last_err

def logout():
    sb = _supabase_client()
    if sb:
        try: sb.auth.sign_out()
        except Exception: pass
    delete_draft(st.session_state.get("auth_user", ""))
    _clear_login_cookie()
    # FULL RESET on exit — wipe assessment state and runtime flags. Keep language
    # preference (it's a UI choice, not assessment data).
    _lang_keep = st.session_state.get("lang", "el")
    for k in list(st.session_state.keys()):
        st.session_state.pop(k, None)
    for k, v in defaults.items():
        st.session_state[k] = v
    st.session_state["lang"] = _lang_keep
    try:
        if "pe" in st.query_params: del st.query_params["pe"]
    except Exception:
        pass

def render_login_gate():
    """Inline email->OTP login. Returns True once the user is logged in.

    UX-hardened (tester report, Cyprus): when Supabase rate-limits or returns a
    transient error, the email STILL gets delivered — but the previous version
    showed an error and never revealed the code-entry field. Result: the user has
    a code in their inbox and nowhere to type it. This version always advances to
    the code-entry stage after the send button is clicked, regardless of the API
    return code. If no code arrives the user can press 'Resend' or 'Different email'."""
    lang = st.session_state.lang
    if is_logged_in():
        return True

    # Friendly header
    st.markdown(f'''<div style="background:rgba(45,63,231,0.06);border:1px solid rgba(45,63,231,0.15);border-radius:14px;padding:20px 22px;text-align:center;margin:10px 0">
        <div style="font-size:34px;margin-bottom:6px">🔒</div>
        <div style="font-size:16px;font-weight:700;color:#1A1A2E">{"Σύνδεση" if lang=="el" else "Sign in"}</div>
        <div style="font-size:13px;color:#6B7280;margin-top:4px">{"Email + κωδικός μίας χρήσης. Χωρίς password." if lang=="el" else "Email + one-time code. No password."}</div>
    </div>''', unsafe_allow_html=True)

    # Recover pending email across mobile reloads / fresh tabs
    sent_to = st.session_state.get("otp_sent_to")
    if not sent_to:
        pe = st.query_params.get("pe")
        if pe:
            st.session_state["otp_sent_to"] = pe
            sent_to = pe

    if not sent_to:
        # ── STAGE 1: enter email ────────────────────────────────────────────
        email = st.text_input("Email", key="otp_email", placeholder="you@example.com")
        if st.button(("📩 " + ("Στείλε μου τον κωδικό" if lang=="el" else "Send me the code")),
                     type="primary", use_container_width=True, key="otp_send"):
            if email and "@" in email:
                # Best-effort send: ADVANCE to stage 2 regardless of API result.
                # Email is usually delivered even when Supabase rate-limits the
                # response — the user just needs the code field to appear.
                ok, err = send_otp(email)
                st.session_state["otp_sent_to"] = email
                st.query_params["pe"] = email
                if not ok:
                    st.session_state["_otp_send_warning"] = (err or "")[:140]
                st.rerun()
            else:
                st.warning("Έγκυρο email, παρακαλώ." if lang=="el" else "Please enter a valid email.")
    else:
        # ── STAGE 2: enter code ─────────────────────────────────────────────
        warn = st.session_state.pop("_otp_send_warning", None)
        if warn:
            # Soft warning — DON'T block the code field. Email may still have arrived.
            st.warning(("⚠️ Πιθανό πρόβλημα στην αποστολή — αλλά ο κωδικός μπορεί να έχει φτάσει στο email σου. "
                        "Έλεγξε το inbox και το spam folder, και βάλε τον κωδικό παρακάτω. "
                        "Αν δεν λάβεις τίποτα σε 1 λεπτό, πάτα «Νέος κωδικός»."
                        if lang=="el" else
                        "⚠️ The send response had an issue — but the code may still have reached your email. "
                        "Check your inbox and spam folder, then enter the code below. "
                        "If nothing arrives within 1 minute, press 'New code'."))
        else:
            st.success(f"📧 " + (f"Σου στείλαμε κωδικό στο **{sent_to}**" if lang=="el"
                                  else f"We sent a code to **{sent_to}**"))
        st.caption(("Έλεγξε το inbox και το spam folder. Ο κωδικός φτάνει σε λίγα δευτερόλεπτα."
                    if lang=="el" else
                    "Check your inbox and spam folder. The code arrives within a few seconds."))

        code = st.text_input(
            ("Κωδικός από το email" if lang=="el" else "Code from your email"),
            key="otp_code",
            placeholder="12345678",
            max_chars=8,
        )
        if st.button(("✓ " + ("Επιβεβαίωση & Σύνδεση" if lang=="el" else "Verify & Sign in")),
                     type="primary", use_container_width=True, key="otp_verify"):
            _code_clean = str(code or "").strip().replace(" ", "")
            if not _code_clean.isdigit() or len(_code_clean) < 6:
                st.warning(("Βάλε τον κωδικό από το email (6-8 ψηφία)." if lang=="el"
                            else "Enter the code from your email (6-8 digits)."))
            else:
                ok, err = verify_otp(sent_to, _code_clean)
                if ok:
                    st.session_state.pop("otp_sent_to", None)
                    if "pe" in st.query_params: del st.query_params["pe"]
                    st.rerun()
                else:
                    st.error(("Λάθος ή ληγμένος κωδικός — δοκίμασε ξανά ή πάτα «Νέος κωδικός»."
                              if lang=="el" else
                              "Wrong or expired code — try again or press 'New code'."))

        c1, c2 = st.columns(2)
        with c1:
            if st.button(("📩 " + ("Νέος κωδικός" if lang=="el" else "New code")),
                         use_container_width=True, key="otp_resend"):
                ok2, err2 = send_otp(sent_to)
                # Always show user-friendly message — code may have arrived regardless
                if ok2:
                    st.success(("Νέος κωδικός στάλθηκε." if lang=="el" else "New code sent."))
                else:
                    st.info(("Αν δεν λάβεις νέο κωδικό σε 60'', χρησιμοποίησε τον προηγούμενο που έλαβες."
                             if lang=="el" else
                             "If no new code arrives in 60s, use the previous one you received."))
        with c2:
            if st.button(("Άλλο email" if lang=="el" else "Different email"),
                         use_container_width=True, key="otp_reset"):
                st.session_state.pop("otp_sent_to", None)
                if "pe" in st.query_params: del st.query_params["pe"]
                st.rerun()

    return is_logged_in()

def render_ad_banner(lang):
    """Formeto-inspired value-prop banner shown on the login screen: soft-circle
    icons, large rounded cards, single blue accent — same visual language as
    the bottom-nav/history redesign, replacing the earlier Playfair-serif
    editorial treatment. Same honest claims as before:
      • Heart rate yes (rPPG is reliable for HR)
      • NO blood pressure or "30+ vitals" promise — rPPG can't reliably do those
      • GDPR not HIPAA — we are EU-based
    Cards visualize the actual product: chat bubble (symptoms),
    vitals readout (HR + BP/SpO₂ entered by user), report checklist."""
    if lang == "en":
        d = {
            "pill_l":"ASKLEPIOS · AI NURSE", "pill_r":"🔒 GDPR · Encrypted",
            "h_l":"Symptoms.", "h_m":"Assessment.", "h_r":"In Greek.",
            "sub":"Describe what you're feeling. Get a clinical assessment with PubMed references. In a few minutes.",
            "s1_lbl":"YOU", "s1_text":"\"Headache and nausea for 3 days…\"",
            "s2_lbl":"VITALS",
            "s2_v1":"HR", "s2_v1v":"78 bpm",
            "s2_v2":"BP", "s2_v2v":"120/80",
            "s3_lbl":"REPORT",
            "s3_l1":"Clinical assessment",
            "s3_l2":"PubMed references",
            "s3_l3":"Drug interactions",
            "t1":"🇬🇷 Greek", "t2":"🔒 GDPR",
            "t3":"📚 PubMed", "t4":"🤖 Claude + GPT-4o", "t5":"⚡ Free",
        }
    else:
        d = {
            "pill_l":"ASKLEPIOS · AI ΝΟΣΗΛΕΥΤΗΣ", "pill_r":"🔒 GDPR · Κρυπτογράφηση",
            "h_l":"Συμπτώματα.", "h_m":"Εκτίμηση.", "h_r":"Στα Ελληνικά.",
            "sub":"Περίγραψε τι νιώθεις. Λάβε κλινική εκτίμηση με τεκμηρίωση από PubMed. Σε λίγα λεπτά.",
            "s1_lbl":"ΕΣΥ", "s1_text":"«Πονοκέφαλος και ναυτία 3 μέρες…»",
            "s2_lbl":"ΖΩΤΙΚΑ",
            "s2_v1":"HR", "s2_v1v":"78 bpm",
            "s2_v2":"BP", "s2_v2v":"120/80",
            "s3_lbl":"ΑΝΑΦΟΡΑ",
            "s3_l1":"Κλινική εκτίμηση",
            "s3_l2":"Αναφορές PubMed",
            "s3_l3":"Αλληλεπιδράσεις",
            "t1":"🇬🇷 Ελληνικά", "t2":"🔒 GDPR",
            "t3":"📚 PubMed", "t4":"🤖 Claude + GPT-4o", "t5":"⚡ Δωρεάν",
        }
    css = """
<style>
.ad-hero {
  background: #F4F6FF;
  border-radius: 28px; padding: 44px 32px 32px;
  margin: 12px 0 28px; text-align: center;
  font-family: 'Inter', system-ui, sans-serif;
  border: 1px solid #E0E5FF;
}
.ad-pill {
  display: inline-flex; align-items: center; gap: 12px;
  background: white; border: 1px solid #E0E5FF;
  border-radius: 999px; padding: 8px 18px;
  font-size: 11.5px; font-weight: 700; letter-spacing: 0.08em;
  color: #2D3FE7; margin-bottom: 20px;
  box-shadow: 0 1px 3px rgba(45,63,231,0.06);
}
.ad-pill .sep { color: #D1D5DB; font-weight: 400; }
.ad-pill .gdpr { color: #10B981; letter-spacing: 0.04em; }
.ad-title {
  font-size: 42px; font-weight: 800; line-height: 1.12;
  letter-spacing: -1px; color: #1A1A2E; margin: 0 0 4px;
}
.ad-title .word { display: inline-block; }
.ad-title .accent { color: #2D3FE7; }
.ad-sub {
  font-size: 16px; color: #4B5563;
  max-width: 540px; margin: 16px auto 32px;
  line-height: 1.6; font-weight: 400;
}

/* Formeto-style square cards: soft-circle icon on top, content below */
.ad-flow {
  display: flex; align-items: stretch; justify-content: center;
  gap: 16px; margin: 32px 0 32px; flex-wrap: wrap;
}
.ad-card {
  background: white; border: 1px solid #E0E5FF;
  border-radius: 22px; padding: 22px 18px;
  width: 200px; max-width: 230px; min-height: 150px;
  box-shadow: 0 2px 10px rgba(45,63,231,0.05);
  display: flex; flex-direction: column; align-items: flex-start;
  text-align: left;
}
.ad-card-icon {
  width: 42px; height: 42px; border-radius: 50%;
  background: #E8ECFE; display: flex; align-items: center;
  justify-content: center; font-size: 19px; margin-bottom: 12px;
}
.ad-card-label {
  font-size: 10px; font-weight: 700; letter-spacing: 0.12em;
  color: #9CA3AF; text-transform: uppercase; margin-bottom: 10px;
}

/* Card 1: Chat bubble */
.ad-bubble {
  background: #F4F6FF; border-radius: 12px 12px 12px 4px;
  padding: 11px 13px; font-size: 13px;
  color: #1A1A2E; line-height: 1.45; font-style: italic;
  font-weight: 500; width: 100%;
}
/* Card 2: Vitals readout */
.ad-vitals { display: flex; flex-direction: column; gap: 8px; width: 100%; }
.ad-vital-row {
  display: flex; align-items: center; justify-content: space-between;
  background: #F7F8FC; border-radius: 9px;
  padding: 8px 11px; font-size: 12.5px;
}
.ad-vital-row .lbl { color: #6B7280; font-weight: 600; letter-spacing: 0.02em; }
.ad-vital-row .val { color: #1A1A2E; font-weight: 700; font-variant-numeric: tabular-nums; }
/* Card 3: Report checklist */
.ad-report { display: flex; flex-direction: column; gap: 7px; width: 100%; }
.ad-report-line {
  display: flex; align-items: center; gap: 9px;
  font-size: 13px; color: #1A1A2E; font-weight: 500;
}
.ad-report-line .check {
  width: 18px; height: 18px; border-radius: 50%;
  background: #ECFDF5; color: #059669;
  display: inline-flex; align-items: center; justify-content: center;
  font-size: 11px; font-weight: 700; flex-shrink: 0;
}
.ad-arrow {
  display: flex; align-items: center;
  font-size: 20px; color: #2D3FE7; font-weight: 700; opacity: 0.4;
}

/* Trust badges — inline with dot separators */
.ad-trust {
  display: flex; justify-content: center; align-items: center;
  gap: 10px; flex-wrap: wrap; font-size: 12.5px;
  color: #6B7280; font-weight: 500;
  padding-top: 14px; border-top: 1px solid #E0E5FF;
  margin-top: 16px;
}
.ad-trust .item { white-space: nowrap; }
.ad-trust .sep-dot {
  color: #D1D5DB; font-weight: 400; font-size: 14px;
  line-height: 1;
}

@media (max-width: 640px) {
  .ad-hero { padding: 32px 20px 24px; border-radius: 22px; }
  .ad-title { font-size: 30px; letter-spacing: -0.6px; }
  .ad-sub { font-size: 14.5px; margin: 14px auto 24px; }
  .ad-arrow { display: none; }
  .ad-card { width: 100%; max-width: 340px; padding: 18px; min-height: auto; }
  .ad-flow { gap: 10px; margin: 24px 0 24px; }
  .ad-trust { gap: 6px; font-size: 11.5px; }
  .ad-pill { font-size: 10.5px; padding: 7px 14px; }
}
</style>
"""
    body = f"""
<div class="ad-hero">
  <div class="ad-pill">✦ {d["pill_l"]} <span class="sep">|</span> <span class="gdpr">{d["pill_r"]}</span></div>
  <h1 class="ad-title">
    <span class="word">{d["h_l"]}</span>
    <span class="word">{d["h_m"]}</span><br>
    <span class="word accent">{d["h_r"]}</span>
  </h1>
  <p class="ad-sub">{d["sub"]}</p>
  <div class="ad-flow">
    <div class="ad-card ad-card-1">
      <div class="ad-card-icon">💬</div>
      <div class="ad-card-label">{d["s1_lbl"]}</div>
      <div class="ad-bubble">{d["s1_text"]}</div>
    </div>
    <div class="ad-arrow">→</div>
    <div class="ad-card ad-card-2">
      <div class="ad-card-icon">❤️</div>
      <div class="ad-card-label">{d["s2_lbl"]}</div>
      <div class="ad-vitals">
        <div class="ad-vital-row"><span class="lbl">❤️ {d["s2_v1"]}</span><span class="val">{d["s2_v1v"]}</span></div>
        <div class="ad-vital-row"><span class="lbl">💉 {d["s2_v2"]}</span><span class="val">{d["s2_v2v"]}</span></div>
      </div>
    </div>
    <div class="ad-arrow">→</div>
    <div class="ad-card ad-card-3">
      <div class="ad-card-icon">📋</div>
      <div class="ad-card-label">{d["s3_lbl"]}</div>
      <div class="ad-report">
        <div class="ad-report-line"><span class="check">✓</span>{d["s3_l1"]}</div>
        <div class="ad-report-line"><span class="check">✓</span>{d["s3_l2"]}</div>
        <div class="ad-report-line"><span class="check">✓</span>{d["s3_l3"]}</div>
      </div>
    </div>
  </div>
  <div class="ad-trust">
    <span class="item">{d["t1"]}</span><span class="sep-dot">·</span>
    <span class="item">{d["t2"]}</span><span class="sep-dot">·</span>
    <span class="item">{d["t3"]}</span><span class="sep-dot">·</span>
    <span class="item">{d["t4"]}</span><span class="sep-dot">·</span>
    <span class="item">{d["t5"]}</span>
  </div>
</div>
"""
    st.markdown(css + body, unsafe_allow_html=True)


def render_explainer_video(lang):
    """Photo-slider style walkthrough: horizontal scrollable cards (one per step).
    Mobile: swipeable with snap. Desktop: 3-4 cards visible + scroll. No iframe,
    no JS, no tabs — visible immediately."""
    el = (lang == "el")
    if el:
        steps = [
            ("01", "🩺", "#EEF6FF", "ASKLEPIOS",
             "Ο ψηφιακός σου νοσηλευτής",
             "Αξιολόγηση συμπτωμάτων με τεχνητή νοημοσύνη — γρήγορα, στα Ελληνικά."),
            ("02", "✉️", "#F0EEFE", "ΣΥΝΔΕΣΗ",
             "Σύνδεση με email",
             "Email + κωδικός μίας χρήσης. Χωρίς password, χωρίς πολύπλοκη εγγραφή."),
            ("03", "👤", "#ECFDF5", "ΠΡΟΦΙΛ",
             "Συμπλήρωσε το προφίλ σου",
             "Όνομα, ηλικία, φύλο, ιατρικό ιστορικό, αλλεργίες, φάρμακα."),
            ("04", "💬", "#FFF7ED", "ΣΥΜΠΤΩΜΑΤΑ",
             "Περίγραψε τι νιώθεις",
             "Ο Asklepios κάνει στοχευμένες ερωτήσεις — μία κάθε φορά."),
            ("05", "❤️", "#FEF2F2", "ΖΩΤΙΚΑ",
             "Μέτρηση ζωτικών — 3 επιλογές",
             "Χειροκίνητα · συσκευή · σάρωση προσώπου (καρδιακός ρυθμός)."),
            ("06", "📷", "#F0FDFA", "ΦΩΤΟ",
             "Φωτογραφία — μόνο αν χρειαστεί",
             "Προτείνεται για ορατά συμπτώματα: δερματικά, τραύματα, εξογκώματα."),
            ("07", "📋", "#FDF4FF", "ΑΝΑΦΟΡΑ",
             "Αναλυτική αναφορά υγείας",
             "Κλινική εκτίμηση με PubMed + GPT-4o δεύτερη γνώμη. PDF για τον γιατρό σου."),
        ]
        header = "Πώς λειτουργεί"
        hint   = "← σύρε για περισσότερα →"
    else:
        steps = [
            ("01", "🩺", "#EEF6FF", "ASKLEPIOS",
             "Your digital nurse",
             "AI-powered symptom assessment — fast, in your language."),
            ("02", "✉️", "#F0EEFE", "SIGN-IN",
             "Sign in with email",
             "Email + one-time code. No password, no complex registration."),
            ("03", "👤", "#ECFDF5", "PROFILE",
             "Fill in your profile",
             "Name, age, sex, medical history, allergies, medications."),
            ("04", "💬", "#FFF7ED", "SYMPTOMS",
             "Describe what you're feeling",
             "Asklepios asks targeted questions — one at a time."),
            ("05", "❤️", "#FEF2F2", "VITALS",
             "Measure vitals — 3 options",
             "Manual entry · device · face scan (heart rate only)."),
            ("06", "📷", "#F0FDFA", "PHOTO",
             "Photo — only when needed",
             "Suggested for visible symptoms: skin, wounds, lumps."),
            ("07", "📋", "#FDF4FF", "REPORT",
             "Detailed health report",
             "Clinical assessment with PubMed + GPT-4o second opinion. PDF for your doctor."),
        ]
        header = "How it works"
        hint   = "← swipe for more →"
    cards = "".join(
        f"""<div class="exp-card" style="background:{tint};">
              <div class="exp-num">{num}</div>
              <div class="exp-icon">{icon}</div>
              <div class="exp-label">{label}</div>
              <div class="exp-title">{title}</div>
              <div class="exp-sub">{sub}</div>
            </div>"""
        for (num, icon, tint, label, title, sub) in steps
    )
    st.markdown(
        f"""
<style>
.exp-section {{
  margin: 32px 0 16px;
}}
.exp-header {{
  display: flex; justify-content: space-between; align-items: baseline;
  margin: 0 4px 12px;
  font-family: 'Inter', system-ui, sans-serif;
}}
.exp-header .ttl {{
  font-size: 18px; font-weight: 700; color: #1A1A2E;
  letter-spacing: -0.01em;
}}
.exp-header .hint {{
  font-size: 11px; color: #9CA3AF; font-weight: 500;
  letter-spacing: 0.02em;
}}
.exp-scroll {{
  display: flex; gap: 12px;
  overflow-x: auto; overflow-y: hidden;
  padding: 4px 4px 18px;
  scroll-snap-type: x mandatory;
  -webkit-overflow-scrolling: touch;
  scrollbar-width: thin;
  scrollbar-color: #CBD5E1 transparent;
}}
.exp-scroll::-webkit-scrollbar {{ height: 6px; }}
.exp-scroll::-webkit-scrollbar-thumb {{
  background: #CBD5E1; border-radius: 3px;
}}
.exp-scroll::-webkit-scrollbar-track {{ background: transparent; }}
.exp-card {{
  flex: 0 0 250px; max-width: 250px;
  border-radius: 18px; padding: 22px 20px;
  scroll-snap-align: start;
  border: 1px solid rgba(0,0,0,0.04);
  text-align: left;
  font-family: 'Inter', system-ui, sans-serif;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}}
.exp-num {{
  font-size: 11px; font-weight: 800; letter-spacing: 0.14em;
  color: rgba(0,0,0,0.28); margin-bottom: 12px;
}}
.exp-icon {{
  font-size: 30px; line-height: 1; margin-bottom: 10px;
}}
.exp-label {{
  font-size: 9.5px; font-weight: 700; letter-spacing: 0.14em;
  color: #9CA3AF; text-transform: uppercase; margin-bottom: 6px;
}}
.exp-title {{
  font-size: 15px; font-weight: 700; color: #1A1A2E;
  line-height: 1.35; margin-bottom: 8px;
}}
.exp-sub {{
  font-size: 12.5px; color: #4B5563; line-height: 1.55;
}}
@media (max-width: 640px) {{
  .exp-card {{ flex: 0 0 220px; padding: 18px 16px; }}
  .exp-icon {{ font-size: 26px; }}
  .exp-title {{ font-size: 14px; }}
  .exp-sub {{ font-size: 12px; }}
}}
</style>
<div class="exp-section">
  <div class="exp-header"><span class="ttl">{header}</span><span class="hint">{hint}</span></div>
  <div class="exp-scroll">{cards}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_login_screen():
    """Hero landing + login. Shows value-prop, gov.gr links, how-it-works,
    and audience cards before (and around) the OTP login form."""
    lang = st.session_state.lang
    el = (lang == "el")

    # ── Language picker — top of page, first thing the user sees ─────────────
    _lp_css = """
<style>
.lang-bar{display:flex;align-items:center;justify-content:space-between;
  padding:10px 4px 14px;font-family:'Inter',system-ui,sans-serif;}
.lang-bar-logo{font-size:17px;font-weight:800;color:#1A1A2E;}
.lang-bar-logo span{color:#2D3FE7;}
.lang-btns{display:flex;gap:6px;}
.lang-btn{background:#F4F6FF;border:1px solid #E0E5FF;border-radius:999px;
  padding:6px 14px;font-size:12px;font-weight:700;color:#2D3FE7;cursor:pointer;
  font-family:'Inter',system-ui,sans-serif;}
.lang-btn.active{background:#2D3FE7;color:white;border-color:#2D3FE7;}
</style>
"""
    st.markdown(_lp_css, unsafe_allow_html=True)
    lc1, lc2 = st.columns([6, 1])
    with lc1:
        st.markdown(
            '<div class="lang-bar"><div class="lang-bar-logo">⚕ <span>Asklepios</span></div></div>',
            unsafe_allow_html=True)
    with lc2:
        if st.button("🇬🇧 EN" if el else "🇬🇷 ΕΛ", key="login_lang"):
            st.session_state.lang = "en" if el else "el"; st.rerun()

    # ── HERO CARD ─────────────────────────────────────────────────────────────
    _h1     = ("Περίγραψε τι νιώθεις.<br><span style='color:#2D3FE7'>Λάβε κλινική εκτίμηση.</span>"
               if el else
               "Describe what you feel.<br><span style='color:#2D3FE7'>Get a clinical assessment.</span>")
    _sub    = ("Τεκμηριωμένη αξιολόγηση με αναφορές PubMed + δεύτερη γνώμη GPT-4o. Για τον <strong>ιατρό</strong> σου. Στα Ελληνικά."
               if el else
               "Evidence-based assessment with PubMed references + GPT-4o second opinion. For your <strong>doctor</strong>.")
    _f1t = "Περιγραφή συμπτωμάτων" if el else "Symptom description"
    _f1s = "Μιλάς φυσικά — το AI ρωτά & οργανώνει" if el else "Speak naturally — AI asks & organises"
    _f2t = "Καταγραφή ζωτικών" if el else "Vital signs"
    _f2s = "HR, BP, SpO₂, θερμοκρασία" if el else "HR, BP, SpO₂, temperature"
    _f3t = "Εξετάσεις & φωτογραφία" if el else "Lab results & photos"
    _f3s = "Ανέβασε αιματολογικά, PDF εξετάσεων ή φωτογραφία" if el else "Upload blood tests, PDF results or a photo"
    _f4t = "Κλινική αναφορά για τον ιατρό" if el else "Clinical report for your doctor"
    _f4s = "PubMed + δεύτερη γνώμη GPT-4o" if el else "PubMed + GPT-4o second opinion"
    _p1t  = "Δεύτερη ιατρική γνώμη" if el else "Second medical opinion"
    _p1s  = "Claude + GPT-4o ανεξάρτητα — σύγκριση αξιολογήσεων" if el else "Claude + GPT-4o independently — compare assessments"
    _p1b  = "Μοναδικό" if el else "Unique"
    _p2t  = "Σύνδεση gov.gr" if el else "gov.gr integration"
    _p2s  = "Ηλεκτρ. Φάκελος, ΑΜΚΑ, e-Συνταγογράφηση" if el else "Health record, AMKA, e-Prescription"
    _p2b  = "Ελληνικό ΣΥ" if el else "Greek NHS"
    _disc = ("Δεν αντικαθιστά τον <strong>ιατρό</strong>. Σε επείγον: <strong>166</strong> (ΕΚΑΒ) ή <strong>112</strong>. Δεν αποθηκεύουμε ιατρικά δεδομένα. 🔒 GDPR"
             if el else
             "Does not replace your <strong>doctor</strong>. In emergency: <strong>112</strong>. We store no medical data. 🔒 GDPR")

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
.ask-hr-hero{{background:#F4F6FF;border:1px solid #E0E5FF;border-radius:24px;
  padding:30px 22px 22px;margin:0 0 16px;text-align:center;
  font-family:'Inter',system-ui,sans-serif;}}
.ask-hr-kicker{{font-size:10px;font-weight:700;letter-spacing:0.18em;color:#2D3FE7;
  margin-bottom:12px;text-transform:uppercase;}}
.ask-hr-h1{{font-size:28px;font-weight:800;line-height:1.18;color:#1A1A2E;
  letter-spacing:-0.5px;margin-bottom:12px;}}
.ask-hr-sub{{font-size:14px;color:#4B5563;line-height:1.6;max-width:400px;
  margin:0 auto 20px;}}
.ask-hr-fcards{{display:flex;flex-direction:column;gap:8px;text-align:left;margin-bottom:14px;}}
.ask-hr-fc{{background:white;border:1px solid #E0E5FF;border-radius:12px;
  padding:10px 13px;display:flex;align-items:center;gap:10px;}}
.ask-hr-fc-ic{{width:30px;height:30px;border-radius:50%;display:flex;
  align-items:center;justify-content:center;font-size:14px;flex-shrink:0;}}
.ask-hr-fc-ic.blue{{background:#E8ECFE;}}
.ask-hr-fc-ic.red{{background:#FEE2E2;}}
.ask-hr-fc-ic.grn{{background:#ECFDF5;}}
.ask-hr-fc-txt{{font-size:12.5px;font-weight:600;color:#1A1A2E;flex:1;line-height:1.3;}}
.ask-hr-fc-txt small{{font-weight:400;color:#6B7280;display:block;font-size:11px;}}
.ask-hr-fc-badge{{background:#E8ECFE;color:#2D3FE7;font-size:10.5px;font-weight:700;
  padding:2px 8px;border-radius:999px;flex-shrink:0;white-space:nowrap;}}
.ask-hr-fc-badge.ok{{background:#ECFDF5;color:#059669;}}
.ask-hr-power{{display:grid;grid-template-columns:1fr 1fr;gap:9px;margin-bottom:13px;}}
.ask-hr-pow{{background:white;border:1.5px solid #2D3FE7;border-radius:12px;
  padding:11px 12px;display:flex;align-items:flex-start;gap:9px;text-align:left;}}
.ask-hr-pow.gov{{border-color:#059669;}}
.ask-hr-pow-ic{{font-size:18px;flex-shrink:0;margin-top:1px;}}
.ask-hr-pow-t{{font-size:11.5px;font-weight:700;color:#1A1A2E;margin-bottom:2px;}}
.ask-hr-pow-s{{font-size:10.5px;color:#6B7280;line-height:1.4;}}
.ask-hr-pow-tag{{display:inline-block;background:#E8ECFE;color:#2D3FE7;
  font-size:10px;font-weight:700;padding:2px 7px;border-radius:999px;margin-top:4px;}}
.ask-hr-pow.gov .ask-hr-pow-tag{{background:#ECFDF5;color:#059669;}}
.ask-hr-disc{{background:white;border:1px solid #E5E7EB;border-radius:10px;
  padding:9px 12px;font-size:11px;color:#6B7280;line-height:1.5;
  display:flex;gap:8px;align-items:flex-start;text-align:left;}}
</style>
<div class="ask-hr-hero">
  <div class="ask-hr-kicker">ASKLEPIOS · {"AI ΝΟΣΗΛΕΥΤΗΣ" if el else "AI NURSE"}</div>
  <div class="ask-hr-h1">{_h1}</div>
  <div class="ask-hr-sub">{_sub}</div>
  <div class="ask-hr-fcards">
    <div class="ask-hr-fc">
      <div class="ask-hr-fc-ic blue">💬</div>
      <div class="ask-hr-fc-txt">{_f1t}<small>{_f1s}</small></div>
      <div class="ask-hr-fc-badge">{"Βήμα 1" if el else "Step 1"}</div>
    </div>
    <div class="ask-hr-fc">
      <div class="ask-hr-fc-ic red">❤️</div>
      <div class="ask-hr-fc-txt">{_f2t}<small>{_f2s}</small></div>
      <div class="ask-hr-fc-badge">{"Βήμα 2" if el else "Step 2"}</div>
    </div>
    <div class="ask-hr-fc">
      <div class="ask-hr-fc-ic blue">🧬</div>
      <div class="ask-hr-fc-txt">{_f3t}<small>{_f3s}</small></div>
      <div class="ask-hr-fc-badge">{"Βήμα 3" if el else "Step 3"}</div>
    </div>
    <div class="ask-hr-fc">
      <div class="ask-hr-fc-ic grn">📋</div>
      <div class="ask-hr-fc-txt">{_f4t}<small>{_f4s}</small></div>
      <div class="ask-hr-fc-badge ok">✓ {"Αναφορά" if el else "Report"}</div>
    </div>
  </div>
  <div class="ask-hr-power">
    <div class="ask-hr-pow">
      <div class="ask-hr-pow-ic">🤖</div>
      <div>
        <div class="ask-hr-pow-t">{_p1t}</div>
        <div class="ask-hr-pow-s">{_p1s}</div>
        <div class="ask-hr-pow-tag">{_p1b}</div>
      </div>
    </div>
    <div class="ask-hr-pow gov">
      <div class="ask-hr-pow-ic">🇬🇷</div>
      <div>
        <div class="ask-hr-pow-t">{_p2t}</div>
        <div class="ask-hr-pow-s">{_p2s}</div>
        <div class="ask-hr-pow-tag">{_p2b}</div>
      </div>
    </div>
  </div>
  <div class="ask-hr-disc">
    <span style="font-size:14px;flex-shrink:0">ℹ️</span>
    <span>{_disc}</span>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── gov.gr QUICK ACCESS ───────────────────────────────────────────────────
    _gov_title = "🔒 Ηλεκτρονικές Υπηρεσίες Υγείας (gov.gr)" if el else "🔒 Digital Health Services (gov.gr)"
    _gov_note  = "Ανοίγουν σε νέα καρτέλα στο gov.gr — δεν αποθηκεύουμε δεδομένα." if el else "Open in a new tab on gov.gr — we store no data."
    _gov_links = [
        ("📂", "Ηλεκτρονικός Φάκελος Υγείας" if el else "Health Record",
         "https://www.gov.gr/ipiresies/ugeia-kai-pronoia/phakelos-ugeias"),
        ("💊", "e-Συνταγογράφηση" if el else "e-Prescription",
         "https://esyntagografisi.amka.gr"),
        ("👨‍⚕️", "Ιατροί ΕΟΠΥΥ" if el else "EOPYY Doctors",
         "https://www.eopyy.gov.gr"),
        ("📋", "ΑΜΚΑ" if el else "AMKA",
         "https://www.gov.gr/ipiresies/apasxolisi-kai-syntaxiodotisi/amka"),
        ("🔔", "ΕΟΔΥ" if el else "EODY",
         "https://eody.gov.gr"),
    ]
    _gov_btn_html = " ".join(
        f'<a href="{url}" target="_blank" rel="noopener" style="'
        f'display:inline-flex;align-items:center;gap:5px;background:white;'
        f'border:1px solid #6EE7B7;border-radius:999px;padding:7px 13px;'
        f'font-size:12px;font-weight:600;color:#065F46;text-decoration:none;'
        f'margin:3px 2px;">{ic} {label}</a>'
        for ic, label, url in _gov_links
    )
    st.markdown(f"""
<div style="background:#F0FDF4;border:1.5px solid #A7F3D0;border-radius:16px;
  padding:15px 17px;margin:0 0 18px;font-family:'Inter',system-ui,sans-serif;">
  <div style="font-size:11px;font-weight:700;letter-spacing:0.1em;color:#065F46;
    text-transform:uppercase;margin-bottom:11px;">{_gov_title}</div>
  <div style="line-height:2;">{_gov_btn_html}</div>
  <div style="font-size:10.5px;color:#6B7280;margin-top:9px;">{_gov_note}</div>
</div>
""", unsafe_allow_html=True)

    # ── HOW IT WORKS ─────────────────────────────────────────────────────────
    _steps_el = [
        ("1","👤","Προφίλ","Ηλικία, φύλο, ιστορικό"),
        ("2","💬","Συμπτώματα","Το AI ρωτά δομημένα"),
        ("3","❤️","Ζωτικά","HR, BP, SpO₂"),
        ("4","🧬","Εξετάσεις","Ανέβασε αιματολογικά, PDF εξετάσεων, φωτογραφία πληγής/δέρματος"),
        ("5","🧠","Triage AI","Claude + GPT-4o"),
        ("6","📄","Αναφορά","PubMed + ιατρός"),
    ]
    _steps_en = [
        ("1","👤","Profile","Age, sex, history"),
        ("2","💬","Symptoms","AI asks step by step"),
        ("3","❤️","Vitals","HR, BP, SpO₂"),
        ("4","🧬","Lab & Photos","Upload blood tests, PDF results, wound/skin photo"),
        ("5","🧠","Triage AI","Claude + GPT-4o"),
        ("6","📄","Report","PubMed + doctor"),
    ]
    _steps = _steps_el if el else _steps_en
    _arrow = '<div style="flex:0 0 auto;align-self:flex-start;padding-top:14px;font-size:11px;color:#C7D2FE;">›</div>'
    _steps_with_arrows = _arrow.join(f"""<div style="flex:1 1 0;min-width:0;text-align:center;padding:0 2px;">
  <div style="width:28px;height:28px;border-radius:50%;background:#2D3FE7;color:white;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:12px;margin:0 auto 8px;">{n}</div>
  <div style="font-size:20px;margin-bottom:5px;">{ic}</div>
  <div style="font-size:12px;font-weight:700;color:#1A1A2E;margin-bottom:2px;line-height:1.25;">{t}</div>
  <div style="font-size:10px;color:#6B7280;line-height:1.35;">{s}</div>
</div>""" for n, ic, t, s in _steps)
    _how_title = "Πώς λειτουργεί" if el else "How it works"
    st.markdown(f"""
<div style="font-family:'Inter',system-ui,sans-serif;margin:0 0 20px;">
  <div style="font-size:18px;font-weight:800;color:#1A1A2E;text-align:center;margin-bottom:16px;">{_how_title}</div>
  <div style="display:flex;align-items:flex-start;gap:0;width:100%;">{_steps_with_arrows}</div>
</div>
""", unsafe_allow_html=True)

    # ── STATS BAND ────────────────────────────────────────────────────────────
    _s1l = "Ακρίβεια triage (Semigran-45)" if el else "Triage accuracy (Semigran-45)"
    _s2l = "Unsafe undertriage" if el else "Unsafe undertriage"
    _s3l = "Claude + GPT-4o δεύτερη γνώμη" if el else "Claude + GPT-4o second opinion"
    st.markdown(f"""
<div style="display:flex;border:1px solid #E0E5FF;border-radius:14px;overflow:hidden;
  background:white;margin:0 0 20px;font-family:'Inter',system-ui,sans-serif;">
  <div style="flex:1;text-align:center;padding:13px 6px;border-right:1px solid #E0E5FF;">
    <div style="font-size:19px;font-weight:800;color:#2D3FE7;">88.9%</div>
    <div style="font-size:10.5px;color:#6B7280;margin-top:3px;line-height:1.3;">{_s1l}</div>
  </div>
  <div style="flex:1;text-align:center;padding:13px 6px;border-right:1px solid #E0E5FF;">
    <div style="font-size:19px;font-weight:800;color:#2D3FE7;">0%</div>
    <div style="font-size:10.5px;color:#6B7280;margin-top:3px;line-height:1.3;">{_s2l}</div>
  </div>
  <div style="flex:1;text-align:center;padding:13px 6px;">
    <div style="font-size:19px;font-weight:800;color:#2D3FE7;">2 AI</div>
    <div style="font-size:10.5px;color:#6B7280;margin-top:3px;line-height:1.3;">{_s3l}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── FOR WHOM ─────────────────────────────────────────────────────────────
    _aud_title = "Για ποιον είναι" if el else "Who is it for"
    if el:
        _aud = [
            ("👨‍👩‍👧","ic-blue","Για όλη την οικογένεια",
             "Για σένα, τα παιδιά σου, τους γονείς σου. Πήγαινε στον ιατρό έτοιμος, με ιστορικό στα χέρια σου.",
             "Ενήλικες · Παιδιά · Ηλικιωμένοι","#E8ECFE","#2D3FE7"),
            ("🤝","ic-amber","Για φροντιστές υγείας",
             "Φροντίζεις κάποιον άλλο; Το Asklepios δουλεύει σε caregiver mode — περιγράφεις αυτό που παρατηρείς και λαμβάνεις δομημένη εκτίμηση.",
             "Caregiver mode ενσωματωμένο","#FFFBEB","#92400E"),
            ("👨‍⚕️","ic-green","Για ιατρούς & ιατρεία",
             "Ο ασθενής φτάνει με οργανωμένο ιστορικό και PubMed εκτίμηση. Λιγότερα τηλεφωνήματα ρουτίνας, ποιοτικότερος χρόνος.",
             "Εξοικονόμηση χρόνου · Ποιοτικότερες επισκέψεις","#ECFDF5","#065F46"),
        ]
    else:
        _aud = [
            ("👨‍👩‍👧","ic-blue","For the whole family",
             "For you, your children, your parents. Go to your doctor prepared, with an organised history.",
             "Adults · Children · Elderly","#E8ECFE","#2D3FE7"),
            ("🤝","ic-amber","For caregivers",
             "Caring for someone else? Asklepios works in caregiver mode — describe what you observe and get a structured assessment.",
             "Caregiver mode built-in","#FFFBEB","#92400E"),
            ("👨‍⚕️","ic-green","For doctors & clinics",
             "Patients arrive with organised history and PubMed assessment. Fewer routine calls, higher-quality consultations.",
             "Save time · Better appointments","#ECFDF5","#065F46"),
        ]
    _aud_cards = "".join(f"""
<div style="flex:1 1 180px;background:white;border:1px solid #E0E5FF;border-radius:14px;padding:14px 14px 12px;">
  <div style="width:34px;height:34px;border-radius:50%;background:{bg};display:flex;
    align-items:center;justify-content:center;font-size:17px;margin-bottom:9px;">{ic}</div>
  <div style="font-size:13px;font-weight:800;color:#1A1A2E;margin-bottom:4px;">{t}</div>
  <div style="font-size:11.5px;color:#4B5563;line-height:1.5;margin-bottom:7px;">{d}</div>
  <div style="display:inline-block;background:{bg};color:{tc};font-size:10px;font-weight:700;
    padding:2px 9px;border-radius:999px;">{badge}</div>
</div>""" for ic, _, t, d, badge, bg, tc in _aud)
    st.markdown(f"""
<div style="font-family:'Inter',system-ui,sans-serif;margin:0 0 22px;">
  <div style="font-size:18px;font-weight:800;color:#1A1A2E;text-align:center;margin-bottom:16px;">{_aud_title}</div>
  <div style="display:flex;gap:10px;flex-wrap:wrap;">{_aud_cards}</div>
</div>
""", unsafe_allow_html=True)

    # ── LOGIN FORM / CONTINUE BUTTON ─────────────────────────────────────────
    _login_title = "Ξεκίνα — δωρεάν, χωρίς password" if el else "Get started — free, no password"
    st.markdown(f"""
<div style="font-size:16px;font-weight:800;color:#1A1A2E;text-align:center;
  margin:4px 0 12px;font-family:'Inter',system-ui,sans-serif;">{_login_title}</div>
""", unsafe_allow_html=True)

    # If already logged in, show a single "Συνέχεια" CTA that sets _hero_seen
    # and sends the user to home. Otherwise show the OTP login form.
    if is_logged_in():
        _cta_lbl = "✦ Ξεκίνα αξιολόγηση & αναφορά" if el else "✦ Start assessment & report"
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(_cta_lbl, type="primary", use_container_width=True, key="hero_cta_loggedin"):
                st.session_state["_hero_seen"] = True
                st.rerun()
    else:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            render_login_gate()
        # render_login_gate sets is_logged_in() on success and st.rerun()s.
        # After rerun the block above (is_logged_in()) fires and shows the CTA.

    st.markdown(f'<div class="disclaimer">{t("disclaimer_main")}</div>', unsafe_allow_html=True)


# ── ADMIN PANEL ─────────────────────────────────────────────────────────────────
# Reached via ?admin=1 in the URL (e.g. asklepiosainurse.up.railway.app/?admin=1).
# Completely separate from the patient OTP login — gated by a single shared
# password (ADMIN_PASSWORD secret), since this is for internal/business use,
# not patient accounts. Session-only: closing the tab requires re-entering
# the password (no persistent admin cookie, by design — this controls content
# other people see, so we don't want it staying logged in indefinitely on a
# shared/public computer).

def _admin_is_unlocked():
    return bool(st.session_state.get("_admin_unlocked"))

def render_admin_gate():
    """Password prompt for the admin panel. Returns True once unlocked."""
    if _admin_is_unlocked():
        return True
    st.markdown("""
<div style="max-width:380px;margin:60px auto 0;text-align:center">
  <div style="font-size:34px;margin-bottom:8px">🔐</div>
  <div style="font-size:18px;font-weight:800;color:#1A1A2E">Admin Panel</div>
  <div style="font-size:13px;color:#6B7280;margin-top:4px">Asklepios · Internal</div>
</div>
""", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        with st.container(border=True):
            pw = st.text_input("Password", type="password", key="admin_pw_input")
            if st.button("Unlock", type="primary", use_container_width=True, key="admin_unlock_btn"):
                expected = get_admin_password()
                if not expected:
                    st.error("ADMIN_PASSWORD δεν έχει ρυθμιστεί στα Railway Variables.")
                elif pw == expected:
                    st.session_state["_admin_unlocked"] = True
                    st.rerun()
                else:
                    st.error("Λάθος password.")
    return False


def _admin_video_tab():
    st.markdown("#### 🎬 Intro Video")
    st.caption(
        "Το βίντεο που βλέπουν οι χρήστες στην αρχική σελίδα. Παράγεται εκτός "
        "εφαρμογής (π.χ. με A2E Talking Photo) — εδώ μπαίνει μόνο το τελικό URL."
    )
    for lang_code, lang_lbl in (("el", "🇬🇷 Ελληνικά"), ("en", "🇬🇧 English")):
        cur = get_setting(f"intro_video_url_{lang_code}", "") or ""
        with st.container(border=True):
            st.markdown(f"**{lang_lbl}**")
            new_url = st.text_input(
                "Video URL", value=cur, key=f"admin_video_{lang_code}",
                label_visibility="collapsed", placeholder="https://.../video.mp4",
            )
            if new_url:
                st.video(new_url)
            if st.button("Αποθήκευση", key=f"admin_video_save_{lang_code}"):
                if set_setting(f"intro_video_url_{lang_code}", new_url):
                    st.success("Αποθηκεύτηκε.")
                else:
                    st.error("Αποτυχία αποθήκευσης — έλεγξε ότι υπάρχει ο πίνακας app_settings στη Supabase.")
    st.caption(
        "ℹ️ Αυτό αντικαθιστά τα παλιά secrets A2E_INTRO_VIDEO_URL_EL/_EN — αν "
        "υπάρχει τιμή εδώ, έχει προτεραιότητα έναντι του secret."
    )


def _admin_partners_tab():
    st.markdown("#### 🏥 Συμβεβλημένα Νοσοκομεία / Ιατρεία")
    st.caption("Λίστα partners για τη B2B έκδοση — εμφανίζονται στο app στους χρήστες όταν χρειάζονται παραπομπή.")
    with st.expander("➕ Νέος Partner", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            p_name = st.text_input("Όνομα", key="np_name")
            p_spec = st.text_input("Ειδικότητα / Τύπος", key="np_spec", placeholder="π.χ. Καρδιολογικό, Γενικό Νοσοκομείο")
            p_city = st.text_input("Πόλη", key="np_city")
        with c2:
            p_phone = st.text_input("Τηλέφωνο", key="np_phone")
            p_email = st.text_input("Email", key="np_email")
            p_site  = st.text_input("Website", key="np_site")
        p_notes = st.text_area("Σημειώσεις", key="np_notes", height=70)
        if st.button("Προσθήκη", type="primary", key="np_add"):
            if p_name:
                ok = _admin_insert("partners", {
                    "name": p_name, "specialty": p_spec, "city": p_city,
                    "phone": p_phone, "email": p_email, "website": p_site,
                    "notes": p_notes, "active": True,
                })
                if ok:
                    st.success(f"Προστέθηκε: {p_name}"); st.rerun()
                else:
                    st.error("Αποτυχία — έλεγξε ότι υπάρχει ο πίνακας 'partners' στη Supabase.")
            else:
                st.warning("Το όνομα είναι απαραίτητο.")

    partners = _admin_list("partners")
    if not partners:
        st.info("Δεν υπάρχουν partners ακόμα.")
    for row in partners:
        with st.container(border=True):
            c1, c2 = st.columns([5,1])
            with c1:
                st.markdown(f"**{row.get('name','—')}** · {row.get('specialty','—')} · {row.get('city','—')}")
                _contact = " · ".join(x for x in [row.get("phone",""), row.get("email",""), row.get("website","")] if x)
                if _contact:
                    st.caption(_contact)
                if row.get("notes"):
                    st.caption(f"📝 {row['notes']}")
            with c2:
                _active = row.get("active", True)
                _rid = row.get("id", "")
                if st.button("✓ Ενεργό" if _active else "✕ Ανενεργό", key=f"pt_toggle_{_rid}"):
                    _admin_update("partners", _rid, {"active": not _active}); st.rerun()
                if st.button("🗑️", key=f"pt_del_{_rid}"):
                    _admin_delete("partners", _rid); st.rerun()


def _admin_articles_tab():
    st.markdown("#### 📰 Άρθρα / Blog")
    st.caption("Άρθρα από ιατρούς ή επιστημονικά περιοδικά — εμφανίζονται στην ενότητα 'Άρθρα' του app.")
    with st.expander("➕ Νέο Άρθρο", expanded=False):
        a_title = st.text_input("Τίτλος", key="na_title")
        c1, c2 = st.columns(2)
        with c1:
            a_author = st.text_input("Συγγραφέας", key="na_author", placeholder="π.χ. Δρ. Ιωάννου")
            a_source = st.text_input("Πηγή", key="na_source", placeholder="π.χ. Ελληνική Καρδιολογική Εταιρεία")
        with c2:
            a_lang = st.selectbox("Γλώσσα", ["el", "en"], key="na_lang")
            a_pub  = st.date_input("Ημερομηνία δημοσίευσης", key="na_pub", value=datetime.now())
        a_summary = st.text_area("Σύνοψη", key="na_summary", height=70)
        a_body = st.text_area("Πλήρες κείμενο (προαιρετικό — αν λείπει, δείχνει μόνο σύνοψη + link)", key="na_body", height=140)
        c3, c4 = st.columns(2)
        with c3:
            a_url = st.text_input("Εξωτερικό link (προαιρετικό)", key="na_url", placeholder="https://...")
        with c4:
            a_img = st.text_input("Εικόνα URL (προαιρετικό)", key="na_img", placeholder="https://...")
        if st.button("Δημοσίευση", type="primary", key="na_add"):
            if a_title and a_summary:
                ok = _admin_insert("articles", {
                    "title": a_title, "author": a_author, "source": a_source,
                    "summary": a_summary, "body": a_body, "url": a_url,
                    "image_url": a_img, "lang": a_lang,
                    "published_at": a_pub.isoformat(), "active": True,
                })
                if ok:
                    st.success(f"Δημοσιεύτηκε: {a_title}"); st.rerun()
                else:
                    st.error("Αποτυχία — έλεγξε ότι υπάρχει ο πίνακας 'articles' στη Supabase.")
            else:
                st.warning("Τίτλος και σύνοψη είναι απαραίτητα.")

    articles = _admin_list("articles", order_col="published_at")
    if not articles:
        st.info("Δεν υπάρχουν άρθρα ακόμα.")
    for row in articles:
        with st.container(border=True):
            c1, c2 = st.columns([5,1])
            with c1:
                _flag = "🇬🇷" if row.get("lang") == "el" else "🇬🇧"
                st.markdown(f"**{_flag} {row.get('title','—')}**")
                _meta = " · ".join(x for x in [row.get("author",""), row.get("source",""), str(row.get("published_at",""))] if x)
                if _meta:
                    st.caption(_meta)
                st.caption((row.get("summary") or "")[:160])
            with c2:
                _active = row.get("active", True)
                _rid = row.get("id", "")
                if st.button("✓ Live" if _active else "✕ Κρυφό", key=f"ar_toggle_{_rid}"):
                    _admin_update("articles", _rid, {"active": not _active}); st.rerun()
                if st.button("🗑️", key=f"ar_del_{_rid}"):
                    _admin_delete("articles", _rid); st.rerun()


def _admin_campaigns_tab():
    st.markdown("#### 📢 Banners / Campaigns")
    st.caption(
        "Banner ή pop-up διαφημίσεις από νοσοκομεία — π.χ. για παγκόσμιες ημέρες, "
        "καλοκαιρινό check-up, παιδιατρικό έλεγχο πριν τα σχολεία. Εμφανίζονται "
        "αυτόματα μόνο μέσα στο εύρος ημερομηνιών που ορίζεις."
    )
    with st.expander("➕ Νέο Campaign", expanded=False):
        c_title = st.text_input("Τίτλος", key="nc_title", placeholder="π.χ. Καλοκαιρινός Έλεγχος Σπίλων")
        c1, c2 = st.columns(2)
        with c1:
            c_img = st.text_input("Εικόνα URL", key="nc_img", placeholder="https://...")
            c_placement = st.selectbox("Τοποθέτηση", ["banner", "popup"], key="nc_placement")
        with c2:
            c_link = st.text_input("Link (όταν πατηθεί)", key="nc_link", placeholder="https://...")
        c3, c4 = st.columns(2)
        with c3:
            c_start = st.date_input("Από", key="nc_start", value=datetime.now())
        with c4:
            c_end = st.date_input("Έως", key="nc_end", value=datetime.now())
        if st.button("Δημιουργία", type="primary", key="nc_add"):
            if c_title:
                ok = _admin_insert("campaigns", {
                    "title": c_title, "image_url": c_img, "link_url": c_link,
                    "placement": c_placement,
                    "starts_on": c_start.isoformat(), "ends_on": c_end.isoformat(),
                    "active": True,
                })
                if ok:
                    st.success(f"Δημιουργήθηκε: {c_title}"); st.rerun()
                else:
                    st.error("Αποτυχία — έλεγξε ότι υπάρχει ο πίνακας 'campaigns' στη Supabase.")
            else:
                st.warning("Ο τίτλος είναι απαραίτητος.")

    campaigns = _admin_list("campaigns", order_col="starts_on")
    if not campaigns:
        st.info("Δεν υπάρχουν campaigns ακόμα.")
    _today = datetime.now().date().isoformat()
    for row in campaigns:
        with st.container(border=True):
            c1, c2 = st.columns([5,1])
            with c1:
                _live = row.get("active", True) and row.get("starts_on","") <= _today <= row.get("ends_on","9999-12-31")
                _badge = "🟢 LIVE" if _live else "⚪ Ανενεργό/Εκτός εύρους"
                st.markdown(f"**{row.get('title','—')}** — {_badge}")
                st.caption(f"📅 {row.get('starts_on','—')} → {row.get('ends_on','—')} · {row.get('placement','banner')}")
                if row.get("image_url"):
                    st.caption(f"🖼️ {row['image_url']}")
            with c2:
                _active = row.get("active", True)
                _rid = row.get("id", "")
                if st.button("✓ Ενεργό" if _active else "✕ Ανενεργό", key=f"cp_toggle_{_rid}"):
                    _admin_update("campaigns", _rid, {"active": not _active}); st.rerun()
                if st.button("🗑️", key=f"cp_del_{_rid}"):
                    _admin_delete("campaigns", _rid); st.rerun()


def render_admin_panel():
    """Full admin dashboard — video config, B2B partners, blog articles, ad
    campaigns. Reached via ?admin=1, gated by render_admin_gate()."""
    st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:18px">
  <div style="font-size:22px;font-weight:800;color:#1A1A2E">🔐 Asklepios Admin</div>
</div>
""", unsafe_allow_html=True)
    if not _supabase_client():
        st.warning(
            "⚠️ Η Supabase δεν είναι ρυθμισμένη (SUPABASE_URL / SUPABASE_ANON_KEY) — "
            "οι αλλαγές εδώ δεν θα αποθηκευτούν."
        )
    tabs = st.tabs(["🎬 Video", "🏥 Partners", "📰 Άρθρα", "📢 Campaigns"])
    with tabs[0]: _admin_video_tab()
    with tabs[1]: _admin_partners_tab()
    with tabs[2]: _admin_articles_tab()
    with tabs[3]: _admin_campaigns_tab()
    st.divider()
    if st.button("🔒 Κλείδωμα / Έξοδος"):
        st.session_state["_admin_unlocked"] = False
        st.rerun()


def save_feedback(rating, comment=""):
    """Store a minimal, non-medical feedback row in Supabase. No report/identifiers."""
    sb = _supabase_client()
    if not sb:
        return False  # demo mode: nothing stored
    try:
        sb.table("feedback").insert({
            "user_email": st.session_state.get("auth_user", ""),
            "rating": rating,
            "comment": (comment or "")[:1000],
            "lang": st.session_state.lang,
        }).execute()
        return True
    except Exception:
        return False

# ── ADMIN CONTENT STORAGE ──────────────────────────────────────────────────────
# Everything the admin panel manages (intro video URL, partner hospitals,
# blog articles, ad campaigns) lives in a small set of Supabase tables. If
# Supabase isn't configured, these all fail soft (empty list / False) —
# same "demo mode" pattern as save_feedback above. Expected schema (create
# these tables once in the Supabase SQL editor):
#
#   app_settings(key text primary key, value jsonb, updated_at timestamptz)
#   partners(id uuid default gen_random_uuid() primary key, name text,
#            specialty text, city text, phone text, email text, website text,
#            logo_url text, notes text, active boolean default true,
#            created_at timestamptz default now())
#   articles(id uuid default gen_random_uuid() primary key, title text,
#            author text, source text, summary text, body text, url text,
#            image_url text, published_at date, lang text default 'el',
#            active boolean default true, created_at timestamptz default now())
#   campaigns(id uuid default gen_random_uuid() primary key, title text,
#             image_url text, link_url text, placement text default 'banner',
#             starts_on date, ends_on date, active boolean default true,
#             created_at timestamptz default now())

def get_setting(key, default=None):
    """Read one key from app_settings. Returns `default` on any failure
    (table missing, Supabase not configured, key not found)."""
    sb = _supabase_client()
    if not sb:
        return default
    try:
        res = sb.table("app_settings").select("value").eq("key", key).limit(1).execute()
        rows = res.data or []
        if rows:
            return rows[0].get("value", default)
    except Exception:
        pass
    return default

def set_setting(key, value):
    """Upsert one key into app_settings. Returns True/False."""
    sb = _supabase_client()
    if not sb:
        return False
    try:
        sb.table("app_settings").upsert({"key": key, "value": value}).execute()
        return True
    except Exception:
        return False

def _admin_list(table, order_col="created_at", ascending=False):
    """Generic 'fetch all rows' for the admin-managed tables. Returns []
    on any failure instead of raising — admin pages should degrade to an
    empty list (with an inline error already shown by the caller) rather
    than crash the whole panel."""
    sb = _supabase_client()
    if not sb:
        return []
    try:
        q = sb.table(table).select("*").order(order_col, desc=not ascending)
        return (q.execute().data) or []
    except Exception:
        return []

def _admin_insert(table, row):
    sb = _supabase_client()
    if not sb:
        return False
    try:
        sb.table(table).insert(row).execute()
        return True
    except Exception:
        return False

def _admin_update(table, row_id, patch):
    sb = _supabase_client()
    if not sb:
        return False
    try:
        sb.table(table).update(patch).eq("id", row_id).execute()
        return True
    except Exception:
        return False

def _admin_delete(table, row_id):
    sb = _supabase_client()
    if not sb:
        return False
    try:
        sb.table(table).delete().eq("id", row_id).execute()
        return True
    except Exception:
        return False

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

# Pillar-targeted PubMed query: scopes results to *high-evidence* publication
# types (Practice Guideline, Systematic Review, Meta-Analysis, Review) crossed
# with the MeSH heading that matches the pillar — so each recommendation gets
# 1-2 references from guideline-quality literature rather than single studies.
_PILLAR_MESH = {
    "exercise":  '("Exercise Therapy"[MeSH] OR "Exercise"[MeSH] OR "Physical Activity"[MeSH:NoExp] OR "Exercise Movement Techniques"[MeSH])',
    "nutrition": '("Diet Therapy"[MeSH] OR "Diet"[MeSH] OR "Nutrition Therapy"[MeSH] OR "Diet, Healthy"[MeSH])',
    "lifestyle": '("Life Style"[MeSH] OR "Risk Reduction Behavior"[MeSH] OR "Health Behavior"[MeSH])',
}
_PILLAR_PTYPE = '(Practice Guideline[ptyp] OR Systematic Review[ptyp] OR Meta-Analysis[ptyp] OR Review[ptyp])'

def pubmed_pillar_search(condition, pillar, n=2):
    """High-evidence PubMed search for one of: 'exercise', 'nutrition', 'lifestyle'.
    Returns the same list-of-dicts shape as pubmed_search. Falls back to a
    broader keyword query if the strict MeSH+ptyp combo returns nothing."""
    if not condition: return []
    mesh = _PILLAR_MESH.get(pillar)
    if not mesh: return []
    cond_q = condition.strip()
    # Try strict (MeSH + high-evidence ptype) first
    strict = f"{cond_q} AND {mesh} AND {_PILLAR_PTYPE}"
    res = pubmed_search(strict, n=n)
    if res:
        return res
    # Fallback: drop ptype filter — still MeSH-scoped, just any pub type
    return pubmed_search(f"{cond_q} AND {mesh}", n=n)

# ── PHYSIOTHERAPY EVIDENCE (PubMed/MEDLINE, PEDro-equivalent MeSH scope) ─────
# PEDro (pedro.org.au) itself has no public API — it is search-UI only — so we
# reuse the existing NCBI eutils pipeline (same one powering pubmed_search) but
# scope the query to physiotherapy/rehabilitation MeSH headings + high-evidence
# publication types. This mirrors how PEDro itself prioritises RCTs/systematic
# reviews/guidelines, using infrastructure we already have a key for.
_PHYSIO_MESH = ('("Physical Therapy Modalities"[MeSH] OR "Exercise Therapy"[MeSH] OR '
                '"Rehabilitation"[MeSH] OR "Musculoskeletal Manipulations"[MeSH])')

def pedro_pillar_search(condition_hint, n=3):
    """Physiotherapy-evidence search (PEDro-equivalent) via PubMed/MEDLINE.
    Returns the same list-of-dicts shape as pubmed_search: pmid/title/authors/
    journal/date/url. Falls back to a looser query if the strict combo is empty."""
    if not condition_hint:
        return []
    cond_q = condition_hint.strip()
    strict = f"{cond_q} AND {_PHYSIO_MESH} AND {_PILLAR_PTYPE}"
    res = pubmed_search(strict, n=n)
    if res:
        return res
    return pubmed_search(f"{cond_q} AND {_PHYSIO_MESH}", n=n)

# ── PSYCHOLOGY EVIDENCE (PubMed/MEDLINE — peer-reviewed, same as physio) ────
# Earlier version of this used OSF/PsyArXiv (preprints, not peer-reviewed) —
# the same quality objection raised against medRxiv applies there too, so it
# was replaced. PubMed/MEDLINE already indexes the bulk of psychology and
# psychiatry literature (it includes journals covering psychotherapy, CBT,
# anxiety/mood disorders, etc.), so this reuses the same eutils pipeline and
# the same high-evidence publication-type filter as pedro_pillar_search.
_PSYCH_MESH = ('("Psychotherapy"[MeSH] OR "Cognitive Behavioral Therapy"[MeSH] OR '
               '"Mental Health"[MeSH] OR "Anxiety Disorders"[MeSH] OR '
               '"Stress, Psychological"[MeSH] OR "Counseling"[MeSH])')

def psychology_pillar_search(condition_hint, n=3):
    """Psychology-evidence search via PubMed/MEDLINE (peer-reviewed).
    Returns the same list-of-dicts shape as pubmed_search: pmid/title/authors/
    journal/date/url. Falls back to a looser query if the strict combo is empty."""
    if not condition_hint:
        return []
    cond_q = condition_hint.strip()
    strict = f"{cond_q} AND {_PSYCH_MESH} AND {_PILLAR_PTYPE}"
    res = pubmed_search(strict, n=n)
    if res:
        return res
    return pubmed_search(f"{cond_q} AND {_PSYCH_MESH}", n=n)

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

# ── WHISPER (voice → text) ────────────────────────────────────────────────────
def whisper_transcribe(audio_bytes, filename="recording.webm", lang="el"):
    """Greek/English voice → text via OpenAI Whisper. Reuses the existing
    OPENAI_API_KEY — no new dependency. The transcribed text is shown to the
    user for review/edit BEFORE sending to chat (safety + privacy).
    
    Audio is sent to OpenAI for processing but NEVER stored on our side.
    Returns (text, error) where one of them is None.
    """
    key = get_openai_key()
    if not key:
        return None, "⚠️ OpenAI API key not set."
    try:
        import requests
        # Map common audio MIME types so Whisper recognises the format
        mime = "audio/webm"
        if filename.lower().endswith(".wav"):  mime = "audio/wav"
        elif filename.lower().endswith(".mp3"): mime = "audio/mpeg"
        elif filename.lower().endswith(".m4a"): mime = "audio/mp4"
        files = {"file": (filename, audio_bytes, mime)}
        data = {
            "model": "whisper-1",
            "language": lang if lang in ("el","en") else "el",
            "response_format": "text",
        }
        r = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {key}"},
            files=files, data=data, timeout=60,
        )
        if r.status_code == 200:
            return r.text.strip(), None
        return None, f"⚠️ Whisper {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return None, f"⚠️ {e}"

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
    "_hero_seen": False,  # hero landing shown once per session before home
    "profile": {},
    "vitals": {},
    "vitals_analysis": "",
    "triage_chat": [],
    "triage_ready": False,
    "report": "",
    "report_pubmed": [],
    "report_gpt": "",
    "report_recs": None,  # {"exercise": "...", "nutrition": "...", "lifestyle": "..."} from Claude
    "report_recs_refs": {},  # {"exercise": [...refs...], "nutrition": [...], "lifestyle": [...]}
    "report_physio_refs": [],  # PEDro-equivalent PubMed refs for the physiotherapy card
    "report_psych_refs": [],   # PubMed/MEDLINE refs for the psychology card
    "photo_findings": [],  # list of dicts — visual analyses added to assessment
    "lab_findings": [],    # list of dicts — lab PDF/image analyses added to assessment
    "_voice_widget_counter": 0,  # increments to force audio_input widget reset
    "medications": [],
    "med_inputs": [],
    "symptom_chips": [],
    "fb_rating": "",
    "fb_sent": False,
    "output_lang": None,  # AI response language; None = follow UI lang
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── TRANSLATIONS ──────────────────────────────────────────────────────────────
T = {
    "el": {
        "title": "Asklepios",
        "subtitle": "\u039f AI \u039d\u03bf\u03c3\u03b7\u03bb\u03b5\u03c5\u03c4\u03ae\u03c2 \u03c3\u03bf\u03c5",
        "tagline": "\u0388\u03b3\u03ba\u03c5\u03c1\u03b7 \u03b9\u03b1\u03c4\u03c1\u03b9\u03ba\u03ae \u03c0\u03bb\u03b7\u03c1\u03bf\u03c6\u03cc\u03c1\u03b7\u03c3\u03b7 \u00b7 \u03a0\u03ac\u03bd\u03c4\u03b1 \u03b4\u03af\u03c0\u03bb\u03b1 \u03c3\u03bf\u03c5",
        "start": "\u039e\u03b5\u03ba\u03af\u03bd\u03b1 \u0395\u03ba\u03c4\u03af\u03bc\u03b7\u03c3\u03b7",
        "disclaimer_main": "\u26a0\ufe0f \u039f Asklepios \u03c0\u03b1\u03c1\u03ad\u03c7\u03b5\u03b9 \u03c0\u03bb\u03b7\u03c1\u03bf\u03c6\u03bf\u03c1\u03af\u03b5\u03c2 \u03c5\u03b3\u03b5\u03af\u03b1\u03c2 \u03b1\u03c0\u03bf\u03ba\u03bb\u03b5\u03b9\u03c3\u03c4\u03b9\u03ba\u03ac \u03b3\u03b9\u03b1 \u03b5\u03bd\u03b7\u03bc\u03b5\u03c1\u03c9\u03c4\u03b9\u03ba\u03bf\u03cd\u03c2 \u03c3\u03ba\u03bf\u03c0\u03bf\u03cd\u03c2. \u0394\u03b5\u03bd \u03b1\u03bd\u03c4\u03b9\u03ba\u03b1\u03b8\u03b9\u03c3\u03c4\u03ac \u03b9\u03b1\u03c4\u03c1\u03b9\u03ba\u03ae \u03b4\u03b9\u03ac\u03b3\u03bd\u03c9\u03c3\u03b7 \u03ae \u03b8\u03b5\u03c1\u03b1\u03c0\u03b5\u03af\u03b1. \u03a3\u03b5 \u03b5\u03c0\u03b5\u03af\u03b3\u03bf\u03c5\u03c3\u03b1 \u03b1\u03bd\u03ac\u03b3\u03ba\u03b7 \u03ba\u03b1\u03bb\u03ad\u03c3\u03c4\u03b5 **166** (\u0395\u039a\u0391\u0392) \u03ae **112**.",
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
        "triage_sub": "\u03a0\u03b5\u03c1\u03b9\u03b3\u03c1\u03ac\u03c8\u03c4\u03b5 \u03c4\u03b1 \u03c3\u03c5\u03bc\u03c0\u03c4\u03ce\u03bc\u03b1\u03c4\u03ac \u03c3\u03b1\u03c2. \u039f Asklepios \u03b8\u03b1 \u03c3\u03b1\u03c2 \u03ba\u03ac\u03bd\u03b5\u03b9 \u03ba\u03b1\u03c4\u03b5\u03c5\u03b8\u03c5\u03bd\u03cc\u03bc\u03b5\u03bd\u03b5\u03c2 \u03b5\u03c1\u03c9\u03c4\u03ae\u03c3\u03b5\u03b9\u03c2.",
        "triage_placeholder": "\u03a0.\u03c7. \u0388\u03c7\u03c9 \u03c0\u03bf\u03bd\u03bf\u03ba\u03ad\u03c6\u03b1\u03bb\u03bf \u03c4\u03c1\u03b9\u03ce\u03bd \u03b7\u03bc\u03b5\u03c1\u03ce\u03bd \u03bc\u03b5 \u03bd\u03b1\u03c5\u03c4\u03af\u03b1...",
        "generate_report": "\u0394\u03b7\u03bc\u03b9\u03bf\u03c5\u03c1\u03b3\u03af\u03b1 \u03a0\u03bb\u03ae\u03c1\u03bf\u03c5\u03c2 \u0391\u03bd\u03b1\u03c6\u03bf\u03c1\u03ac\u03c2",
        "report_title": "\u039b\u03b5\u03c0\u03c4\u03bf\u03bc\u03b5\u03c1\u03ae\u03c2 \u0395\u03ba\u03c4\u03af\u03bc\u03b7\u03c3\u03b7 \u03a5\u03b3\u03b5\u03af\u03b1\u03c2",
        "second_opinion": "\u0394\u03b5\u03cd\u03c4\u03b5\u03c1\u03b7 \u0393\u03bd\u03ce\u03bc\u03b7 GPT-4o",
        "pubmed": "\u0395\u03c0\u03b9\u03c3\u03c4\u03b7\u03bc\u03bf\u03bd\u03b9\u03ba\u03ad\u03c2 \u0391\u03bd\u03b1\u03c6\u03bf\u03c1\u03ad\u03c2 PubMed",
        "skip_vitals": "\u03a0\u03b1\u03c1\u03ac\u03bb\u03b5\u03b9\u03c8\u03b7 (\u03c7\u03c9\u03c1\u03af\u03c2 \u03bc\u03b5\u03c4\u03c1\u03ae\u03c3\u03b5\u03b9\u03c2)",
    },
    "en": {
        "title": "Asklepios",
        "subtitle": "Your AI Nurse",
        "tagline": "Evidence-based health guidance · Always by your side",
        "start": "Start Assessment",
        "disclaimer_main": "\u26a0\ufe0f Asklepios provides health information for informational purposes only. It does not replace medical diagnosis or treatment. In an emergency call **166** (EKAB) or **112**.",
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
        "triage_sub": "Describe your symptoms. Asklepios will ask targeted follow-up questions.",
        "triage_placeholder": "E.g. I have had a headache for three days with nausea...",
        "generate_report": "Generate Full Clinical Report",
        "report_title": "Detailed Health Assessment",
        "second_opinion": "GPT-4o Second Opinion",
        "pubmed": "PubMed Evidence",
        "skip_vitals": "Skip (no measurements)",
    }
}

def t(key): return T[st.session_state.lang].get(key, key)


def render_topbar():
    """Top-right bar visible on every post-login screen: language toggle + logout.
    Centralises both so each screen does not duplicate them."""
    lang = st.session_state.lang
    _t1, _t2, _t3 = st.columns([7, 1, 1])
    with _t2:
        if st.button(("🇬🇧 EN" if lang=="el" else "🇬🇷 ΕΛ"),
                     key="topbar_lang", use_container_width=True):
            st.session_state.lang = "en" if lang=="el" else "el"
            st.rerun()
    with _t3:
        if is_logged_in():
            if st.button("🚪 " + ("Έξοδος" if lang=="el" else "Logout"),
                         key="topbar_logout", use_container_width=True):
                logout()
                st.rerun()


def render_doc_header(title_el, title_en, *, icon="📋",
                      sub_el=None, sub_en=None, show_date=True):
    """Compact doc-template style header card for each screen.
    White card with blue circular logo, org caps, friendly title, optional subtitle
    and date. Establishes the medical-form aesthetic on intake/vitals/triage/report
    while keeping Streamlit widgets unchanged below."""
    lang = st.session_state.lang
    title = title_el if lang == "el" else title_en
    sub = (sub_el if lang == "el" else sub_en) or ""
    org = "ASKLEPIOS · AI ΝΟΣΗΛΕΥΤΗΣ" if lang == "el" else "ASKLEPIOS · AI NURSE"
    date_str = datetime.now().strftime("%d.%m.%Y")
    date_lbl = "ΗΜΕΡ." if lang == "el" else "DATE"
    date_html = (
        f'<div class="dph-date"><div class="dph-date-lbl">{date_lbl}</div>'
        f'<div class="dph-date-val">{date_str}</div></div>'
    ) if show_date else ""
    sub_html = f'<div class="dph-sub">{sub}</div>' if sub else ""
    st.markdown(
        f"""
<style>
.doc-page-head {{
  display: flex; align-items: center; gap: 16px;
  padding: 18px 22px;
  background: white;
  border: 1px solid #E5E7EB;
  border-radius: 14px;
  margin: 4px 0 22px;
  font-family: 'Inter', system-ui, sans-serif;
  box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}}
.dph-logo {{
  width: 50px; height: 50px; border-radius: 50%;
  background: #DBEAFE;
  display: flex; align-items: center; justify-content: center;
  font-size: 23px; flex-shrink: 0;
}}
.dph-text {{ flex: 1; min-width: 0; }}
.dph-org {{
  font-size: 9.5px; font-weight: 700; letter-spacing: 0.14em;
  color: #6B7280; text-transform: uppercase; margin-bottom: 3px;
}}
.dph-title {{
  font-size: 19px; font-weight: 700; color: #111827;
  letter-spacing: -0.015em; line-height: 1.2;
}}
.dph-sub {{
  font-size: 12.5px; color: #6B7280; margin-top: 3px; font-weight: 500;
}}
.dph-date {{
  text-align: right; flex-shrink: 0;
  border-left: 1px solid #E5E7EB; padding-left: 14px;
}}
.dph-date-lbl {{
  font-size: 9px; font-weight: 700; letter-spacing: 0.14em;
  color: #9CA3AF; text-transform: uppercase;
}}
.dph-date-val {{
  font-size: 13px; font-weight: 700; color: #111827;
  font-variant-numeric: tabular-nums; margin-top: 2px;
}}
@media (max-width: 640px) {{
  .doc-page-head {{ padding: 14px 16px; gap: 12px; }}
  .dph-logo {{ width: 42px; height: 42px; font-size: 19px; }}
  .dph-title {{ font-size: 16px; }}
  .dph-sub {{ font-size: 11.5px; }}
  .dph-date {{ display: none; }}
}}
</style>
<div class="doc-page-head">
  <div class="dph-logo">{icon}</div>
  <div class="dph-text">
    <div class="dph-org">{org}</div>
    <div class="dph-title">{title}</div>
    {sub_html}
  </div>
  {date_html}
</div>
""",
        unsafe_allow_html=True,
    )


def render_bottom_nav():
    """Persistent bottom tab bar: Αρχική / Ζωτικά / Συμπτώματα / Ιστορικό.
    Lets the person jump between sections instead of being locked into the
    linear intake→vitals→triage→report flow. The active assessment's own
    progress (render_stepper) keeps showing inside intake/vitals/triage/report
    — this nav is the higher-level "where in the app am I" layer, not a
    replacement for it.

    Tapping a tab that needs a profile (Ζωτικά/Συμπτώματα/Ιστορικό) while no
    profile exists yet sends the person to intake first, rather than showing
    a vitals/triage screen with no name attached to it.
    """
    lang = st.session_state.lang
    has_profile = bool(st.session_state.profile.get("name"))
    cur = st.session_state.screen

    # Which nav item is "active" for the current screen. intake/vitals both
    # light up "Ζωτικά" is wrong — intake maps to no tab being forced active
    # other than by section: intake counts toward the assessment, so we treat
    # it as part of the "Συμπτώματα" entry point conceptually, but visually
    # it's clearer to highlight nothing extra: home/vitals/triage/report each
    # map 1:1 to a tab; intake highlights the same tab as wherever it leads.
    tab_for_screen = {
        "home": "home", "intake": "triage", "vitals": "vitals",
        "triage": "triage", "report": "history",
    }
    active_tab = tab_for_screen.get(cur, "home")

    items = [
        ("home",    "🏠", "Αρχική"      if lang=="el" else "Home"),
        ("vitals",  "❤️", "Ζωτικά"      if lang=="el" else "Vitals"),
        ("triage",  "💬", "Συμπτώματα"  if lang=="el" else "Symptoms"),
        ("history", "📋", "Ιστορικό"    if lang=="el" else "History"),
    ]

    st.markdown("""
<style>
.bottom-nav-spacer { height: 76px; }  /* keeps page content from hiding under the fixed bar */

/* .bn-marker is rendered INSIDE the first column below, making it a real
   descendant of the st.columns() stHorizontalBlock. :has() then lets us pin
   THAT ancestor block to the bottom of the viewport. st.columns() has no
   built-in way to opt into fixed positioning — this is the reliable way to
   do it without a custom component. */
div[data-testid="stHorizontalBlock"]:has(.bn-marker) {
  position: fixed; left: 0; right: 0; bottom: 0; z-index: 999;
  background: white; border-top: 1px solid #EEF2FA;
  padding: 8px 6px calc(8px + env(safe-area-inset-bottom));
  box-shadow: 0 -2px 12px rgba(15,42,82,0.05);
  max-width: 480px; margin: 0 auto;
  flex-wrap: nowrap !important;
  justify-content: space-between !important;
  gap: 2px !important;
}
div[data-testid="stHorizontalBlock"]:has(.bn-marker) > div[data-testid="stColumn"] {
  min-width: 0 !important; width: 25% !important; flex: 0 0 25% !important;
}
.bn-marker { display: none; }
div[data-testid="stHorizontalBlock"]:has(.bn-marker) button {
  background: transparent !important; border: none !important; box-shadow: none !important;
  color: #B8C2D6 !important; font-weight: 700 !important; font-size: 9.5px !important;
  line-height: 1.4 !important; padding: 4px 2px !important; min-height: 0 !important;
  white-space: nowrap !important; width: 100% !important;
  display: flex !important; align-items: center !important; justify-content: center !important;
  text-align: center !important;
}
div[data-testid="stHorizontalBlock"]:has(.bn-marker) button p {
  text-align: center !important; width: 100%;
}
div[data-testid="stHorizontalBlock"]:has(.bn-marker) button[kind="primary"] {
  color: #2D6FE0 !important;
}
div[data-testid="stHorizontalBlock"]:has(.bn-marker) button[kind="primary"] p {
  color: #2D6FE0 !important; font-weight: 800 !important;
}
</style>
""", unsafe_allow_html=True)

    cols = st.columns(len(items))
    for i, (col, (key, icon, label)) in enumerate(zip(cols, items)):
        with col:
            if i == 0:
                st.markdown('<div class="bn-marker"></div>', unsafe_allow_html=True)
            is_active = (key == active_tab)
            if st.button(f"{icon}  {label}", key=f"bn_{key}",
                         use_container_width=True,
                         type=("primary" if is_active else "secondary")):
                if key != "home" and not has_profile:
                    st.session_state.screen = "intake"
                else:
                    st.session_state.screen = key
                st.rerun()


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

def classify_vitals(v, age=None):
    """Classify vitals as green/yellow/red. Age-aware: pediatric ranges differ
    significantly from adult, especially HR and BR.
    
    Reference ranges (PALS/AHA + WHO):
      Infant <1y:   HR 100-160, BR 30-60, SBP >70+(age×2)
      Toddler 1-3:  HR 90-150,  BR 24-40
      Preschool 3-5: HR 80-140, BR 22-34
      School 6-12:  HR 70-120,  BR 18-30
      Adolescent 13-17: HR 60-100, BR 12-20
      Adult 18+:    HR 60-100,  BR 12-20
    
    Temp, SpO2, BMI are essentially the same across ages (pediatric BMI uses
    percentiles, but our coarse green/yellow/red still gives useful signal).
    """
    status = {}
    a = age if age is not None else 99  # treat unknown as adult
    
    # Heart rate — age-stratified
    hr = v.get("hr")
    if hr:
        if a < 1:
            g_lo, g_hi, y_lo, y_hi = 100, 160, 90, 180
        elif a < 3:
            g_lo, g_hi, y_lo, y_hi = 90, 150, 80, 170
        elif a < 6:
            g_lo, g_hi, y_lo, y_hi = 80, 140, 70, 160
        elif a < 13:
            g_lo, g_hi, y_lo, y_hi = 70, 120, 60, 140
        else:
            g_lo, g_hi, y_lo, y_hi = 60, 100, 50, 110
        if g_lo <= hr <= g_hi:  status["hr"] = "green"
        elif y_lo <= hr <= y_hi: status["hr"] = "yellow"
        else:                   status["hr"] = "red"
    
    # Blood pressure
    sys_ = v.get("bp_sys"); dia = v.get("bp_dia")
    if sys_ and dia:
        if a < 13:
            # Pediatric: rough rule "70 + 2×age" for hypotension threshold,
            # pediatric hypertension >95th percentile (~ 1.2× normal). Use
            # coarse ranges — recommend physician for accurate pediatric BP.
            expected_sys = 90 + (a * 2) if a >= 1 else 70 + (a * 2)
            if sys_ < expected_sys - 15 or sys_ > expected_sys + 25:
                status["bp"] = "red"
            elif sys_ < expected_sys - 5 or sys_ > expected_sys + 15:
                status["bp"] = "yellow"
            else:
                status["bp"] = "green"
        else:
            if sys_ < 120 and dia < 80:        status["bp"] = "green"
            elif sys_ < 130:                   status["bp"] = "yellow"
            elif sys_ < 140 or dia < 90:       status["bp"] = "yellow"
            else:                              status["bp"] = "red"
    
    # Breathing rate — age-stratified
    br = v.get("br")
    if br:
        if a < 1:
            g_lo, g_hi, y_lo, y_hi = 30, 60, 24, 70
        elif a < 3:
            g_lo, g_hi, y_lo, y_hi = 24, 40, 20, 50
        elif a < 6:
            g_lo, g_hi, y_lo, y_hi = 22, 34, 18, 40
        elif a < 13:
            g_lo, g_hi, y_lo, y_hi = 18, 30, 14, 36
        else:
            g_lo, g_hi, y_lo, y_hi = 12, 20, 10, 24
        if g_lo <= br <= g_hi:   status["br"] = "green"
        elif y_lo <= br <= y_hi: status["br"] = "yellow"
        else:                    status["br"] = "red"
    
    # SpO2 — same across ages
    spo2 = v.get("spo2")
    if spo2:
        if spo2 >= 95:   status["spo2"] = "green"
        elif spo2 >= 90: status["spo2"] = "yellow"
        else:            status["spo2"] = "red"
    
    # Temperature — same
    temp = v.get("temp")
    if temp:
        if 36.1 <= temp <= 37.2:  status["temp"] = "green"
        elif 37.3 <= temp <= 38.0: status["temp"] = "yellow"
        else:                      status["temp"] = "red"
    
    # BMI — only for adults (pediatric BMI requires percentile charts)
    w = v.get("weight"); h = v.get("height")
    if w and h:
        bmi = w / ((h/100)**2); v["bmi"] = round(bmi, 1)
        if a >= 18:
            if 18.5 <= bmi <= 24.9:  status["bmi"] = "green"
            elif 25 <= bmi <= 29.9:  status["bmi"] = "yellow"
            else:                    status["bmi"] = "red"
        # For pediatric, we don't classify — the value is recorded but no
        # green/yellow/red without percentile data
    
    return status

def demographic_bp_risk(age, bmi, hr, weight=None, height=None):
    """
    Evidence-based BP risk classification using demographic features.
    Based on: Chowdhury et al. (2020) - top ReliefF features for BP estimation.
    Returns: dict with risk_level, sbp_range, dbp_range, explanation
    """
    score = 0
    factors = []

    # Age — strongest demographic predictor (Feature #105 in paper)
    if age >= 70:   score += 4; factors.append("age ≥70" if True else "")
    elif age >= 60: score += 3; factors.append("age 60-69")
    elif age >= 50: score += 2; factors.append("age 50-59")
    elif age >= 40: score += 1; factors.append("age 40-49")

    # BMI — second strongest (Feature #107)
    if bmi:
        if bmi >= 35:   score += 3; factors.append("BMI ≥35 (obese II)")
        elif bmi >= 30: score += 2; factors.append("BMI 30-34 (obese I)")
        elif bmi >= 25: score += 1; factors.append("BMI 25-29 (overweight)")

    # Heart Rate — Feature #106
    if hr:
        if hr > 90:   score += 2; factors.append("elevated HR")
        elif hr > 80: score += 1; factors.append("high-normal HR")
        elif hr < 55: score -= 1; factors.append("low HR (fit/athletic)")

    # Weight/Height ratio proxy if BMI not computed yet
    if weight and height and not bmi:
        bmi_calc = weight / ((height/100)**2)
        if bmi_calc >= 30: score += 2
        elif bmi_calc >= 25: score += 1

    # Map score to risk level + estimated range
    if score <= 0:
        return {"level":"optimal","color":"#10B981","label_el":"Βέλτιστη","label_en":"Optimal",
                "sbp":"<115","dbp":"<75","note_el":"Εξαιρετικό καρδιαγγειακό προφίλ.","note_en":"Excellent cardiovascular profile.","score":score}
    elif score <= 2:
        return {"level":"normal","color":"#10B981","label_el":"Φυσιολογική","label_en":"Normal",
                "sbp":"115-129","dbp":"75-84","note_el":"Φυσιολογικά επίπεδα για το προφίλ σας.","note_en":"Normal levels for your profile.","score":score}
    elif score <= 4:
        return {"level":"elevated","color":"#F59E0B","label_el":"Ελαφρά Αυξημένη","label_en":"Elevated Risk",
                "sbp":"130-144","dbp":"85-89","note_el":"Σχετικά αυξημένος κίνδυνος. Μέτρηση πίεσης συνιστάται.","note_en":"Moderately elevated risk. BP measurement advised.","score":score}
    elif score <= 6:
        return {"level":"high","color":"#EF4444","label_el":"Υψηλός Κίνδυνος","label_en":"High Risk",
                "sbp":"140-159","dbp":"90-99","note_el":"Αυξημένος κίνδυνος υπέρτασης. Επισκεφθείτε γιατρό.","note_en":"Elevated hypertension risk. See a doctor.","score":score}
    else:
        return {"level":"very_high","color":"#DC2626","label_el":"Πολύ Υψηλός Κίνδυνος","label_en":"Very High Risk",
                "sbp":"≥160","dbp":"≥100","note_el":"Πολύ υψηλός κίνδυνος. Απαιτείται ιατρική αξιολόγηση.","note_en":"Very high risk. Medical evaluation required.","score":score}


KIRA_SYSTEM_EL = """Είσαι ο Asklepios — AI νοσηλευτής για Έλληνες χρήστες. Είσαι κλινικά ακριβής, άμεσος και υποστηρικτικός.
Ρόλος: Τριάζ συμπτωμάτων (μία ερώτηση κάθε φορά), ερμηνεία ζωτικών, φάρμακα, ελληνικό σύστημα υγείας (ΕΟΠΥΥ, ΕΟΔΥ, ΕΟΦ).
Φωτογραφία: Αν το σύμπτωμα είναι οπτικό (δέρμα/εξάνθημα, μάτι, τραύμα/πληγή, στόμα/λαιμός, νύχια, ορατή αλλοίωση), αφού κάνεις την αρχική σου εκτίμηση πρότεινε στον χρήστη να ανεβάσει φωτογραφία από την επιλογή «📷 Ανάλυση φωτογραφίας» πιο κάτω, για πιο ακριβή εκτίμηση. Για μη-οπτικά συμπτώματα (π.χ. πονοκέφαλος, ζάλη) ΜΗΝ ζητάς φωτογραφία. Η φωτογραφία είναι ΠΡΟΑΙΡΕΤΙΚΗ: αν ο χρήστης δεν ανεβάσει ή δεν θέλει, ΣΥΝΕΧΙΣΕ κανονικά την εκτίμηση χωρίς να σταματάς, να περιμένεις ή να επιμένεις.
Κανόνες: Πάντα συστήνεις επαγγελματία. Κόκκινες σημαίες → 166/112. Όταν έχεις αρκετά: "Έχω αρκετά στοιχεία — μπορούμε να δημιουργήσουμε πλήρη αναφορά." Μία ερώτηση κάθε φορά.
Ζωτικά: Αν τα συμπτώματα είναι καρδιακά/αυτόνομα (αίσθημα παλμών, ταχυπαλμία, πόνος/σφίξιμο στο στήθος, δύσπνοια, ζάλη, λιποθυμία, κρύος ιδρώτας/εφίδρωση), πρότεινε ήπια στον χρήστη να μετρήσει ζωτικά (καρδιακός ρυθμός/πίεση) — ΠΡΟΑΙΡΕΤΙΚΟ, συνέχισε κανονικά αν δεν το κάνει.
Triage — κανόνας NON-EMERGENCY: Σύστηνε επίσκεψη σε γιατρό (όχι self-care) όταν ισχύει ΟΠΟΙΟΔΗΠΟΤΕ από τα παρακάτω:
  • Χρειάζεται συνταγογραφούμενο φάρμακο (αντιβιοτικό, steroid, antifungal κλπ.)
  • Χρειάζεται imaging για αποκλεισμό κατάγματος/σοβαρής βλάβης (π.χ. στρέψιμο αστραγάλου, οξύς πόνος πλάτης)
  • Η διάγνωση χρειάζεται επιβεβαίωση από επαγγελματία πριν από θεραπεία (π.χ. κολπική καντιντίαση για πρώτη φορά, δερματίτιδα με άγνωστο αίτιο)
  • Υπάρχει κίνδυνος επιδείνωσης χωρίς παρακολούθηση (π.χ. γαστρεντερίτιδα με κίνδυνο αφυδάτωσης, ημικρανία με νέα χαρακτηριστικά)
  • Τα συμπτώματα επιμένουν > 48-72 ώρες χωρίς βελτίωση"""

KIRA_SYSTEM_EN = """You are Asklepios — an AI nurse for users in Greece. Clinically accurate, direct, supportive.
Role: Symptom triage (one question at a time), vitals interpretation, medications, Greek health system (EOPYY, EODY, EOF).
Photo: If the symptom is visual (skin/rash, eye, wound, mouth/throat, nails, any visible lesion), after giving your initial assessment, invite the user to upload a photo via the "📷 Photo analysis" option below for a more accurate assessment. For non-visual symptoms (e.g. headache, dizziness) do NOT ask for a photo. The photo is OPTIONAL: if the user doesn't upload one or declines, CONTINUE the assessment normally — do not stop, wait, or insist.
Rules: Always recommend a professional. Red flags → 166/112. When ready: "I have enough information — we can generate a full clinical report." One question at a time.
Vitals: If the symptoms are cardiac/autonomic (palpitations, racing heart, chest pain/tightness, shortness of breath, dizziness, fainting, cold sweat/sweating), gently suggest the user measure vitals (heart rate/blood pressure) — OPTIONAL, continue normally if they don't.
Triage — NON-EMERGENCY rule: Recommend seeing a doctor (not self-care) when ANY of the following apply:
  • A prescription medication is needed (antibiotic, steroid, antifungal, etc.)
  • Imaging is needed to rule out fracture or serious injury (e.g. ankle sprain, acute back pain)
  • Diagnosis requires professional confirmation before treatment (e.g. first-time vaginal candidiasis, unidentified dermatitis)
  • There is risk of deterioration without monitoring (e.g. gastroenteritis with dehydration risk, migraine with new features)
  • Symptoms persist > 48-72 hours without improvement"""

def kira_system(): return KIRA_SYSTEM_EL if st.session_state.lang=="el" else KIRA_SYSTEM_EN

# ── MULTILINGUAL AI OUTPUT ─────────────────────────────────────────────────────
# Decoupled from the UI language so Bulgarian/Romanian/Albanian etc. speakers
# can receive the report and chat in their native language while the UI stays el/en.
OUTPUT_LANGUAGES = {
    "el": ("🇬🇷 Ελληνικά",  "Greek (Ελληνικά)"),
    "en": ("🇬🇧 English",   "English"),
    "bg": ("🇧🇬 Български", "Bulgarian (Български)"),
    "ro": ("🇷🇴 Română",    "Romanian (Română)"),
    "al": ("🇦🇱 Shqip",     "Albanian (Shqip)"),
    "ru": ("🇷🇺 Русский",   "Russian (Русский)"),
    "de": ("🇩🇪 Deutsch",   "German (Deutsch)"),
    "fr": ("🇫🇷 Français",  "French (Français)"),
}

def output_lang_code():
    """Effective AI-output language. Falls back to UI language if not set."""
    code = st.session_state.get("output_lang")
    if code and code in OUTPUT_LANGUAGES:
        return code
    return st.session_state.get("lang", "el")

def output_language_directive():
    """Append to any Claude prompt to force response into the chosen output language.
    Returns empty string when output lang matches UI lang (no override needed)."""
    code = output_lang_code()
    if code == st.session_state.get("lang", "el"):
        return ""
    name = OUTPUT_LANGUAGES[code][1]
    return (
        f"\n\nOUTPUT LANGUAGE OVERRIDE: Respond ONLY in {name}. "
        "This overrides any earlier language instruction. "
        "Use correct clinical terminology in the target language. "
        "Do NOT mix languages within a sentence."
    )

def render_output_language_picker(lang, *, key_suffix=""):
    """Compact dropdown to pick the AI-output language."""
    label = ("🌍 Γλώσσα αναφοράς & AI απαντήσεων" if lang == "el"
             else "🌍 Report & AI response language")
    codes   = list(OUTPUT_LANGUAGES.keys())
    current = output_lang_code()
    try:    idx = codes.index(current)
    except: idx = 0
    choice = st.selectbox(
        label, codes, index=idx,
        format_func=lambda c: OUTPUT_LANGUAGES[c][0],
        key=f"output_lang_picker_{key_suffix}",
        help=("Το UI παραμένει στα ελληνικά/αγγλικά. Η αναφορά και η συνομιλία AI θα εμφανίζονται στη γλώσσα που επιλέγεις."
              if lang=="el" else
              "The UI stays in Greek/English. The report and AI chat will appear in the selected language."),
    )
    if choice != current:
        st.session_state["output_lang"] = choice
        st.rerun()
def _strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s.lower())
                   if unicodedata.category(c) != "Mn")
# Each category maps symptom roots → the vital that helps. "scan"=True only where
# the camera face-scan can actually produce the value (heart rate → cardiac only).
_VITAL_CATEGORIES = [
    {"key":"cardio","scan":True,
     "el":"καρδιακός ρυθμός / πίεση","en":"heart rate / blood pressure",
     # Cardiac symptoms only. We deliberately do NOT use generic "ιδρωτ/ιδρωσ/
     # ιδρων/εφιδρ" (sweating) — those false-positive on workout/fever/photo
     # descriptions. Cold sweat ("κρυος ιδρωτ" / "cold sweat") is cardiac and
     # is checked specifically below.
     "roots":["παλμ","ταχυκαρδ","αρρυθμ","στηθ","θωρακ","λιποθυμ","λιγοθυμ",
              "κρυος ιδρωτ","ζαλ",
              "palpit","racing heart","irregular heart","tachycard","arrhythm",
              "chest pain","chest tightness","faint","cold sweat","dizz","lightheaded","light-headed"]},
    {"key":"bp","scan":False,
     "el":"αρτηριακή πίεση","en":"blood pressure",
     "roots":["πιεση","υπερτασ","υποτασ","αρτηριακ",
              "blood pressure","hypertens","hypotens"]},
    {"key":"temp","scan":False,
     "el":"θερμοκρασία","en":"temperature",
     "roots":["πυρετ","θερμοκρασ","δεκατ","εμπυρετ","ριγος","ριγη","κρυαδ",
              "fever","febrile","chills","temperature","high temp"]},
    {"key":"resp","scan":False,
     "el":"οξυγόνο (SpO₂) & αναπνοές","en":"oxygen (SpO₂) & breathing",
     "roots":["δυσπν","βηχ","ασθμ","πνευμον","αναπν","λαχαν","συριγμ","βρογχ","κορον","covid",
              "cough","wheez","asthma","pneumonia","breathless","short of breath",
              "shortness of breath","respiratory","oxygen"]},
]
def _relevant_vitals():
    # Only consider what the USER actually reported — NOT photo-analysis text
    # we injected as user messages. Those are Claude's AI descriptions and
    # routinely list cardiac warning signs (sweating, palpitations) even for
    # unrelated cases like an elbow lump, which would falsely trigger nudges.
    _PHOTO_PREFIXES = ("Αποτέλεσμα φωτογραφικής", "Photo analysis result")
    user_msgs = [m["content"] for m in st.session_state.triage_chat
                 if m["role"] == "user"
                 and not str(m["content"]).startswith(_PHOTO_PREFIXES)]
    txt = _strip_accents(" ".join(user_msgs))
    return [c for c in _VITAL_CATEGORIES if any(_strip_accents(r) in txt for r in c["roots"])]

# A photo only helps for VISUAL complaints (skin/rash, eye, wound/swelling, mouth/
# throat, nails, lesions...). For non-visual ones (e.g. chest pain, dizziness) the
# camera adds nothing and just confuses, so the photo option is hidden unless the
# conversation is about something visible.
_VISUAL_ROOTS = [
    # Greek (accent-insensitive)
    "δερμα","εξανθημ","σπυρ","πληγ","τραυμ","κοψιμ","εκδορ","εξογκωμ","πρηξ","πρησμ",
    "πρησιμ","οιδημ","μωλωπ","ελια","σπιλ","μελανωμ","εγκαυμ","καψιμ","δαγκωμ","τσιμπ",
    "κνησμ","φαγουρ","φουσκαλ","φλυκταιν","εκζεμ","ψωριασ","ελκος","εξελκωσ","αφθ",
    "οφθαλμ","ματι","λαιμ","αμυγδαλ","φαρυγγ","γλωσσ","νυχι","ονυχ","ουλη","κονδυλωμ",
    "αλλοιωσ","κηλιδ","δοθιην","αποστημ","σπυρακ","πρηξιμ","οζο",
    # Additional: swelling / lump / bump variants (medical + casual Greek)
    "διογκωσ","καρουμπαλ","ογκο","πεταξ","βγηκ","πρισμ","φουσκωμ","φουσκωσ",
    # English
    "skin","rash","lesion","wound","laceration","abrasion","lump","bump","swelling",
    "swollen","bruise","mole","melanoma","eye","throat","tonsil","tongue","nail",
    "burn","bite","itch","blister","eczema","psoriasis","ulcer","pimple","cyst","wart",
]
def _visual_relevant():
    """Show the photo upload option when EITHER:
    (a) the user explicitly mentions a visual symptom (skin/rash/wound/lump/…), OR
    (b) Asklepios's most recent reply explicitly suggested the photo option.
    Case (b) catches descriptions in casual/regional Greek (e.g. «πετάξει κάτι σαν
    βυζί στον αγκώνα» = an elbow lump) where Claude understood and offered the
    photo, but our keyword list couldn't match the unusual phrasing. We only
    check the LAST assistant message so old differential-diagnosis mentions of
    'εξάνθημα' in unrelated chest-pain workups do NOT trigger false positives."""
    # (a) User explicitly mentions a visual symptom
    user_txt = _strip_accents(" ".join(m["content"] for m in st.session_state.triage_chat
                                       if m["role"] == "user"))
    if any(r in user_txt for r in _VISUAL_ROOTS):
        return True
    # (b) Asklepios's LAST message suggests the photo option
    last_assistant = next((m["content"] for m in reversed(st.session_state.triage_chat)
                           if m["role"] == "assistant"), "")
    a_txt = _strip_accents(last_assistant)
    photo_hints = [
        _strip_accents("αναλυση φωτογραφιας"),
        _strip_accents("ανεβασεις φωτογραφια"),
        _strip_accents("ανεβασετε φωτογραφια"),
        _strip_accents("φωτογραφια απο την επιλογη"),
        "photo analysis",
        "upload a photo",
        "upload photo",
    ]
    return any(p in a_txt for p in photo_hints)

# Musculoskeletal / physiotherapy-relevant complaints (shoulder/back/joint pain,
# sprains, post-injury stiffness...). Same accent-insensitive root-matching as
# _relevant_vitals / _visual_relevant. Used to surface the Physiotherapy card
# proactively during triage, not only at the end inside the final report.
_PHYSIO_ROOTS = [
    # Greek (accent-insensitive)
    "ωμ","αυχεν","πλατ","μεσ","οσφ","γοφ","γονατ","αστραγαλ","καρπ","αγκων",
    "διαστρεμ","στρεμπουλ","θλασ","τραβηγ","πιασιμ","πονος στ","μυικ","αρθρ",
    "αρθριτ","οσφυαλγ","αυχεναλγ","ισχιαλγ","δισκοπαθ","τενοντ","θλιψ",
    "κηλη δισκου","φυσικοθεραπ","φυσιοθεραπ","αποκαταστασ","ακαμψ","δυσκαμψ",
    # English
    "shoulder pain","back pain","neck pain","knee pain","hip pain","joint pain",
    "sprain","strain","stiffness","herniated disc","disc herniation","sciatica",
    "tendinitis","tendonitis","physiotherapy","physical therapy","rehab",
]
def _physio_relevant():
    user_txt = _strip_accents(" ".join(m["content"] for m in st.session_state.triage_chat
                                       if m["role"] == "user"))
    return any(_strip_accents(r) in user_txt for r in _PHYSIO_ROOTS)

# Mental-health / psychological-support-relevant complaints (anxiety, stress,
# low mood, sleep issues tied to stress, panic...). Same pattern as above.
# Used to surface the Psychology card proactively during triage.
_PSYCH_ROOTS = [
    # Greek (accent-insensitive)
    "αγχ","στρες","καταθλ","πανικ","φοβ","ανησυχ","θλιψ","απελπισ","μελαγχολ",
    "αυπν","κρισ πανικου","συναισθηματ","ψυχολογ","ψυχικ","ευερεθιστ",
    "ταση παν","σκεψεις","μοναξ","εξαντλησ","καψιμο","burnout",
    # English
    "anxiety","stress","depress","panic","worried","worry","hopeless",
    "insomnia","panic attack","overwhelmed","psycholog","mental health",
    "burnout","low mood","can't sleep","cant sleep",
]
def _psych_relevant():
    user_txt = _strip_accents(" ".join(m["content"] for m in st.session_state.triage_chat
                                       if m["role"] == "user"))
    return any(_strip_accents(r) in user_txt for r in _PSYCH_ROOTS)

def _triage_condition_hint():
    """Best-effort condition phrase from the in-progress triage chat, for use
    BEFORE the final report (and its clean AI-extracted CONDITION) exists.
    Just the recent user text, trimmed — good enough as a PubMed search seed
    since pedro_pillar_search/psychology_pillar_search already fall back to
    a looser query if the strict MeSH+ptype combo finds nothing."""
    user_msgs = [m["content"] for m in st.session_state.triage_chat if m["role"] == "user"]
    return " ".join(user_msgs)[-200:].strip()

def _cached_physio_refs():
    """Physio refs for the proactive triage-time card, cached in session_state
    so repeated Streamlit reruns don't re-hit PubMed on every keystroke."""
    hint = _triage_condition_hint()
    cache = st.session_state.get("_physio_refs_cache")
    if cache and cache.get("hint") == hint:
        return cache.get("refs", [])
    refs = pedro_pillar_search(hint, n=3) if hint else []
    st.session_state["_physio_refs_cache"] = {"hint": hint, "refs": refs}
    return refs

def _cached_psych_refs():
    """Psychology refs for the proactive triage-time card, cached the same way."""
    hint = _triage_condition_hint()
    cache = st.session_state.get("_psych_refs_cache")
    if cache and cache.get("hint") == hint:
        return cache.get("refs", [])
    refs = psychology_pillar_search(hint, n=3) if hint else []
    st.session_state["_psych_refs_cache"] = {"hint": hint, "refs": refs}
    return refs

# Quick-select symptom chips, tailored to the person (age + sex from the profile).
# These are common PRESENTING COMPLAINTS per group — not diagnoses — to speed up the
# first message. Age takes precedence over sex (a child gets paediatric chips). The
# user can always type freely or tap "Άλλο/Other".
_CHIP_SETS = {
    "female": {
        "el": (["Πονοκέφαλος/Ημικρανία","Κοιλιακός/πυελικός πόνος","Διαταραχές περιόδου",
                "Ούρα: καύσος/συχνουρία","Κόπωση","Ναυτία","Ζάλη","Πόνος στήθους",
                "Δύσπνοια","Πόνος μέσης","Εξάνθημα/δέρμα","Άλλο"], "συχνά σε γυναίκες"),
        "en": (["Headache/Migraine","Abdominal/pelvic pain","Menstrual changes",
                "Urinary burning/frequency","Fatigue","Nausea","Dizziness","Chest pain",
                "Shortness of breath","Back pain","Rash/skin","Other"], "common in women"),
    },
    "male": {
        "el": (["Πόνος στήθους","Δύσπνοια","Κοιλιακός πόνος","Πόνος μέσης",
                "Ούρα: δυσουρία/συχνουρία","Πονοκέφαλος","Ζάλη","Κόπωση","Βήχας",
                "Πόνος αρθρώσεων","Εξάνθημα/δέρμα","Άλλο"], "συχνά σε άνδρες"),
        "en": (["Chest pain","Shortness of breath","Abdominal pain","Back pain",
                "Urinary problems","Headache","Dizziness","Fatigue","Cough",
                "Joint pain","Rash/skin","Other"], "common in men"),
    },
    "infant": {
        "el": (["Πυρετός","Ανήσυχο/κλάματα","Εμετός/αναγωγές","Διάρροια","Βήχας/συνάχι",
                "Δυσκολία αναπνοής","Εξάνθημα","Δυσκολία σίτισης","Δυσκοιλιότητα",
                "Ίκτερος (κιτρίνισμα)","Άλλο"], "συχνά σε βρέφη"),
        "en": (["Fever","Irritable/crying","Vomiting/spit-up","Diarrhoea","Cough/congestion",
                "Breathing difficulty","Rash","Feeding difficulty","Constipation",
                "Jaundice","Other"], "common in infants"),
    },
    "child": {
        "el": (["Πυρετός","Βήχας","Πονόλαιμος","Πόνος αυτιού","Κοιλιακός πόνος","Εμετός",
                "Διάρροια","Εξάνθημα","Πονοκέφαλος","Δυσκολία αναπνοής","Άλλο"],
               "συχνά σε παιδιά/εφήβους"),
        "en": (["Fever","Cough","Sore throat","Ear pain","Abdominal pain","Vomiting",
                "Diarrhoea","Rash","Headache","Breathing difficulty","Other"],
               "common in children/teens"),
    },
    "adult": {
        "el": (["Πονοκέφαλος","Πυρετός","Βήχας","Δύσπνοια","Ναυτία","Πόνος στήθους",
                "Κοιλιακός πόνος","Ζάλη","Κόπωση","Πόνος πλάτης","Διάρροια","Άλλο"], ""),
        "en": (["Headache","Fever","Cough","Shortness of breath","Nausea","Chest pain",
                "Abdominal pain","Dizziness","Fatigue","Back pain","Diarrhoea","Other"], ""),
    },
}
def _symptom_chips(profile, lang):
    """Return (chips, group_label) for the person's age/sex group."""
    age = profile.get("age", 0) or 0
    sex = profile.get("sex", "")
    if age <= 16:
        g = "infant" if age < 2 else "child"
    elif sex in ("Γυναίκα", "Female"):
        g = "female"
    elif sex in ("Άνδρας", "Male"):
        g = "male"
    else:
        g = "adult"
    return _CHIP_SETS[g]["el" if lang == "el" else "en"]

def generate_html_report(profile, vitals, report_text, pubmed_refs, lang="el", recs=None, photo_findings=None, lab_findings=None):
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
    # PNOE-style Recommendations section (Exercise / Nutrition / Lifestyle)
    recs_html = ""
    if recs and any(recs.get(k) for k in ("exercise","nutrition","lifestyle")):
        _ex = _html.escape(recs.get("exercise","—"))
        _nu = _html.escape(recs.get("nutrition","—"))
        _li = _html.escape(recs.get("lifestyle","—"))
        _t = ("Εξατομικευμένες Συστάσεις", "Φυσική Δραστηριότητα", "Διατροφή", "Τρόπος Ζωής",
              "Οδηγίες & μετα-αναλύσεις") if lang=="el" \
             else ("Personalised Recommendations", "Exercise", "Nutrition", "Lifestyle",
                   "Guidelines & meta-analyses")
        def _refs_box(pillar):
            items = (recs.get("_refs", {}) or {}).get(pillar) or []
            if not items: return ""
            lis = "".join(
                f'<li><a href="{_html.escape(r.get("url",""))}" target="_blank" '
                f'style="color:#1E40AF;text-decoration:none">'
                f'{_html.escape((r.get("title","—") or "")[:120])}</a>'
                f'<span style="color:#9CA3AF"> · {_html.escape(r.get("journal","") or "")}'
                f'{(" " + _html.escape((r.get("date","") or "")[:4])) if r.get("date") else ""}</span></li>'
                for r in items
            )
            return (f'<div class="recs-refs"><div class="recs-refs-lbl">📚 {_t[4]}</div>'
                    f'<ul>{lis}</ul></div>')
        recs_html = (
            f'<h2>📍 {_t[0]}</h2>'
            '<div class="recs-grid">'
            f'<div class="recs-box exercise"><div class="recs-lbl">🏃 {_t[1]}</div><div>{_ex}</div>{_refs_box("exercise")}</div>'
            f'<div class="recs-box nutrition"><div class="recs-lbl">🥗 {_t[2]}</div><div>{_nu}</div>{_refs_box("nutrition")}</div>'
            f'<div class="recs-box lifestyle"><div class="recs-lbl">🌿 {_t[3]}</div><div>{_li}</div>{_refs_box("lifestyle")}</div>'
            '</div>'
        )
    # Photo findings — if visual analyses exist, add a section with each one.
    photo_html = ""
    if photo_findings and isinstance(photo_findings, list):
        _pf_title = "📷 Ευρήματα από Φωτογραφίες" if lang=="el" else "📷 Photo Findings"
        _pf_items = ""
        for i, pf in enumerate(photo_findings, 1):
            _lbl = _html.escape(pf.get("scan_label","—"))
            _an = _re.sub(r"\s+", " ", (pf.get("analysis","") or "").strip())
            _an = _html.escape(_an)
            _an = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", _an)
            _pf_items += (
                f'<div class="pf-row"><div class="pf-row-head">'
                f'<span class="pf-row-num">{i}</span><span class="pf-row-lbl">{_lbl}</span>'
                f'</div><div class="pf-row-body">{_an}</div></div>'
            )
        photo_html = f'<h2>{_pf_title}</h2><div class="pf-list">{_pf_items}</div>'
    # Lab findings — same structure as photo, green accent for lab data.
    lab_html = ""
    if lab_findings and isinstance(lab_findings, list):
        _lf_title = "🧪 Ευρήματα Εργαστηριακών Εξετάσεων" if lang=="el" else "🧪 Lab Findings"
        _lf_items = ""
        for i, lf in enumerate(lab_findings, 1):
            _lbl = _html.escape(lf.get("file_name","—"))
            _an = _re.sub(r"\s+", " ", (lf.get("analysis","") or "").strip())
            _an = _html.escape(_an)
            _an = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", _an)
            _lf_items += (
                f'<div class="lf-row"><div class="lf-row-head">'
                f'<span class="lf-row-num">{i}</span><span class="lf-row-lbl">📄 {_lbl}</span>'
                f'</div><div class="lf-row-body">{_an}</div></div>'
            )
        lab_html = f'<h2>{_lf_title}</h2><div class="lf-list">{_lf_items}</div>'
    html_out=f"""<!DOCTYPE html><html lang="{lang}"><head><meta charset="UTF-8"><title>Asklepios Report — {name}</title>
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
.recs-grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin:10px 0 16px}}
.recs-box{{border:1px solid;border-radius:10px;padding:12px 14px;font-size:12px;line-height:1.55}}
.recs-box.exercise{{background:#EFF6FF;border-color:#BFDBFE}}
.recs-box.nutrition{{background:#ECFDF5;border-color:#A7F3D0}}
.recs-box.lifestyle{{background:#FEF3F2;border-color:#FECDD3}}
.recs-lbl{{font-size:10px;font-weight:800;letter-spacing:.1em;text-transform:uppercase;color:#1F2937;margin-bottom:6px}}
.recs-refs{{margin-top:8px;padding-top:6px;border-top:1px dashed rgba(0,0,0,0.10)}}
.recs-refs-lbl{{font-size:9px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#6B7280;margin-bottom:4px}}
.recs-refs ul{{list-style:none;padding:0;margin:0}}.recs-refs li{{font-size:10.5px;line-height:1.4;margin-bottom:3px}}
.pf-list{{margin:8px 0 16px}}.pf-row{{padding:10px 0;border-bottom:1px solid #F3F4F6}}.pf-row:last-child{{border-bottom:none}}
.pf-row-head{{display:flex;align-items:center;gap:8px;margin-bottom:5px}}
.pf-row-num{{background:#DBEAFE;color:#1E40AF;font-size:10px;font-weight:700;width:18px;height:18px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center}}
.pf-row-lbl{{font-size:12px;font-weight:700;color:#111827}}.pf-row-body{{font-size:11.5px;color:#374151;line-height:1.55}}
.lf-list{{margin:8px 0 16px}}.lf-row{{padding:10px 0;border-bottom:1px solid #F3F4F6}}.lf-row:last-child{{border-bottom:none}}
.lf-row-head{{display:flex;align-items:center;gap:8px;margin-bottom:5px}}
.lf-row-num{{background:#D1FAE5;color:#065F46;font-size:10px;font-weight:700;width:18px;height:18px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center}}
.lf-row-lbl{{font-size:12px;font-weight:700;color:#111827}}.lf-row-body{{font-size:11.5px;color:#374151;line-height:1.55}}
@media print{{.recs-box{{-webkit-print-color-adjust:exact;print-color-adjust:exact}}.recs-grid{{grid-template-columns:1fr 1fr 1fr !important}}}}
.hint{{text-align:center;margin:24px 0 0;font-size:12px;color:#94A3B8;border-top:1px dashed #E0E5FF;padding-top:14px}}
@media print{{body{{padding:16px}}.patient,.emergency{{-webkit-print-color-adjust:exact;print-color-adjust:exact}}@page{{margin:15mm}}}}</style></head><body>
<div class="hdr"><div class="hdr-logo">🩺 Asklepios AI Nurse</div><div class="hdr-date">Κλινική Εκτίμηση<br>{ts}</div></div>
<div class="patient"><div class="patient-name">{name}</div><div class="patient-meta">{age} ετών · {sex}</div>
<div class="patient-detail"><strong>Ιστορικό:</strong> {hx}<br><strong>Αλλεργίες:</strong> {allg}<br><strong>Φάρμακα:</strong> {meds}</div></div>
{vitals_sec}<h2>Κλινική Αξιολόγηση</h2>{md2h(report_text or "")}{photo_html}{lab_html}{recs_html}{refs_html}
<div class="emergency">🚨 ΣΕ ΕΠΕΙΓΟΥΣΑ ΑΝΑΓΚΗ: ΚΑΛΕΣΤΕ 166 (ΕΚΑΒ) ή 112</div>
<div class="disclaimer">⚠️ AI-generated. Δεν αποτελεί ιατρική διάγνωση. Απαιτείται επίσκεψη σε επαγγελματία υγείας.</div>
<div class="hint">💡 Ctrl+P → Save as PDF</div></body></html>"""
    return html_out.encode("utf-8")

def _render_symptom_tracker(lang):
    """Browser-only symptom log. All data in localStorage — nothing on servers.
    User can add dated entries (symptom + severity + notes), view history,
    and export as text. Built as a self-contained HTML/JS component so it works
    regardless of login state. Privacy: we never see this data.
    """
    # Symptom tracker uses st.iframe (HTML string mode)
    _title = "📅 Ημερολόγιο Συμπτωμάτων" if lang=="el" else "📅 Symptom Log"
    _privacy = ("Αποθηκεύεται μόνο στον browser σου — δεν αποστέλλεται πουθενά."
                if lang=="el" else
                "Stored only in your browser — never sent anywhere.")
    with st.expander(f"{_title} — {_privacy}", expanded=False):
        if lang == "el":
            tx = {
                "add_title":   "Προσθήκη σημερινού συμπτώματος",
                "symptom_ph":  "π.χ. πονοκέφαλος, βήχας, κοιλιακός πόνος",
                "sev_lbl":     "Βαρύτητα (1–10)",
                "notes_ph":    "Επιπλέον παρατηρήσεις (προαιρετικό)",
                "add_btn":     "➕ Καταχώρηση",
                "history":     "Ιστορικό",
                "no_entries":  "Κανένα σύμπτωμα ακόμη.",
                "clear_btn":   "🗑️ Διαγραφή όλων",
                "export_btn":  "📋 Αντιγραφή ιστορικού",
                "exported":    "✅ Αντιγράφηκε!",
                "sev_prefix":  "Βαρύτητα",
                "confirm_clear":"Διαγραφή ΟΛΩΝ των συμπτωμάτων; Δεν αναιρείται.",
            }
        else:
            tx = {
                "add_title":   "Log today's symptom",
                "symptom_ph":  "e.g. headache, cough, stomach pain",
                "sev_lbl":     "Severity (1–10)",
                "notes_ph":    "Additional notes (optional)",
                "add_btn":     "➕ Add entry",
                "history":     "History",
                "no_entries":  "No symptoms logged yet.",
                "clear_btn":   "🗑️ Clear all",
                "export_btn":  "📋 Copy log",
                "exported":    "✅ Copied!",
                "sev_prefix":  "Severity",
                "confirm_clear":"Delete ALL symptom entries? Cannot be undone.",
            }
        st.iframe(f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0;font-family:system-ui,sans-serif}}
body{{background:transparent;padding:0;font-size:14px;color:#1F2937}}
.st-card{{background:white;border:1px solid #E5E7EB;border-radius:12px;padding:16px 18px;margin-bottom:12px}}
.st-card h3{{font-size:12px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#6B7280;margin-bottom:12px}}
.st-row{{display:flex;gap:8px;align-items:flex-end;flex-wrap:wrap}}
input[type=text],textarea{{width:100%;border:1px solid #D1D5DB;border-radius:8px;padding:8px 10px;font-size:13px;color:#1F2937;background:white}}
input[type=text]:focus,textarea:focus{{outline:none;border-color:#2D3FE7;box-shadow:0 0 0 2px rgba(45,63,231,.10)}}
textarea{{resize:vertical;min-height:48px}}
input[type=range]{{width:100%;accent-color:#2D3FE7}}
.sev-row{{display:flex;align-items:center;gap:8px}}
.sev-label{{font-size:11px;color:#6B7280;white-space:nowrap}}
.sev-val{{font-size:18px;font-weight:700;color:#2D3FE7;min-width:24px;text-align:right}}
.btn{{padding:9px 16px;border-radius:8px;border:none;cursor:pointer;font-weight:600;font-size:13px;transition:all .15s}}
.btn-primary{{background:#2D3FE7;color:white}}.btn-primary:hover{{background:#1E30CC}}
.btn-ghost{{background:#F3F4F6;color:#374151;border:1px solid #E5E7EB}}.btn-ghost:hover{{background:#E5E7EB}}
.btn-danger{{background:#FEF2F2;color:#DC2626;border:1px solid #FCA5A5}}.btn-danger:hover{{background:#FEE2E2}}
.entry{{border-bottom:1px solid #F3F4F6;padding:10px 0;display:flex;justify-content:space-between;align-items:flex-start;gap:8px}}
.entry:last-child{{border-bottom:none}}
.entry-main{{flex:1}}
.entry-date{{font-size:11px;color:#9CA3AF;margin-bottom:2px}}
.entry-symptom{{font-size:14px;font-weight:600;color:#111827}}
.entry-sev{{display:inline-block;padding:2px 8px;border-radius:20px;font-size:11px;font-weight:700;margin-left:6px}}
.entry-notes{{font-size:12px;color:#6B7280;margin-top:3px}}
.del-btn{{background:none;border:none;cursor:pointer;color:#9CA3AF;font-size:16px;padding:2px 4px;flex-shrink:0}}.del-btn:hover{{color:#DC2626}}
.empty{{text-align:center;padding:24px;color:#9CA3AF;font-size:13px}}
.tools{{display:flex;gap:8px;margin-top:8px}}
</style></head><body>

<div class="st-card">
  <h3>{tx['add_title']}</h3>
  <input type="text" id="symp" placeholder="{tx['symptom_ph']}" />
  <div style="margin-top:10px">
    <div class="sev-row">
      <span class="sev-label">{tx['sev_lbl']}</span>
      <input type="range" id="sev" min="1" max="10" value="5"
             oninput="document.getElementById('sev-val').textContent=this.value" />
      <span class="sev-val" id="sev-val">5</span>
    </div>
  </div>
  <textarea id="notes" placeholder="{tx['notes_ph']}" style="margin-top:10px"></textarea>
  <div style="margin-top:10px">
    <button class="btn btn-primary" onclick="addEntry()">{tx['add_btn']}</button>
  </div>
</div>

<div class="st-card">
  <h3>{tx['history']}</h3>
  <div id="list"></div>
  <div class="tools" id="tools" style="display:none">
    <button class="btn btn-ghost" onclick="exportLog()">{tx['export_btn']}</button>
    <button class="btn btn-danger" onclick="clearAll()">{tx['clear_btn']}</button>
  </div>
</div>

<script>
var STORE_KEY = "asklepios_symptoms_v1";

function load() {{
  try {{ return JSON.parse(localStorage.getItem(STORE_KEY) || "[]"); }}
  catch(e) {{ return []; }}
}}
function save(entries) {{
  localStorage.setItem(STORE_KEY, JSON.stringify(entries));
}}

function sevColor(s) {{
  if(s<=3) return "#ECFDF5;color:#065F46";
  if(s<=6) return "#FFFBEB;color:#92400E";
  return "#FEF2F2;color:#991B1B";
}}

function renderList() {{
  var entries = load();
  var el = document.getElementById("list");
  var tools = document.getElementById("tools");
  if(!entries.length) {{
    el.innerHTML = '<div class="empty">{tx['no_entries']}</div>';
    tools.style.display = "none";
    return;
  }}
  tools.style.display = "flex";
  // newest first
  var html = "";
  for(var i=entries.length-1; i>=0; i--) {{
    var e = entries[i];
    var sc = sevColor(e.sev);
    var sc_parts = sc.split(";color:");
    var bg = sc_parts[0];
    var fg = sc_parts[1] || "#111";
    html += '<div class="entry">';
    html += '<div class="entry-main">';
    html += '<div class="entry-date">'+e.date+'</div>';
    html += '<div class="entry-symptom">'+escape_html(e.symptom);
    html += ' <span class="entry-sev" style="background:'+bg+';color:'+fg+'">'+e.sev+'/10</span></div>';
    if(e.notes) html += '<div class="entry-notes">'+escape_html(e.notes)+'</div>';
    html += '</div>';
    html += '<button class="del-btn" onclick="deleteEntry('+i+')" title="Delete">✕</button>';
    html += '</div>';
  }}
  el.innerHTML = html;
}}

function escape_html(s) {{
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}}

function addEntry() {{
  var symp = document.getElementById("symp").value.trim();
  if(!symp) {{ document.getElementById("symp").focus(); return; }}
  var sev  = parseInt(document.getElementById("sev").value);
  var notes= document.getElementById("notes").value.trim();
  var now  = new Date();
  var date = now.toLocaleDateString("{("el-GR" if lang=="el" else "en-GB")}",
    {{day:"2-digit",month:"short",year:"numeric",hour:"2-digit",minute:"2-digit"}});
  var entries = load();
  entries.push({{date:date, symptom:symp, sev:sev, notes:notes}});
  save(entries);
  document.getElementById("symp").value="";
  document.getElementById("notes").value="";
  document.getElementById("sev").value=5;
  document.getElementById("sev-val").textContent="5";
  renderList();
}}

function deleteEntry(idx) {{
  var entries = load();
  entries.splice(idx,1);
  save(entries);
  renderList();
}}

function clearAll() {{
  if(confirm("{tx['confirm_clear']}")) {{
    localStorage.removeItem(STORE_KEY);
    renderList();
  }}
}}

function exportLog() {{
  var entries = load();
  if(!entries.length) return;
  var txt = entries.map(function(e){{
    var line = e.date+" | "+e.symptom+" | {tx['sev_prefix']}: "+e.sev+"/10";
    if(e.notes) line += " | "+e.notes;
    return line;
  }}).join("\\n");
  navigator.clipboard.writeText(txt).then(function(){{
    var b = document.querySelector(".btn-ghost");
    var orig = b.textContent;
    b.textContent="{tx['exported']}";
    setTimeout(function(){{b.textContent=orig;}},2000);
  }});
}}

renderList();
</script>
</body></html>""", height=520)


def render_home():
    """Formeto-style home screen, matching the approved mockup:
      1) Topbar — wordmark + greeting + circular avatar with initial
      2) Action-grid — 2 big square cards (Συμπτώματα / Ζωτικά) as the
         primary entry points
      3) Explainer banner — shown until profile is completed
      4) Emergency disclaimer"""
    lang = st.session_state.lang
    p = st.session_state.profile
    name = p.get("name", "")
    has_profile = bool(name)
    el = (lang == "el")

    # ── Shared styles for the Formeto-style home layout ──────────────────
    st.markdown("""
<style>
.home-topbar { display:flex; align-items:center; justify-content:space-between; margin:4px 0 20px; }
.home-brand { font-size:24px; font-weight:800; color:#1A1A2E; }
.home-greeting { font-size:12.5px; color:#6B89B0; font-weight:600; margin-top:2px; }
.home-avatar {
  width:46px; height:46px; border-radius:50%; background:#2D3FE7;
  display:flex; align-items:center; justify-content:center;
  color:white; font-size:18px; font-weight:700;
  box-shadow:0 4px 10px rgba(45,63,231,0.28); flex-shrink:0;
}
.home-action-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:18px; }
.home-action-icon {
  width:52px; height:52px; border-radius:50%; background:#E8ECFE; margin:0 auto 12px;
  display:flex; align-items:center; justify-content:center; font-size:23px;
}
.home-action-icon.warm { background:#FFEFE8; }
.home-action-label { font-size:14.5px; font-weight:700; color:#1A1A2E; line-height:1.3; }
.home-group-title { font-size:15px; font-weight:800; color:#1A1A2E; margin-bottom:16px; }
.home-vrow { display:flex; align-items:center; gap:13px; margin-bottom:16px; }
.home-vrow:last-child { margin-bottom:0; }
.home-vrow-icon {
  width:38px; height:38px; border-radius:50%; background:#E8ECFE; flex-shrink:0;
  display:flex; align-items:center; justify-content:center; font-size:17px;
}
.home-vrow-body { flex:1; min-width:0; }
.home-vrow-top { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:6px; }
.home-vrow-label { font-size:13.5px; font-weight:700; color:#1A1A2E; }
.home-vrow-val { font-size:13px; font-weight:700; color:#2D3FE7; }
.home-vrow-bar { height:6px; background:#EAF0FB; border-radius:6px; overflow:hidden; }
.home-vrow-bar-fill { height:100%; border-radius:6px; background:#2D3FE7; }
.home-vrow-bar-fill.ok { background:#10B981; }
.home-vrow-bar-fill.warn { background:#F59E0B; }
.home-vrow-bar-fill.danger { background:#EF4444; }
.home-emergency {
  background:#FEF2F2; border:1px solid #FECACA; border-radius:12px;
  padding:13px 16px; font-size:12.5px; color:#7F1D1D; line-height:1.55; margin-top:18px;
}
.home-emergency strong { color:#991B1B; }
.home-campaign-banner {
  background: white; border: 1px solid #E0E5FF; border-radius: 18px;
  padding: 14px 16px; margin-bottom: 16px;
  display: flex; align-items: center; gap: 12px;
  box-shadow: 0 2px 8px rgba(45,63,231,0.06);
  text-decoration: none;
}
.home-campaign-banner img {
  width: 44px; height: 44px; border-radius: 12px; object-fit: cover; flex-shrink: 0;
}
.home-campaign-banner .cb-title { font-size:13.5px; font-weight:700; color:#1A1A2E; }
.home-campaign-banner .cb-sub { font-size:11.5px; color:#6B89B0; margin-top:1px; }
</style>
""", unsafe_allow_html=True)

    # ── Active campaign banner + popup (admin-managed, see render_admin_panel) ──
    # Banner: a small inline card at the top of Home, always visible while active.
    # Popup: a dismissible modal (st.dialog), shown ONCE per browser session —
    # tracked via st.session_state so it doesn't reappear on every rerun/click
    # within the same visit. Both use the same date-range filtering logic.
    _today_iso = datetime.now().date().isoformat()
    _all_active_campaigns = [c for c in _admin_list("campaigns", order_col="starts_on")
                              if c.get("active", True)
                              and c.get("starts_on", "") <= _today_iso <= c.get("ends_on", "9999-12-31")]
    _campaigns = [c for c in _all_active_campaigns if c.get("placement", "banner") == "banner"]
    if _campaigns:
        _camp = _campaigns[0]
        _img_html = f'<img src="{_camp["image_url"]}">' if _camp.get("image_url") else ""
        _link = _camp.get("link_url") or "#"
        st.markdown(
            f'<a href="{_link}" target="_blank" class="home-campaign-banner">'
            f'{_img_html}'
            f'<div><div class="cb-title">{_camp.get("title","")}</div>'
            f'<div class="cb-sub">{"Ενημέρωση από συνεργαζόμενο φορέα" if el else "Update from a partner organisation"}</div></div>'
            f'</a>',
            unsafe_allow_html=True,
        )

    _popups = [c for c in _all_active_campaigns if c.get("placement", "banner") == "popup"]
    if _popups:
        _pop = _popups[0]
        _seen_key = f"_popup_seen_{_pop.get('id') or _pop.get('title','')}"
        if not st.session_state.get(_seen_key):
            @st.dialog(_pop.get("title", ""))
            def _show_campaign_popup():
                if _pop.get("image_url"):
                    st.image(_pop["image_url"], use_container_width=True)
                st.write(_pop.get("title", ""))
                c1, c2 = st.columns(2)
                with c1:
                    if _pop.get("link_url"):
                        st.link_button(
                            "Περισσότερα →" if el else "Learn more →",
                            _pop["link_url"], use_container_width=True,
                        )
                with c2:
                    if st.button("Κλείσιμο" if el else "Close", use_container_width=True):
                        st.session_state[_seen_key] = True
                        st.rerun()
            _show_campaign_popup()
            st.session_state[_seen_key] = True

    # ── 1) Topbar ─────────────────────────────────────────────────────────
    initial = (name[:1] or "?").upper() if name else "?"
    greeting = (f"Καλημέρα, {name}" if name else "Καλημέρα!") if el else (f"Hi, {name}" if name else "Hi there!")
    # Avatar: shows user's initial if profile exists, or a prompt to complete intake
    _avatar_content = initial if (name and initial != "?") else ("👤" if el else "👤")
    _avatar_title   = name if name else ("Συμπλήρωσε το προφίλ σου" if el else "Complete your profile")
    st.markdown(f"""
<div class="home-topbar">
  <div>
    <div class="home-brand">Asklepios</div>
    <div class="home-greeting">{greeting}</div>
  </div>
  <div class="home-avatar" title="{_avatar_title}" style="font-size:{'18px' if (name and initial != '?') else '22px'};">{_avatar_content}</div>
</div>
""", unsafe_allow_html=True)

    # ── Explainer banner — shown until user completes first assessment ────────
    if not has_profile:
        _exp_title = "Από πού ξεκινάω;" if el else "Where do I start?"
        _exp_body  = ("Πάτα <strong>Έλεγχος Συμπτωμάτων</strong> για να ξεκινήσεις. Το Asklepios θα σε ρωτήσει για το προφίλ σου και μετά θα αξιολογήσει τα συμπτώματά σου βήμα-βήμα." if el else
                      "Tap <strong>Check Symptoms</strong> to begin. Asklepios will ask for your profile and then assess your symptoms step by step.")
        st.markdown(f"""
<div style="background:#EEF2FF;border:1px solid #C7D2FE;border-radius:14px;
  padding:13px 16px;margin:0 0 16px;font-family:'Inter',system-ui,sans-serif;
  display:flex;gap:12px;align-items:flex-start;">
  <span style="font-size:22px;flex-shrink:0;margin-top:1px;">💡</span>
  <div>
    <div style="font-size:13.5px;font-weight:700;color:#1A1A2E;margin-bottom:3px;">{_exp_title}</div>
    <div style="font-size:12.5px;color:#4B5563;line-height:1.5;">{_exp_body}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    def _go(target):
        st.session_state.screen = target if has_profile else "intake"
        st.rerun()

    # Same reliable marker+:has() pattern already used for the bottom nav:
    # a hidden marker inside each card's container lets us target that
    # specific container's button via CSS, since Streamlit buttons carry no
    # per-instance attribute we could otherwise select on.
    st.markdown("""
<style>
div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .home-action-marker) button {
  background: transparent !important; border: none !important; box-shadow: none !important;
  color: #1A1A2E !important; font-weight: 700 !important; font-size: 14.5px !important;
  padding: 4px 0 0 !important; line-height: 1.3 !important;
}
</style>
""", unsafe_allow_html=True)

    ac1, ac2 = st.columns(2, gap="small")
    with ac1:
        with st.container(border=True):
            st.markdown(
                '<div class="home-action-marker"></div>'
                '<div style="text-align:center"><div class="home-action-icon">💬</div></div>',
                unsafe_allow_html=True,
            )
            _lbl1 = ("Έλεγχος Συμπτωμάτων" if el else "Symptom Check")
            if st.button(_lbl1, key="home_go_triage", use_container_width=True):
                _go("triage")
    with ac2:
        with st.container(border=True):
            st.markdown(
                '<div class="home-action-marker"></div>'
                '<div style="text-align:center"><div class="home-action-icon warm">❤️</div></div>',
                unsafe_allow_html=True,
            )
            _lbl2 = ("Ζωτικά Σημεία" if el else "Vital Signs")
            if st.button(_lbl2, key="home_go_vitals", use_container_width=True):
                _go("vitals")

    # ── Intro video (A2E avatar) ─────────────────────────────────────────
    # Pre-generated ONCE offline via a2e_intro_video.py (not at runtime — we
    # don't want to spend A2E credits on every page load). The resulting mp4
    # URL is stored as a secret, same pattern as the other API keys. If it's
    # not configured yet, this section simply doesn't render — no broken UI,
    # no placeholder box.
    _intro_url = (get_setting(f"intro_video_url_{lang}") or get_setting("intro_video_url_el")
                  or _key(f"A2E_INTRO_VIDEO_URL_{lang.upper()}") or _key("A2E_INTRO_VIDEO_URL_EL"))
    if _intro_url:
        with st.container(border=True):
            st.markdown(
                ("##### 🎬 Πώς λειτουργεί ο Ασκληπιός" if lang == "el"
                 else "##### 🎬 How Asklepios works"),
            )
            st.video(_intro_url)

    # Emergency disclaimer — condensed, always visible
    _em_text = (
        "Για πόνο στο στήθος, δυσκολία αναπνοής, σοβαρή αιμορραγία, απώλεια "
        "συνείδησης ή συμπτώματα εγκεφαλικού, καλέστε αμέσως 166 (ΕΚΑΒ) ή 112."
        if el else
        "For chest pain, difficulty breathing, severe bleeding, loss of "
        "consciousness, or stroke symptoms, call 166 (EKAB) or 112 immediately."
    )
    st.markdown(
        f'<div class="home-emergency"><strong>🚨 {"Επείγον" if el else "Emergency"}:</strong> {_em_text}</div>',
        unsafe_allow_html=True,
    )


def render_history():
    """Ιστορικό tab: the latest generated report (if any) + the browser-only
    symptom log. This is where _render_symptom_tracker now lives — it used to
    be buried in an expander at the bottom of Home; giving it its own tab
    makes past data something the person can deliberately go look for instead
    of stumbling onto by scrolling."""
    lang = st.session_state.lang
    render_doc_header(
        "Ιστορικό", "History",
        icon="📋",
        sub_el="Προηγούμενη αναφορά & ημερολόγιο συμπτωμάτων",
        sub_en="Latest report & symptom log",
        show_date=False,
    )
    if st.session_state.report:
        with st.container(border=True):
            st.markdown("##### 📄 " + ("Τελευταία Αναφορά" if lang=="el" else "Latest Report"))
            _preview = st.session_state.report.strip()
            if len(_preview) > 280:
                _preview = _preview[:280].rsplit(" ", 1)[0] + "…"
            st.markdown(_preview)
            if st.button("→ " + ("Άνοιγμα πλήρους αναφοράς" if lang=="el" else "Open full report"),
                         key="hist_open_report", use_container_width=True):
                st.session_state.screen = "report"; st.rerun()
    else:
        st.info(("Δεν έχεις ακόμη ολοκληρωμένη αναφορά. Ξεκίνα μια εκτίμηση από το tab «Συμπτώματα»."
                 if lang=="el" else
                 "No completed report yet. Start an assessment from the «Symptoms» tab."))
    st.divider()
    _render_symptom_tracker(lang)
    # ── Articles / Blog ──────────────────────────────────────────────────────
    # Admin-managed content (see render_admin_panel → _admin_articles_tab).
    # Only shows active articles in the current language; silently shows
    # nothing if Supabase isn't configured or there's no content yet — same
    # graceful-degradation pattern as the rest of the app.
    _articles = [a for a in _admin_list("articles", order_col="published_at")
                 if a.get("active", True) and a.get("lang", "el") == lang]
    if _articles:
        st.divider()
        st.markdown("##### 📰 " + ("Άρθρα" if lang == "el" else "Articles"))
        for art in _articles[:10]:
            with st.container(border=True):
                st.markdown(f"**{art.get('title','—')}**")
                _meta = " · ".join(x for x in [art.get("author",""), art.get("source",""),
                                                str(art.get("published_at",""))] if x)
                if _meta:
                    st.caption(_meta)
                st.write(art.get("summary",""))
                if art.get("body"):
                    with st.expander("Διάβασε περισσότερα" if lang=="el" else "Read more"):
                        st.write(art["body"])
                if art.get("url"):
                    st.markdown(f"[{'Πλήρες άρθρο →' if lang=='el' else 'Full article →'}]({art['url']})")


def render_intake():
    render_stepper("intake")
    lang = st.session_state.lang
    render_doc_header(
        "Πες μας λίγα για σένα", "Tell us about yourself",
        icon="👤",
        sub_el="Όνομα, ηλικία, ιατρικό ιστορικό",
        sub_en="Name, age, medical history",
    )
    # ── Caregiver toggle ───────────────────────────────────────────────────
    # First question: is this assessment for the user themselves or someone
    # they care for (γιαγιά, παιδί, κλπ). Affects copy + Claude system prompt.
    _caregiver_q = ("Για ποιον είναι αυτή η αξιολόγηση;" if lang=="el"
                    else "Who is this assessment for?")
    _opt_self = "Για μένα" if lang=="el" else "For me"
    _opt_other = "Για άλλο άτομο που φροντίζω" if lang=="el" else "For someone I care for"
    _current = st.session_state.profile.get("for_whom", "self")
    with st.container(border=True):
        _choice = st.radio(
            _caregiver_q,
            [_opt_self, _opt_other],
            index=(0 if _current == "self" else 1),
            horizontal=True,
            key="intake_for_whom",
        )
        is_caregiver = (_choice == _opt_other)
        if is_caregiver:
            st.info("💡 " + ("Συμπλήρωσε τα στοιχεία του ατόμου που φροντίζεις (όχι τα δικά σου)."
                             if lang=="el" else
                             "Fill in details of the person you care for (not your own)."))
            _name_lbl = "Όνομα του ασθενούς" if lang=="el" else "Patient's name"
            _name_ph  = "π.χ. Γιαγιά Ελένη" if lang=="el" else "e.g. Grandma Helen"
        else:
            _name_lbl = t("name")
            _name_ph  = "Χριστόφορος"
        c1,c2,c3=st.columns([2,1,1])
        with c1: name=st.text_input(_name_lbl,value=st.session_state.profile.get("name",""),placeholder=_name_ph)
        with c2: age=st.number_input(t("age"),min_value=0,max_value=120,value=st.session_state.profile.get("age",40))
        with c3: sex=st.selectbox(t("sex"),[t("male"),t("female"),t("other")])
        # ── Pregnancy checkbox ──────────────────────────────────────────────
        # Only shown for female + age 12-55 (reproductive age). Affects drug
        # contraindications + Claude system prompt + recs.
        pregnancy = False
        _is_female = sex in ("Γυναίκα", "Female")
        if _is_female and 12 <= age <= 55:
            _preg_lbl = "🤰 Είναι έγκυος;" if lang=="el" else "🤰 Is she pregnant?"
            pregnancy = st.checkbox(_preg_lbl, value=st.session_state.profile.get("pregnancy", False))
            if pregnancy:
                st.info("💡 " + ("Σημειώνεται για έλεγχο αντενδείξεων φαρμάκων και συστάσεων."
                                 if lang=="el" else
                                 "Noted — used to flag drug contraindications and adjusted recommendations."))

    with st.container(border=True):
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
                st.session_state.profile={
                    "name":name, "age":age, "sex":sex,
                    "history":history, "allergies":allergies, "meds_raw":meds_raw,
                    "for_whom": "other" if is_caregiver else "self",
                    "pregnancy": bool(pregnancy),
                }
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
    lang=st.session_state.lang
    nm = p.get("name","")
    render_doc_header(
        "Πώς είναι τα ζωτικά σου;", "How are your vitals?",
        icon="❤️",
        sub_el=(f"για τον/την {nm}" if nm else "Χειροκίνητα, με συσκευή ή σάρωση προσώπου"),
        sub_en=(f"for {nm}" if nm else "Manual, device, or face scan"),
    )

    # ── Tab layout: Manual (default) | Device Import | Face Scan (experimental) ──
    tab_manual, tab_device, tab_scan = st.tabs([
        "✏️ " + ("Χειροκίνητη Εισαγωγή" if lang=="el" else "Manual Entry"),
        "⌚ " + ("Εισαγωγή από Συσκευή" if lang=="el" else "Import from Device"),
        "📷 " + ("Σάρωση (πειραματικό)" if lang=="el" else "Face Scan (experimental)"),
    ])

    with tab_scan:
        st.caption(("⚠️ Πειραματικό. Η σάρωση με κάμερα δίνει μόνο ενδεικτικό καρδιακό ρυθμό — για αξιόπιστες τιμές χρησιμοποίησε «Χειροκίνητη Εισαγωγή» ή «Συσκευή»."
                    if lang=="el" else
                    "⚠️ Experimental. The camera scan gives only an indicative heart rate — for reliable values use 'Manual Entry' or 'Device'."))
        facescan_url=_secret("FACESCAN_URL","https://asklepiosnurse.netlify.app")
        kira_url=_secret("ASKLEPIOS_URL","https://asklepiosainurse.up.railway.app")
        scan_link=f"{facescan_url}?kira_url={urllib.parse.quote(kira_url)}"
        _save_session_for_external_nav()
        st.markdown(f'''<div style="background:linear-gradient(135deg,#2D3FE7,#7B2FE0);border-radius:16px;padding:28px;text-align:center;color:white;margin:8px 0">
            <div style="font-size:40px;margin-bottom:8px">📷</div>
            <div style="font-size:18px;font-weight:700;margin-bottom:8px">{"Σάρωση Προσώπου rPPG" if lang=="el" else "rPPG Face Scan"}</div>
            <div style="font-size:13px;opacity:0.8;margin-bottom:16px">{"Μέτρηση καρδιακού ρυθμού & αναπνοής σε 30 δευτερόλεπτα μέσω κάμερας" if lang=="el" else "Measure heart rate & breathing in 60 seconds via camera"}</div>
            <a href="{scan_link}" target="_blank" style="background:white;color:#2D3FE7;padding:12px 28px;border-radius:8px;font-weight:700;text-decoration:none;font-size:14px">
                {"Έναρξη Σάρωσης →" if lang=="el" else "Start Scan →"}
            </a>
        </div>''', unsafe_allow_html=True)
        st.caption("✅ Μετράει: Καρδιακός ρυθμός, αναπνοή  |  ⚠️ Εκτίμηση: HRV, stress  |  ❌ Δεν μετράει: Αρτηριακή πίεση" if lang=="el"
                   else "✅ Measures: Heart rate, breathing  |  ⚠️ Estimate: HRV, stress  |  ❌ Does not measure: Blood pressure")

    with tab_device:
        st.markdown(f"### {'Εισαγωγή από Smartwatch / Οξύμετρο' if lang=='el' else 'Import from Smartwatch / Oximeter'}")
        st.caption("Apple Watch · Fitbit · Garmin · Polar · Finger oximeter" if lang=="el" else "Apple Watch · Fitbit · Garmin · Polar · Finger oximeter")

        d1, d2 = st.columns(2)
        with d1:
            st.markdown(f"**{'Apple Watch / Smartwatch' if lang=='el' else 'Apple Watch / Smartwatch'}**")
            dev_hr   = st.number_input("Heart Rate (bpm)", min_value=0, max_value=300, value=None, placeholder="76", key="dev_hr")
            dev_hrv  = st.number_input("HRV (ms)", min_value=0, max_value=300, value=None, placeholder="45", key="dev_hrv")
            dev_spo2 = st.number_input("SpO2 (%)", min_value=0, max_value=100, value=None, placeholder="98", key="dev_spo2")
            dev_br   = st.number_input("Breathing Rate (/min)", min_value=0, max_value=60, value=None, placeholder="15", key="dev_br")
        with d2:
            st.markdown(f"**{'Πιεσόμετρο / Άλλη Συσκευή' if lang=='el' else 'Blood Pressure Monitor / Other'}**")
            dev_bps  = st.number_input("BP Systolic (mmHg)", min_value=0, max_value=300, value=None, placeholder="120", key="dev_bps")
            dev_bpd  = st.number_input("BP Diastolic (mmHg)", min_value=0, max_value=200, value=None, placeholder="80", key="dev_bpd")
            dev_temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=45.0, value=None, placeholder="36.6", key="dev_temp", format="%.1f")
            dev_wt   = st.number_input("Weight (kg)", min_value=0.0, max_value=300.0, value=None, placeholder="75", key="dev_wt", format="%.1f")

        st.markdown(f"**{'Ύψος (για ΔΜΣ)' if lang=='el' else 'Height (for BMI)'}**")
        dev_ht = st.number_input("Height (cm)", min_value=0, max_value=250, value=None, placeholder="175", key="dev_ht")

        if st.button(f"{'Φόρτωση δεδομένων συσκευής' if lang=='el' else 'Load device data'}", type="primary", key="load_device", use_container_width=True):
            vd = {}
            if dev_hr:   vd["hr"]     = int(dev_hr)
            if dev_hrv:  vd["hrv"]    = int(dev_hrv)
            if dev_spo2: vd["spo2"]   = int(dev_spo2)
            if dev_br:   vd["br"]     = int(dev_br)
            if dev_bps:  vd["bp_sys"] = int(dev_bps)
            if dev_bpd:  vd["bp_dia"] = int(dev_bpd)
            if dev_temp: vd["temp"]   = float(dev_temp)
            if dev_wt:   vd["weight"] = float(dev_wt)
            if dev_ht:   vd["height"] = int(dev_ht)
            if vd:
                classify_vitals(vd, age=p.get("age"))
                st.session_state.vitals = vd
                st.session_state["_device_loaded"] = True
            else:
                st.warning("Εισάγετε τουλάχιστον έναν δείκτη." if lang=="el" else "Enter at least one metric.")

        # Show confirmation + vitals + proceed button (no rerun needed)
        if st.session_state.get("_device_loaded") and st.session_state.vitals:
            v_loaded = st.session_state.vitals
            st.success(f"{'✅ Δεδομένα φορτώθηκαν:' if lang=='el' else '✅ Data loaded:'} " +
                       " | ".join(f"{k}={v}" for k,v in v_loaded.items()))
            if st.button(f"{'Συνέχεια στην Εκτίμηση →' if lang=='el' else 'Continue to Assessment →'}",
                         type="primary", key="dev_continue", use_container_width=True):
                st.session_state["_device_loaded"] = False
                with st.spinner("Ανάλυση..."):
                    vtext = "\n".join(f"- {k}: {val}" for k,val in v_loaded.items())
                    pp = p.get
                    st.session_state.vitals_analysis = claude(
                        [{"role":"user","content":f"Patient: {pp('name')}, {pp('age')}yo {pp('sex')}, Hx: {pp('history','none')}, Meds: {pp('meds_raw','none')}\n\nVitals:\n{vtext}\n\nInterpret. Categorise each. Flag urgent findings. Be direct."}],
                        system=kira_system(), max_tokens=1200
                    )
                st.session_state.screen = "triage"
                st.rerun()

        # How-to guide
        with st.expander(f"{'Πώς να εξαγάγετε δεδομένα από τη συσκευή σας' if lang=='el' else 'How to export data from your device'}"):
            st.markdown("""
**Apple Watch / iPhone:**
Health app → Browse → Heart → Heart Rate → export or note the value

**Fitbit:**
Fitbit app → Today → Heart Rate tile

**Garmin / Polar:**
Garmin Connect / Polar Flow app → Dashboard → Heart Rate

**Finger oximeter:**
Read SpO2 and HR directly from the device display

**Blood pressure monitor:**
Use a certified upper-arm cuff device, note systolic/diastolic values
            """)
    with tab_manual:
        v=st.session_state.vitals
        with st.container(border=True):
            st.markdown(
                f'<div class="home-group-title">{"Βασικές Μετρήσεις" if lang=="el" else "Core Vitals"}</div>',
                unsafe_allow_html=True,
            )
            cc1,cc2=st.columns(2)
            with cc1:
                hr=st.number_input(t("hr"),min_value=0,max_value=300,value=int(v.get("hr",0)) or None,placeholder="76")
                temp=st.number_input(t("temp"),min_value=0.0,max_value=45.0,value=float(v.get("temp",0.0)) or None,placeholder="36.6",format="%.1f")
            with cc2:
                spo2=st.number_input(t("spo2"),min_value=0,max_value=100,value=int(v.get("spo2",0)) or None,placeholder="98")
                bp_col1, bp_col2 = st.columns(2)
                with bp_col1:
                    bp_s=st.number_input(t("bp_sys"),min_value=0,max_value=300,value=int(v.get("bp_sys",0)) or None,placeholder="120")
                with bp_col2:
                    bp_d=st.number_input(t("bp_dia"),min_value=0,max_value=200,value=int(v.get("bp_dia",0)) or None,placeholder="80")

        with st.expander("＋ " + ("Αναπνοή, βάρος, ύψος (προαιρετικά)" if lang=="el" else "Breathing, weight, height (optional)")):
            ec1, ec2, ec3 = st.columns(3)
            with ec1:
                br=st.number_input(t("br"),min_value=0,max_value=60,value=int(v.get("br",0)) or None,placeholder="15")
            with ec2:
                weight=st.number_input(t("weight"),min_value=0.0,max_value=300.0,value=float(v.get("weight",0.0)) or None,placeholder="75",format="%.1f")
            with ec3:
                height=st.number_input(t("height"),min_value=0,max_value=250,value=int(v.get("height",0)) or None,placeholder="175")

        if st.button(t("analyse_vitals"),type="primary",use_container_width=True,key="analyse_manual"):
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
            st.session_state.vitals=vd; classify_vitals(vd, age=p.get("age"))
            if vd:
                with st.spinner("Ανάλυση..."):
                    vtext="\n".join(f"- {k}: {val}" for k,val in vd.items())
                    pp=p.get
                    st.session_state.vitals_analysis=claude([{"role":"user","content":f"Patient: {pp('name')}, {pp('age')}yo {pp('sex')}, Hx: {pp('history','none')}, Meds: {pp('meds_raw','none')}\n\nVitals:\n{vtext}\n\nInterpret. Categorise each. Flag urgent findings. Be direct."}],system=kira_system(),max_tokens=1200)
            st.session_state.screen="triage"; st.rerun()

    # ── BP Estimation — Railway GPR API + Demographic fallback ───────────────
    st.divider()
    pr = st.session_state.profile
    age_val  = pr.get("age", 0)
    v_now    = st.session_state.vitals
    hr_val   = v_now.get("hr")
    wt_val   = v_now.get("weight") or pr.get("weight")
    ht_val   = v_now.get("height") or pr.get("height")
    bmi_val  = v_now.get("bmi")
    if not bmi_val and wt_val and ht_val:
        bmi_val = round(wt_val / ((ht_val/100)**2), 1)
    sex_val  = pr.get("sex","")
    gender_n = 1 if sex_val in ["Άνδρας","Male"] else 0

    bp_api_url = _secret("BP_API_URL","")
    api_result = None

    # Try Railway GPR model first (real ML prediction)
    if bp_api_url and age_val >= 18 and wt_val and ht_val and hr_val:
        try:
            payload = json.dumps({
                "age": int(age_val), "height": float(ht_val),
                "weight": float(wt_val), "hr": int(hr_val),
                "gender": gender_n
            }).encode()
            req = urllib.request.Request(
                f"{bp_api_url.rstrip('/')}/predict",
                data=payload,
                headers={"Content-Type":"application/json"},
            )
            with urllib.request.urlopen(req, timeout=8) as r:
                api_result = json.loads(r.read())
        except Exception:
            api_result = None

    if age_val >= 18:
        risk = demographic_bp_risk(age_val, bmi_val, hr_val, wt_val, ht_val)
        label = risk["label_el"] if lang=="el" else risk["label_en"]
        note  = risk["note_el"]  if lang=="el" else risk["note_en"]
        color = risk["color"]

        if api_result:
            # ── ML model result (precise estimate with confidence interval) ──
            sbp     = api_result.get("sbp", "—")
            dbp     = api_result.get("dbp", "—")
            sbp_ci  = api_result.get("sbp_ci95", "")
            dbp_ci  = api_result.get("dbp_ci95", "")
            bmi_api = api_result.get("bmi", bmi_val or "—")
            title   = "Εκτίμηση Αρτηριακής Πίεσης — GPR Model" if lang=="el" else "Blood Pressure Estimate — GPR Model"
            subtitle= "Gaussian Process Regression · Chowdhury et al. (2020) · Railway API" if lang=="el" else "Gaussian Process Regression · Chowdhury et al. (2020) · Railway API"
            sbp_disp= f"{sbp} <span style='font-size:11px;color:#6B7280'>± {sbp_ci}</span>"
            dbp_disp= f"{dbp} <span style='font-size:11px;color:#6B7280'>± {dbp_ci}</span>"
            unit    = "mmHg"
            badge   = f"<div style='background:{color};color:white;padding:6px 14px;border-radius:20px;font-size:13px;font-weight:700'>{label}</div><div style='font-size:9px;color:#6B7280;text-align:right;margin-top:4px'>GPR Model ✓</div>"
        else:
            # ── Demographic fallback (range estimate) ──
            sbp_disp= risk["sbp"]
            dbp_disp= risk["dbp"]
            unit    = "mmHg"
            title   = "Εκτίμηση Κινδύνου Αρτηριακής Πίεσης" if lang=="el" else "Blood Pressure Risk Estimate"
            subtitle= "Βάσει: ηλικία, ΔΜΣ, HR — Chowdhury et al. (2020)" if lang=="el" else "Based on: age, BMI, HR — Chowdhury et al. (2020)"
            badge   = f"<div style='background:{color};color:white;padding:6px 14px;border-radius:20px;font-size:13px;font-weight:700'>{label}</div>"

        st.markdown(f"""
<div style="background:rgba(45,63,231,0.06);border:1px solid rgba(45,63,231,0.15);border-radius:14px;padding:18px 20px;">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
    <div>
      <div style="font-size:13px;font-weight:700;color:#1A1A2E">🩺 {title}</div>
      <div style="font-size:11px;color:#6B7280;margin-top:2px">{subtitle}</div>
    </div>
    {badge}
  </div>
  <div style="display:flex;gap:16px;margin-bottom:10px">
    <div style="background:white;border:1px solid #E0E5FF;border-radius:10px;padding:10px 16px;flex:1;text-align:center">
      <div style="font-size:11px;color:#6B7280">{"Εκτιμ. Συστολική" if lang=="el" else "Est. Systolic"}</div>
      <div style="font-size:20px;font-weight:700;color:{color}">{sbp_disp} <span style="font-size:12px;font-weight:400">{unit}</span></div>
    </div>
    <div style="background:white;border:1px solid #E0E5FF;border-radius:10px;padding:10px 16px;flex:1;text-align:center">
      <div style="font-size:11px;color:#6B7280">{"Εκτιμ. Διαστολική" if lang=="el" else "Est. Diastolic"}</div>
      <div style="font-size:20px;font-weight:700;color:{color}">{dbp_disp} <span style="font-size:12px;font-weight:400">{unit}</span></div>
    </div>
  </div>
  <div style="font-size:12px;color:#374151">{note}</div>
  <div style="font-size:10px;color:#9CA3AF;margin-top:6px">⚠️ {"Εκτίμηση μόνο — όχι αντικατάσταση πιεσομέτρου. Χρησιμοποιείστε πιστοποιημένο πιεσόμετρο για ακριβή μέτρηση." if lang=="el" else "Estimate only — not a substitute for a blood pressure monitor. Use a certified BP cuff for accurate measurement."}</div>
</div>
        """, unsafe_allow_html=True)

    # Navigation buttons
    col_b, col_s = st.columns([1, 3])
    with col_b:
        if st.button(t("back")): st.session_state.screen="intake"; st.rerun()
    with col_s:
        if st.button(("Δεν χρειάζομαι ζωτικά — Συνέχεια στα συμπτώματα →" if lang=="el"
                      else "I don't need vitals — Continue to symptoms →"), use_container_width=True):
            st.session_state.vitals={}; st.session_state.screen="triage"; st.rerun()

def render_vitals_summary():
    v=st.session_state.vitals
    if not v: return
    status=classify_vitals(v, age=st.session_state.profile.get("age"))
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

def render_photo_scan():
    """Photo health analysis (Florence-2 + Claude Vision). Lives inside the assessment."""
    p = st.session_state.profile
    lang = st.session_state.lang
    rf_key = _secret("ROBOFLOW_API_KEY","")
    st.caption(("Ανέβασε φωτογραφία για κλινική εκτίμηση" if lang=="el"
                else "Upload photo for clinical assessment"))

    SCAN_OPTS = {
        "el":[("eye","👁️ Μάτι"),("skin","🔬 Δέρμα/Εξάνθημα"),
              ("wound","🤕 Τραύμα/Πληγή"),("throat","🦷 Στόμα/Λαιμός"),
              ("nails","💅 Νύχια"),("body","🩹 Γενική Εμφάνιση")],
        "en":[("eye","👁️ Eye"),("skin","🔬 Skin/Rash"),
              ("wound","🤕 Wound"),("throat","🦷 Mouth/Throat"),
              ("nails","💅 Nails"),("body","🩹 Body/Lesion")],
    }
    opts   = SCAN_OPTS[lang]
    labels = [o[1] for o in opts]
    keys_  = [o[0] for o in opts]
    sel    = st.radio(("Τύπος σάρωσης" if lang=="el" else "Scan type"),
                      labels, horizontal=True, key="h_scan_type",
                      label_visibility="collapsed")
    scan_k = keys_[labels.index(sel)] if sel in labels else "skin"

    tips = {
        "eye":   {"el":"📸 Κοντά (10-15cm), καλό φωτισμό, ανοιχτό μάτι","en":"📸 Close-up (10-15cm), good light, eye open"},
        "skin":  {"el":"📸 Καθαρή εικόνα αλλοίωσης, φυσικό φωτισμό","en":"📸 Clear image of lesion, natural lighting"},
        "wound": {"el":"📸 Καλός φωτισμός, χωρίς αίμα να καλύπτει την πληγή","en":"📸 Good lighting, wound visible and clean"},
        "throat":{"el":"📸 Ανοιχτό στόμα, λαμπάκι αν υπάρχει","en":"📸 Open mouth, torch if available"},
        "nails": {"el":"📸 Κοντινή λήψη νυχιών σε λευκό φόντο","en":"📸 Close-up of nails on white background"},
        "body":  {"el":"📸 Ολόκληρη η πάσχουσα περιοχή στο κάδρο","en":"📸 Full affected area in frame"},
    }
    st.caption(tips.get(scan_k, tips["skin"])[lang])
    st.markdown(f'<div class="disclaimer">{"⚠️ Εργαλείο AI screening. Δεν αντικαθιστά κλινική εξέταση." if lang=="el" else "⚠️ AI screening tool. Does not replace clinical examination."}</div>', unsafe_allow_html=True)

    uploaded_photo = st.file_uploader(
        ("Φωτογραφία" if lang=="el" else "Upload photo"),
        type=["jpg","jpeg","png","webp","heic","heif"],
        key="human_photo_upload"
    )

    # Identity of the currently uploaded file — used to detect when the user
    # swaps to a different photo and we need to discard a stale preview.
    _current_file_id = (f"{uploaded_photo.name}|{uploaded_photo.size}|{scan_k}"
                        if uploaded_photo else None)

    # ── STAGE 1: Analyse button. Runs the vision pipeline and STORES the
    # result in session_state so it survives the rerun. Critically, the
    # 'Πρόσθεση στην εκτίμηση' button is NOT nested inside this if-block —
    # nested Streamlit buttons silently fail because the outer condition
    # becomes False on the next interaction.
    if uploaded_photo:
        c_img, c_info = st.columns([1,1])
        with c_img: st.image(uploaded_photo, use_container_width=True)
        with c_info:
            st.markdown(f"**{p.get('name','')}** · {sel}")
            if st.button("🔬 " + ("Ανάλυση" if lang=="el" else "Analyse"),
                         type="primary", use_container_width=True, key="analyse_human"):
                img_bytes = uploaded_photo.read()
                fname = uploaded_photo.name.lower()
                if fname.endswith((".heic",".heif")):
                    if HEIC_OK:
                        try: img_bytes, img_type = convert_heic_human(img_bytes)
                        except Exception as e: st.error(f"HEIC: {e}"); st.stop()
                    else:
                        st.error("Add pillow-heif to requirements.txt"); st.stop()
                else:
                    img_type = "image/jpeg"
                    if fname.endswith(".png"):  img_type = "image/png"
                    if fname.endswith(".webp"): img_type = "image/webp"

                img_b64 = _b64.b64encode(img_bytes).decode()

                with st.spinner("Ο Asklepios αναλύει τη φωτογραφία..." if lang=="el" else "Asklepios is analysing the photo..."):
                    f2_desc = ""
                    if rf_key:
                        f2 = florence2_human(img_b64, scan_k, rf_key)
                        if f2.get("ok"): f2_desc = f2.get("description","")

                    # Clinical context from the ongoing assessment so the photo is read
                    # WITHIN the reported complaint — not as an isolated, context-free image.
                    conv = st.session_state.triage_chat
                    convo_txt = "\n".join(
                        f"{'Ασθενής' if m['role']=='user' else 'Asklepios'}: {m['content']}"
                        for m in conv[-6:]
                    ) if conv else ("Δεν έχει καταγραφεί συνομιλία ακόμη." if lang=="el" else "No conversation yet.")
                    ctx_el = (f"ΚΛΙΝΙΚΟ ΠΛΑΙΣΙΟ (ο ασθενής έχει ΗΔΗ περιγράψει το πρόβλημα):\n"
                              f"Ασθενής: {p.get('age','?')} ετών, {p.get('sex','')}. Ιστορικό: {p.get('history','') or '—'}.\n"
                              f"Συζήτηση μέχρι τώρα:\n{convo_txt}\n\n"
                              f"Η φωτογραφία αφορά ΑΥΤΟ το παράπονο. Ερμήνευσέ την ΜΕΣΑ σε αυτό το πλαίσιο. "
                              f"ΜΗΝ αλλάζεις την ανατομική περιοχή ή το πρόβλημα που έχει ήδη περιγραφεί (π.χ. αν ο ασθενής λέει αγκώνας, μην το μετατρέπεις σε μασχάλη/θώρακα). "
                              f"ΜΗΝ εφευρίσκεις νέα διάγνωση ή νέο επίπεδο επείγοντος που έρχεται σε αντίθεση με την τρέχουσα εκτίμηση. "
                              f"Αν η εικόνα είναι ασαφής ή δεν προσθέτει κάτι, πες το ειλικρινά.")
                    ctx_en = (f"CLINICAL CONTEXT (the patient has ALREADY described the problem):\n"
                              f"Patient: {p.get('age','?')}yo {p.get('sex','')}. History: {p.get('history','') or '—'}.\n"
                              f"Conversation so far:\n{convo_txt}\n\n"
                              f"The photo relates to THIS complaint. Interpret it WITHIN this context. "
                              f"Do NOT change the anatomical region or the problem already described (e.g. if the patient says elbow, do not turn it into armpit/chest). "
                              f"Do NOT invent a new diagnosis or a new urgency level that contradicts the ongoing assessment. "
                              f"If the image is unclear or adds nothing, say so honestly.")
                    clin_ctx = (ctx_el if lang=="el" else ctx_en)

                    base_prompt = HUMAN_SCAN_PROMPTS.get(scan_k, HUMAN_SCAN_PROMPTS["skin"])
                    rf_context  = f"\n\nFLORENCE-2 DESCRIPTION: {f2_desc}" if f2_desc else ""
                    suffix_el   = "\n\nΔώσε ΣΥΜΠΛΗΡΩΜΑΤΙΚΑ ΟΠΤΙΚΑ ΕΥΡΗΜΑΤΑ (όχι ξεχωριστή διάγνωση): **ΟΡΑΤΑ ΕΥΡΗΜΑΤΑ** (μόνο ό,τι φαίνεται) | **ΣΥΜΒΑΤΟΤΗΤΑ με το παράπονο** (στηρίζει/δεν στηρίζει την τρέχουσα εκτίμηση) | **ΣΗΜΕΙΑ ΠΡΟΣΟΧΗΣ** (μόνο αν φαίνονται καθαρά στην εικόνα). Σύντομα και συνεπή με την τρέχουσα εκτίμηση."
                    suffix_en   = "\n\nGive SUPPLEMENTARY VISUAL FINDINGS (not a separate diagnosis): **VISIBLE FINDINGS** (only what is visible) | **CONSISTENCY with the complaint** (supports/does not support the current assessment) | **WARNING SIGNS** (only if clearly visible in the image). Brief and consistent with the current assessment."
                    full_prompt = clin_ctx + "\n\n" + base_prompt + rf_context + (suffix_el if lang=="el" else suffix_en)
                    sys_prompt  = ("Είσαι ο βοηθός οπτικής εξέτασης του Asklepios AI. Συμπληρώνεις μια εκτίμηση που ήδη εξελίσσεται — ΔΕΝ ξεκινάς νέα. Μένεις πιστός στο παράπονο και στην ανατομική περιοχή που έχει δηλωθεί, είσαι ακριβής, προσεκτικός και δεν δραματοποιείς." if lang=="el"
                                   else "You are Asklepios AI's visual-exam assistant. You SUPPLEMENT an assessment already in progress — you do NOT start a new one. Stay faithful to the stated complaint and anatomical region, be accurate, cautious, and do not dramatise.")
                    analysis = claude_vision_human(img_b64, img_type, full_prompt, sys_prompt)

                # Persist the preview so the next rerun renders Stage 2 at top
                # level — NOT nested inside this button block (which would die
                # on the next interaction).
                st.session_state["_photo_preview"] = {
                    "file_id":     _current_file_id,
                    "scan_type":   scan_k,
                    "scan_label":  sel,
                    "florence_desc": f2_desc,
                    "analysis":    analysis,
                }
                st.rerun()

    # ── STAGE 2: render the preview + 'Add to assessment' button at TOP LEVEL
    # (not nested), so the button actually fires on click.
    preview = st.session_state.get("_photo_preview")
    if preview:
        # If the user uploaded a different file or changed scan type, the old
        # preview is stale — discard it so they can re-analyse the new one.
        if uploaded_photo and preview.get("file_id") and preview["file_id"] != _current_file_id:
            st.session_state.pop("_photo_preview", None)
            preview = None
    if preview:
        analysis = preview["analysis"]
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(analysis)
        st.markdown('</div>', unsafe_allow_html=True)

        urgent_kw = ["urgent","immediate","επείγον","άμεσα","ιατρό αμέσως","emergency","melanoma","cancer","carcinoma","καρκίν"]
        if any(k.lower() in analysis.lower() for k in urgent_kw):
            st.error("🚨 " + ("Επείγοντα ευρήματα — επικοινωνήστε με ιατρό ΑΜΕΣΑ" if lang=="el"
                              else "Urgent findings — contact a doctor IMMEDIATELY"))

        if st.button("➤ " + ("Πρόσθεση στην εκτίμηση" if lang=="el" else "Add to assessment"),
                     type="primary", use_container_width=True, key="photo_to_triage_h"):
            _lbl = preview["scan_label"]
            msg = (f"Αποτέλεσμα φωτογραφικής ανάλυσης ({_lbl}):\n\n{analysis}"
                   if lang=="el" else
                   f"Photo analysis result ({_lbl}):\n\n{analysis}")
            st.session_state.triage_chat.append({"role":"user","content":msg})
            # Append to the photo findings LIST so multiple uploads accumulate
            # and all become visible in the final report.
            _pf = st.session_state.get("photo_findings")
            if not isinstance(_pf, list):
                _pf = []
            _pf.append({
                "scan_type":     preview["scan_type"],
                "scan_label":    preview["scan_label"],
                "florence_desc": preview.get("florence_desc",""),
                "analysis":      analysis,
            })
            st.session_state["photo_findings"] = _pf
            st.session_state["photo_added"]    = True
            st.session_state.pop("_photo_preview", None)
            st.rerun()
    elif not uploaded_photo:
        st.info("👆 " + ("Ανεβάστε φωτογραφία για να ξεκινήσει η ανάλυση" if lang=="el"
                        else "Upload a photo to begin analysis"))


def render_lab_analysis():
    """Lab PDF/image upload + Claude interpretation. Multi-file support:
    user can upload αιμοδιάγραμμα + βιοχημικό + ούρα in a single pass.
    Each file is analysed individually and added to the assessment.
    
    Privacy: files are sent to Claude API for analysis and discarded immediately.
    Nothing about the lab values is stored on our servers.
    """
    p = st.session_state.profile
    lang = st.session_state.lang
    st.caption(("Ανέβασε PDF ή φωτογραφία αιματολογικών, ορμονολογικών, βιοχημικών ή ουρολογικών εξετάσεων."
                if lang=="el" else
                "Upload PDF or photo of blood, hormonal, biochemistry, or urinalysis results."))
    st.markdown(f'<div class="disclaimer">{"⚠️ Εκπαιδευτικό εργαλείο, δεν αντικαθιστά ιατρό. Τα αρχεία δεν αποθηκεύονται στους server μας." if lang=="el" else "⚠️ Educational tool, does not replace a doctor. Files are not stored on our servers."}</div>', unsafe_allow_html=True)

    lab_files = st.file_uploader(
        ("Εξετάσεις (PDF, JPG, PNG — πολλαπλά αρχεία)" if lang=="el"
         else "Lab tests (PDF, JPG, PNG — multiple files)"),
        type=["pdf","jpg","jpeg","png","webp","heic","heif"],
        key="lab_upload",
        accept_multiple_files=True,
        help=("Μπορείς να ανεβάσεις πολλαπλά αρχεία ταυτόχρονα — π.χ. αιμοδιάγραμμα + βιοχημικό + ορμόνες."
              if lang=="el" else
              "Upload multiple files at once — e.g. CBC + biochemistry + hormonal panel.")
    )

    if lab_files:
        st.caption((f"📎 {len(lab_files)} αρχεία: " if lang=="el" else f"📎 {len(lab_files)} files: ")
                   + ", ".join(f.name for f in lab_files))

        # Already-analysed filenames (skip re-analysis on rerun)
        _already = {lf.get("file_name", "") for lf in (st.session_state.get("lab_findings") or [])}
        _to_run  = [f for f in lab_files if f.name not in _already]

        btn_lbl = (f"🔬 Ανάλυση {len(_to_run)} αρχείων" if len(_to_run) != 1
                   else "🔬 Ανάλυση εξέτασης") if lang == "el" else (
                   f"🔬 Analyse {len(_to_run)} files" if len(_to_run) != 1
                   else "🔬 Analyse lab result")

        if _to_run:
            if st.button(btn_lbl, type="primary", use_container_width=True, key="analyse_lab"):
                _added = 0
                status_msg = ("Ανάλυση εξετάσεων…" if lang=="el" else "Analysing lab results…")
                with st.status(status_msg, expanded=True) as _stat:
                    for idx, lab_file in enumerate(_to_run, 1):
                        _stat.update(label=(f"📄 ({idx}/{len(_to_run)}) {lab_file.name}"))
                        file_bytes = lab_file.read()
                        fname_lower = lab_file.name.lower()

                        # MIME type detection + HEIC conversion
                        if fname_lower.endswith((".heic", ".heif")):
                            if HEIC_OK:
                                try:
                                    file_bytes, mime = convert_heic_human(file_bytes)
                                except Exception as e:
                                    st.error(f"HEIC conversion failed for {lab_file.name}: {e}")
                                    continue
                            else:
                                st.error("⚠️ Οι φωτογραφίες HEIC χρειάζονται pillow-heif." if lang=="el"
                                         else "⚠️ HEIC photos need pillow-heif.")
                                continue
                        elif fname_lower.endswith(".pdf"):   mime = "application/pdf"
                        elif fname_lower.endswith(".png"):   mime = "image/png"
                        elif fname_lower.endswith(".webp"):  mime = "image/webp"
                        else:                                mime = "image/jpeg"

                        if not file_bytes:
                            continue

                        try:
                            analysis = claude_analyze_lab(
                                file_bytes, mime, p,
                                st.session_state.triage_chat, lang,
                                file_name=lab_file.name,
                            )
                        except Exception as e:
                            st.error(f"⚠️ {lab_file.name}: {e}")
                            continue

                        st.markdown(f"#### 📄 {lab_file.name}")
                        st.markdown(analysis)

                        # Add to findings + inject into triage chat
                        _lf = st.session_state.get("lab_findings")
                        if not isinstance(_lf, list):
                            _lf = []
                        _lf.append({"file_name": lab_file.name, "analysis": analysis})
                        st.session_state["lab_findings"] = _lf

                        msg = (f"Αποτέλεσμα ανάλυσης εξετάσεων ({lab_file.name}):\n\n{analysis}"
                               if lang=="el" else
                               f"Lab analysis result ({lab_file.name}):\n\n{analysis}")
                        st.session_state.triage_chat.append({"role": "user", "content": msg})
                        _added += 1

                    _final = (f"✅ Ολοκληρώθηκαν {_added}/{len(_to_run)} εξετάσεις" if lang=="el"
                              else f"✅ Completed {_added}/{len(_to_run)} files")
                    _stat.update(label=_final, state="complete", expanded=False)

                if _added:
                    st.session_state["lab_added"] = True
                    st.success("✅ " + (f"Προστέθηκαν {_added} εξετάσεις στην εκτίμηση."
                                       if lang=="el" else
                                       f"Added {_added} lab result(s) to the assessment."))
                    st.rerun()
        else:
            st.info("ℹ️ " + ("Όλα τα αρχεία έχουν ήδη αναλυθεί." if lang=="el"
                              else "All files have already been analysed."))

    if st.session_state.get("lab_findings"):
        _lf_names = [lf.get("file_name","") for lf in st.session_state["lab_findings"]]
        st.caption(("✅ Αναλύθηκαν: " if lang=="el" else "✅ Analysed: ")
                   + ", ".join(_lf_names))
    elif not lab_files:
        st.info("👆 " + ("Ανεβάστε PDF ή φωτογραφία για να ξεκινήσει η ανάλυση"
                         if lang=="el" else
                         "Upload a PDF or photo to begin analysis"))


def render_triage():
    render_stepper("triage")
    p=st.session_state.profile
    nm = p.get("name","")
    render_doc_header(
        "Ας μιλήσουμε για τα συμπτώματα", "Let's talk about your symptoms",
        icon="💬",
        sub_el=(f"συνομιλία με {nm}" if nm else "Πες τι σε απασχολεί — μία ερώτηση κάθε φορά"),
        sub_en=(f"chat with {nm}" if nm else "Tell me what's bothering you — one question at a time"),
    )
    render_vitals_summary()
    st.markdown(f'<div class="disclaimer">{t("disclaimer_main")}</div>',unsafe_allow_html=True)
    # Symptom quick-select: only BEFORE the conversation starts, so once chatting
    # the previous Q&A stays visible instead of being buried under the buttons.
    if not st.session_state.triage_chat:
        st.info(("👇 Βήμα 3 — Περίγραψε εδώ τι σε απασχολεί (π.χ. «πόνος στο μάτι 2 μέρες»). "
                 "Ο Asklepios θα σου κάνει ερωτήσεις και στο τέλος θα δημιουργήσει αναφορά."
                 if st.session_state.lang=="el" else
                 "👇 Step 3 — Describe what's bothering you (e.g. 'eye pain for 2 days'). "
                 "Asklepios will ask follow-up questions and then generate a report."))
        chips, _chips_label = _symptom_chips(st.session_state.profile, st.session_state.lang)
        _cap = ("Γρήγορη επιλογή" if st.session_state.lang=="el" else "Quick select")
        if _chips_label:
            _cap += f" ({_chips_label})"
        st.caption(_cap + ":")
        # Wrap-flow pill layout (matches the mockup) instead of st.columns rows,
        # which stack vertically on narrow/mobile viewports. Same marker+:has()
        # CSS targeting pattern already proven reliable for the bottom nav.
        st.markdown("""
<style>
div[data-testid="stHorizontalBlock"]:has(.chip-flow-marker) {
  flex-wrap: wrap !important; gap: 7px !important;
}
div[data-testid="stHorizontalBlock"]:has(.chip-flow-marker) > div[data-testid="stColumn"] {
  width: auto !important; min-width: 0 !important; flex: 0 0 auto !important;
}
div[data-testid="stHorizontalBlock"]:has(.chip-flow-marker) button {
  border-radius: 18px !important; padding: 8px 15px !important;
  font-size: 12.5px !important; font-weight: 700 !important;
  white-space: nowrap !important; min-height: 0 !important;
}
</style>
""", unsafe_allow_html=True)
        _chip_cols = st.columns(len(chips))
        for _i, (chip, _col) in enumerate(zip(chips, _chip_cols)):
            with _col:
                if _i == 0:
                    st.markdown('<div class="chip-flow-marker"></div>', unsafe_allow_html=True)
                sel = chip in st.session_state.symptom_chips
                if st.button(("✓ " if sel else "")+chip, key=f"chip_{_i}"):
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
    # Context-aware vitals: suggest the SPECIFIC measurement that fits the symptoms.
    # Scan button appears only for the cardiac category (camera → heart rate only).
    _lang = st.session_state.lang
    _relv = _relevant_vitals()
    if (any(m["role"]=="assistant" for m in st.session_state.triage_chat)
            and not st.session_state.vitals
            and _relv
            and not st.session_state.get("_vitals_nudge_off")):
        _names = ", ".join(dict.fromkeys(c["el" if _lang=="el" else "en"] for c in _relv))
        _show_scan = any(c["scan"] for c in _relv)
        st.warning("🩺 " + (f"Με βάση όσα περιγράφεις, θα βοηθούσε να μετρηθεί: {_names}. Θες να το κάνεις τώρα;"
                            if _lang=="el" else
                            f"Based on what you describe, it would help to measure: {_names}. Want to do it now?"))
        _cols = st.columns(3 if _show_scan else 2)
        with _cols[0]:
            if st.button(("✏️ Καταχώρηση" if _lang=="el" else "✏️ Enter values"), key="nudge_manual", use_container_width=True):
                st.session_state.screen = "vitals"; st.rerun()
        _ci = 1
        if _show_scan:
            with _cols[_ci]:
                _fs = _secret("FACESCAN_URL","https://asklepiosnurse.netlify.app")
                _ku = _secret("ASKLEPIOS_URL","https://asklepiosainurse.up.railway.app")
                _link = f"{_fs}?kira_url={urllib.parse.quote(_ku)}"
                _save_session_for_external_nav()
                st.markdown(f'<a href="{_link}" target="_blank" style="display:block;text-align:center;padding:8px;border-radius:8px;background:#2D3FE7;color:white;text-decoration:none;font-weight:600;font-size:13px">📷 {"Σάρωση" if _lang=="el" else "Scan"}</a>', unsafe_allow_html=True)
            _ci += 1
        with _cols[_ci]:
            if st.button(("Όχι τώρα" if _lang=="el" else "Not now"), key="nudge_off", use_container_width=True):
                st.session_state["_vitals_nudge_off"] = True; st.rerun()
    # Photo analysis appears only after an initial assessment AND only when the
    # complaint is something visible (skin, eye, wound, throat, nails...). For
    # non-visual issues (e.g. chest pain) a photo adds nothing, so it stays hidden.
    if any(m["role"]=="assistant" for m in st.session_state.triage_chat) and _visual_relevant():
        _pf_list = st.session_state.get("photo_findings") or []
        _has_photo = isinstance(_pf_list, list) and len(_pf_list) > 0
        # Label adapts so the user knows multiple uploads are allowed.
        # Collapsed by default once a photo has been added — keeps the chat
        # uncluttered but the option remains one click away.
        _exp_label = (("📷 Ανέβασε άλλη φωτογραφία (αν χρειαστεί)"
                       if _has_photo else
                       "📷 Ανάλυση φωτογραφίας (προαιρετικό)")
                      if st.session_state.lang=="el" else
                      ("📷 Upload another photo (if needed)"
                       if _has_photo else
                       "📷 Photo analysis (optional)"))
        with st.expander(_exp_label, expanded=not _has_photo):
            if _has_photo:
                st.caption("💡 " + (f"Έχουν προστεθεί {len(_pf_list)} φωτογραφία/ες. "
                                    "Ανέβασε νέα μόνο αν ο Asklepios το ζητήσει "
                                    "ή αν θέλεις άλλη πλευρά / άλλο σημείο."
                                    if st.session_state.lang=="el" else
                                    f"{len(_pf_list)} photo(s) already added. "
                                    "Upload a new one only if Asklepios asks "
                                    "or you want a different angle/area."))
            else:
                st.caption("💡 " + ("Προαιρετικό. Αν ο Asklepios χρειαστεί φωτογραφία για ορατό σύμπτωμα, "
                                    "θα στο αναφέρει — αλλά μπορείς να ανεβάσεις και προληπτικά."
                                    if st.session_state.lang=="el" else
                                    "Optional. If Asklepios needs a photo for a visible symptom, "
                                    "it will say so — but you can also upload proactively."))
            render_photo_scan()
    # Physiotherapy card — surfaces proactively as soon as the conversation
    # (Physio and psychology cards removed — no dedicated API available.)
    # Lab analysis — always available once Asklepios has started talking, since
    # blood/hormonal/urinalysis results help for ANY complaint, not just visual.
    if any(m["role"]=="assistant" for m in st.session_state.triage_chat):
        _lf_list = st.session_state.get("lab_findings") or []
        _has_lab = isinstance(_lf_list, list) and len(_lf_list) > 0
        _lab_label = (("🧪 Ανέβασε άλλες εξετάσεις (αν χρειάζεται)"
                       if _has_lab else
                       "🧪 Ανάλυση εξετάσεων (αιματολογικά, ορμονολογικά, ούρα) — προαιρετικό")
                      if st.session_state.lang=="el" else
                      ("🧪 Upload more lab tests (if needed)"
                       if _has_lab else
                       "🧪 Lab analysis (blood, hormonal, urinalysis) — optional"))
        with st.expander(_lab_label, expanded=False):
            if _has_lab:
                st.caption("💡 " + (f"{len(_lf_list)} αρχείο/α εξετάσεων έχουν προστεθεί. "
                                    "Ανέβασε άλλο αν έχεις περισσότερες εξετάσεις."
                                    if st.session_state.lang=="el" else
                                    f"{len(_lf_list)} lab file(s) added. "
                                    "Upload another if you have more tests."))
            else:
                st.caption("💡 " + ("Ανέβασε εργαστηριακές εξετάσεις (PDF ή φωτογραφία) "
                                    "και ο Asklepios θα τις ερμηνεύσει ΜΕΣΑ στο πλαίσιο των συμπτωμάτων σου."
                                    if st.session_state.lang=="el" else
                                    "Upload lab tests (PDF or photo) and Asklepios will "
                                    "interpret them WITHIN the context of your symptoms."))
            render_lab_analysis()
    # Confirmation after a photo was added — guide the user to keep answering
    if st.session_state.get("photo_added"):
        last_q = next((m["content"] for m in reversed(st.session_state.triage_chat) if m["role"]=="assistant"), "")
        if st.session_state.lang=="el":
            st.success("✅ Η ανάλυση της εικόνας προστέθηκε στην εκτίμηση. Συνέχισε απαντώντας στην τελευταία ερώτηση του Asklepios παρακάτω.")
        else:
            st.success("✅ The image analysis was added to the assessment. Continue by answering Asklepios's last question below.")
        if last_q:
            st.info(("🩺 Τελευταία ερώτηση: " if st.session_state.lang=="el" else "🩺 Last question: ") + last_q)
    # Same confirmation pattern for lab results — keeps the user on track
    if st.session_state.get("lab_added"):
        last_q = next((m["content"] for m in reversed(st.session_state.triage_chat) if m["role"]=="assistant"), "")
        if st.session_state.lang=="el":
            st.success("✅ Η ανάλυση των εξετάσεων προστέθηκε στην εκτίμηση. Συνέχισε απαντώντας στον Asklepios.")
        else:
            st.success("✅ The lab analysis was added to the assessment. Continue chatting with Asklepios.")
    ready_phrases=["έχω αρκετά στοιχεία","μπορούμε να δημιουργήσουμε","i have enough information","we can generate","full clinical report","πλήρη αναφορά"]
    last_kira=next((m["content"].lower() for m in reversed(st.session_state.triage_chat) if m["role"]=="assistant"),"")
    triage_ready=any(ph in last_kira for ph in ready_phrases)
    # ── Voice input ───────────────────────────────────────────────────────────
    # Always shown — Tab 1 (Web Speech API) needs no API key at all.
    # Tab 2 (Whisper) needs Groq or OpenAI key.
    # Critical for 60+ demographic: IOBE data shows this group has the highest
    # unmet healthcare needs and lowest digital comfort.
    _voice_lbl = ("🎤 Φωνητική εισαγωγή (μίλα αντί να γράφεις)"
                  if st.session_state.lang=="el" else
                  "🎤 Voice input (speak instead of typing)")
    with st.expander(_voice_lbl, expanded=False):
        # Web Speech API — uses st.iframe (HTML string mode)
        _has_stt = bool(get_groq_key() or get_openai_key())
        _whisper_tab_lbl = ("🎙️ Whisper AI (Ελληνικά ✓)"
                            if _has_stt else
                            "🎙️ Whisper AI (απαιτεί OPENAI_API_KEY)")
        _wsapi_tab_lbl = ("🌐 Browser (δωρεάν, Chrome/Safari)"
                          if st.session_state.lang=="el" else
                          "🌐 Browser (free, Chrome/Safari)")
        _v_tab1, _v_tab2 = st.tabs([_whisper_tab_lbl, _wsapi_tab_lbl])

        # ── Tab 1: st.audio_input + Whisper ──────────────────────────────────
        with _v_tab1:
            if not _has_stt:
                st.info("💡 " + ("Πρόσθεσε `OPENAI_API_KEY` ή `GROQ_API_KEY` στα Railway env vars για να ενεργοποιήσεις το Whisper."
                                 if st.session_state.lang=="el" else
                                 "Add `OPENAI_API_KEY` or `GROQ_API_KEY` to Railway env vars to enable Whisper."))
            else:
                st.caption("💡 " + ("Πάτησε το μικρόφωνο, μίλα φυσικά, σταμάτα. Η ηχογράφηση δεν αποθηκεύεται."
                                    if st.session_state.lang=="el" else
                                    "Press the microphone, speak naturally, stop. Audio is not stored."))
                _audio = st.audio_input(
                    ("Πες τι νιώθεις" if st.session_state.lang=="el" else "Say what you feel"),
                    key=f"voice_input_widget_{st.session_state.get('_voice_widget_counter', 0)}",
                    label_visibility="collapsed",
                )
                # Track by hash so the same audio isn't transcribed twice across reruns
                if _audio is not None:
                    _audio_bytes = _audio.getvalue()
                    _audio_hash = hashlib.sha256(_audio_bytes).hexdigest()[:16]
                    if st.session_state.get("_voice_last_hash") != _audio_hash:
                        with st.spinner("🎙️ " + ("Μεταγραφή με Whisper..." if st.session_state.lang=="el"
                                                  else "Transcribing with Whisper...")):
                            text, _ = transcribe_audio(
                                _audio_bytes, lang=st.session_state.lang,
                                mime="audio/webm", filename="voice.webm",
                            )
                        st.session_state["_voice_last_hash"] = _audio_hash
                        if text and not text.startswith("⚠️"):
                            st.session_state["_voice_transcript"] = text
                            st.rerun()
                        else:
                            st.error(text or "—")
                # Transcript review + confirm (never auto-submit — Whisper can mishear)
                _pending = st.session_state.get("_voice_transcript")
                if _pending:
                    st.success("📝 " + ("Μεταγραφή — διόρθωσε αν χρειαστεί:" if st.session_state.lang=="el"
                                        else "Transcription — edit if needed:"))
                    _edited = st.text_area("transcript_edit", value=_pending,
                                           label_visibility="collapsed", height=80,
                                           key="voice_edit_area")
                    _vc1, _vc2 = st.columns([3, 1])
                    with _vc1:
                        if st.button(("✓ Αποστολή στον Asklepios" if st.session_state.lang=="el"
                                      else "✓ Send to Asklepios"),
                                     type="primary", use_container_width=True, key="voice_send"):
                            _msg = _edited.strip() or _pending
                            st.session_state.triage_chat.append({"role":"user","content":_msg})
                            st.session_state.pop("photo_added", None)
                            st.session_state.pop("lab_added", None)
                            st.session_state.pop("_voice_transcript", None)
                            st.session_state.pop("_voice_last_hash", None)
                            # Increment counter → new key on next render → fresh widget, no error
                            st.session_state["_voice_widget_counter"] = st.session_state.get("_voice_widget_counter", 0) + 1
                            st.session_state["_voice_send_pending"] = True
                            st.rerun()
                    with _vc2:
                        if st.button(("🗑️ Ακύρωση" if st.session_state.lang=="el" else "🗑️ Cancel"),
                                     use_container_width=True, key="voice_cancel"):
                            st.session_state.pop("_voice_transcript", None)
                            st.session_state.pop("_voice_last_hash", None)
                            # Increment counter → fresh widget so user can record again
                            st.session_state["_voice_widget_counter"] = st.session_state.get("_voice_widget_counter", 0) + 1
                            st.rerun()

        # ── Tab 2: Web Speech API — browser-native, no API key, Greek support ─
        # Same pattern as HAL project. Works on Chrome/Safari.
        # Result shown below the widget for the user to copy → paste into chat.
        with _v_tab2:
            _ws_lang = "el-GR" if st.session_state.lang=="el" else "en-US"
            _ws_hint = ("Μίλα φυσικά — το κείμενο εμφανίζεται αυτόματα."
                        if st.session_state.lang=="el" else
                        "Speak naturally — text appears automatically.")
            _ws_copy_lbl = "📋 Αντιγραφή" if st.session_state.lang=="el" else "📋 Copy"
            _ws_not_sup = ("Δεν υποστηρίζεται — χρησιμοποίησε Chrome ή Safari"
                           if st.session_state.lang=="el" else
                           "Not supported — use Chrome or Safari")
            _ws_listening = "🔴 Ακούω..." if st.session_state.lang=="el" else "🔴 Listening..."
            _ws_idle = ("Πάτησε 🎙️ για ηχογράφηση" if st.session_state.lang=="el"
                        else "Press 🎙️ to record")
            _ws_done = ("✅ Αντίγραψε και επικόλλησε στο chat ↓"
                        if st.session_state.lang=="el" else
                        "✅ Copy and paste into chat ↓")
            st.iframe(f"""<!DOCTYPE html><html><head><style>
body{{margin:0;padding:0;font-family:system-ui,sans-serif;background:transparent}}
#wrap{{display:flex;align-items:flex-start;gap:10px;background:#F0F4FF;border:1px solid #C7D2FE;border-radius:10px;padding:10px 14px;flex-wrap:wrap}}
#mic{{background:none;border:2px solid #2D3FE7;border-radius:50%;width:38px;height:38px;font-size:18px;cursor:pointer;color:#2D3FE7;flex-shrink:0;transition:all .2s}}
#mic.active{{background:#2D3FE7;color:white;box-shadow:0 0 0 4px rgba(45,63,231,.15)}}
#status{{font-size:12px;color:#6B7280;flex:1;padding-top:10px}}
#result{{display:none;width:100%;background:white;border:1px solid #C7D2FE;border-radius:8px;padding:8px 12px;font-size:14px;color:#1F2937;line-height:1.5;margin-top:6px;word-break:break-word}}
#copy{{display:none;background:#2D3FE7;color:white;border:none;border-radius:8px;padding:8px 18px;font-weight:700;cursor:pointer;font-size:13px;margin-top:6px}}
#copy:hover{{background:#1E30CC}}
</style></head><body>
<div id="wrap">
  <button id="mic" onclick="toggleVoice()">🎙️</button>
  <div id="status">{_ws_idle}</div>
  <div id="result"></div>
  <button id="copy" onclick="copyText()">{_ws_copy_lbl}</button>
</div>
<script>
var recognition,listening=false,transcript="";
function toggleVoice(){{
  if(!("webkitSpeechRecognition"in window||"SpeechRecognition"in window)){{
    document.getElementById("status").textContent="{_ws_not_sup}";return;
  }}
  if(listening){{recognition.stop();return;}}
  recognition=new(window.SpeechRecognition||window.webkitSpeechRecognition)();
  recognition.lang="{_ws_lang}";recognition.interimResults=true;recognition.continuous=false;
  recognition.onstart=function(){{
    listening=true;
    document.getElementById("mic").classList.add("active");
    document.getElementById("status").textContent="{_ws_listening}";
    document.getElementById("result").style.display="none";
    document.getElementById("copy").style.display="none";
  }};
  recognition.onresult=function(e){{
    transcript=Array.from(e.results).map(r=>r[0].transcript).join("");
    document.getElementById("result").textContent=transcript;
    document.getElementById("result").style.display="block";
  }};
  recognition.onend=function(){{
    listening=false;
    document.getElementById("mic").classList.remove("active");
    if(transcript){{
      document.getElementById("status").textContent="{_ws_done}";
      document.getElementById("copy").style.display="inline-block";
    }}else{{
      document.getElementById("status").textContent="{_ws_idle}";
    }}
  }};
  recognition.onerror=function(e){{
    listening=false;
    document.getElementById("mic").classList.remove("active");
    document.getElementById("status").textContent="Error: "+e.error;
  }};
  recognition.start();
}}
function copyText(){{
  if(!transcript)return;
  navigator.clipboard.writeText(transcript).then(function(){{
    var b=document.getElementById("copy");
    b.textContent="✅ OK!";
    setTimeout(function(){{b.textContent="{_ws_copy_lbl}";}},2000);
  }});
}}
</script></body></html>""", height=100)
            st.caption("↑ " + ("Αντίγραψε το κείμενο και επικόλλησέ το στο chat παρακάτω."
                                if st.session_state.lang=="el" else
                                "Copy the text and paste it into the chat below."))

    user_input=st.chat_input(t("triage_placeholder"),key="triage_input")
    _auto_reply = st.session_state.pop("_scan_reply_pending", False)
    _voice_reply = st.session_state.pop("_voice_send_pending", False)
    if user_input or _auto_reply or _voice_reply:
        if user_input:
            st.session_state.pop("photo_added", None)
            st.session_state.pop("lab_added", None)
            st.session_state.triage_chat.append({"role":"user","content":user_input})
        with st.spinner("Asklepios..."):
            pp=p.get
            _flags = []
            if pp("pregnancy"):
                _flags.append("ΕΓΚΥΟΣ — πρόσεξε αντενδείξεις φαρμάκων/εξετάσεων κατηγορίας D/X" if st.session_state.lang=="el"
                              else "PREGNANT — flag drug/test contraindications (Category D/X)")
            if pp("for_whom") == "other":
                _flags.append("Αξιολόγηση από φροντιστή για άλλο άτομο" if st.session_state.lang=="el"
                              else "Caregiver-mode: user is asking on behalf of another person")
            _age_v = pp("age", 0) or 0
            if _age_v < 18:
                _flags.append(f"ΠΑΙΔΙΑΤΡΙΚΟΣ ΑΣΘΕΝΗΣ (ηλικία {_age_v}) — χρησιμοποίησε παιδιατρικές δόσεις/όρια"
                              if st.session_state.lang=="el" else
                              f"PEDIATRIC PATIENT (age {_age_v}) — use pediatric dosing/ranges")
            _flags_str = (" | ".join(_flags) + " | ") if _flags else ""
            profile_ctx=f"Patient: {pp('name')}, {pp('age')}yo {pp('sex')}, {_flags_str}Hx: {pp('history','none')}, Allergies: {pp('allergies','none')}, Meds: {pp('meds_raw','none')}"
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
    # Report-language selector: ask only the clinically relevant question.
    # The UI stays in the chosen app language. The report is generated in that
    # same language by default. If the user wants a Greek copy for a Greek
    # doctor, they tick this — the report is then generated in Greek regardless
    # of the UI language (useful for non-Greek-speaking users living in Greece).
    _lang = st.session_state.lang
    if _lang != "el":
        _also_greek = st.checkbox(
            "📋 Δημιούργησε την αναφορά και στα **Ελληνικά** (για να τη δείξεις σε Έλληνα ιατρό)",
            value=st.session_state.get("report_also_greek", False),
            key="report_also_greek_cb",
        )
        if _also_greek != st.session_state.get("report_also_greek", False):
            st.session_state["report_also_greek"] = _also_greek
    if not enabled:
        st.caption("Συνεχίστε — ο Asklepios θα σας ειδοποιήσει όταν έχει αρκετά." if _lang=="el" else "Continue — Asklepios will let you know when it has enough.")

# ── PNOE-inspired report helpers ──────────────────────────────────────────────
# Inspired by the PNOE Metabolic Blueprint report (Frank Shallenberger), which
# packages each recommendation block as three categories (EXERCISE / NUTRITION /
# LIFESTYLE) and uses a 5-level scale. We adapt both ideas:
#   1. Claude is asked to emit a delimited RECS block at the end of the report
#      with three personalised buckets. We parse it out and render as a styled
#      3-column card (PDF/TXT/WhatsApp also get the clean text).
#   2. The existing Wellness Score is augmented with a 5-segment scale bar
#      (Severe Limit. → Limit. → Neutral → Good → Excellent) matching PNOE's
#      visual language.

def _extract_recs(report_text):
    """Pull <<<RECS ... RECS>>> block out of the Claude report.
    Returns (cleaned_text, recs_dict_or_None). Graceful: if no block found,
    returns the original text unchanged and None."""
    import re as _re_r
    if not report_text:
        return report_text, None
    m = _re_r.search(r"<<<RECS\s*(.*?)\s*RECS>>>", report_text, _re_r.DOTALL)
    if not m:
        return report_text, None
    block = m.group(1)
    cleaned = (report_text[:m.start()].rstrip() + "\n\n" + report_text[m.end():].lstrip()).strip()
    recs = {}
    # Multi-line tolerant: accumulate until next label or end
    current = None
    for line in block.splitlines():
        s = line.strip()
        if not s: continue
        upper = s.upper()
        for tag, key in (("CONDITION:", "condition"),
                         ("EXERCISE:", "exercise"),
                         ("NUTRITION:", "nutrition"),
                         ("LIFESTYLE:", "lifestyle")):
            if upper.startswith(tag):
                current = key
                recs[key] = s[len(tag):].strip()
                break
        else:
            # Continuation line
            if current:
                recs[current] = (recs.get(current, "") + " " + s).strip()
    return cleaned, (recs if recs else None)


def _render_recs_card(recs, lang, refs=None):
    """3-column Exercise/Nutrition/Lifestyle card (PNOE-style), with per-pillar
    PubMed references rendered as small links under each column when available."""
    if not recs:
        return
    if lang == "el":
        tx = {
            "title":   "📍 ΕΞΑΤΟΜΙΚΕΥΜΕΝΕΣ ΣΥΣΤΑΣΕΙΣ",
            "ex_lbl":  "ΦΥΣΙΚΗ ΔΡΑΣΤΗΡΙΟΤΗΤΑ",
            "nu_lbl":  "ΔΙΑΤΡΟΦΗ",
            "li_lbl":  "ΤΡΟΠΟΣ ΖΩΗΣ",
            "refs":    "Οδηγίες & μετα-αναλύσεις",
        }
    else:
        tx = {
            "title":   "📍 PERSONALISED RECOMMENDATIONS",
            "ex_lbl":  "EXERCISE",
            "nu_lbl":  "NUTRITION",
            "li_lbl":  "LIFESTYLE",
            "refs":    "Guidelines & meta-analyses",
        }
    import html as _html_r, re as _re_rec
    # Collapse internal whitespace BEFORE escaping. Newlines in recs content
    # break Streamlit's markdown HTML mode → raw </div> tags leak as text
    # (the bug visible in the user's screenshot). Recs are short prose, so a
    # single-line collapse is safe and preserves readability.
    def _flat(t): return _re_rec.sub(r"\s+", " ", (t or "—").strip()) or "—"
    ex = _html_r.escape(_flat(recs.get("exercise")))
    nu = _html_r.escape(_flat(recs.get("nutrition")))
    li = _html_r.escape(_flat(recs.get("lifestyle")))

    def _refs_html(pillar_key):
        items = (refs or {}).get(pillar_key) or []
        if not items:
            return ""
        lis = "".join(
            f'<li><a href="{_html_r.escape(r.get("url",""))}" target="_blank" '
            f'style="color:#1E40AF;text-decoration:none">'
            f'{_html_r.escape((r.get("title","—") or "")[:120])}'
            f'</a><span style="color:#9CA3AF"> · {_html_r.escape(r.get("journal","") or "")}'
            f'{(" " + _html_r.escape(r.get("date","")[:4])) if r.get("date") else ""}</span></li>'
            for r in items
        )
        return (
            f'<div class="pnoe-refs">'
            f'<div class="pnoe-refs-lbl">📚 {tx["refs"]}</div>'
            f'<ul>{lis}</ul>'
            f'</div>'
        )

    st.markdown(f"""
<style>
.pnoe-recs {{
  background: white;
  border: 1px solid #E0E5FF;
  border-radius: 22px;
  padding: 24px 24px 22px;
  margin: 18px 0;
  font-family: 'Inter', system-ui, sans-serif;
  box-shadow: 0 2px 10px rgba(45,63,231,0.05);
}}
.pnoe-recs-title {{
  font-size: 13.5px; font-weight: 800; letter-spacing: 0.01em;
  color: #1A1A2E;
  border-bottom: 1px solid #EEF1FC;
  padding-bottom: 14px; margin-bottom: 16px;
}}
.pnoe-recs-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 14px;
}}
.pnoe-recs-col {{
  border-radius: 16px;
  padding: 16px 16px 18px;
  border: 1px solid;
}}
.pnoe-recs-col.exercise  {{ background: #EFF6FF; border-color: #BFDBFE; }}
.pnoe-recs-col.nutrition {{ background: #ECFDF5; border-color: #A7F3D0; }}
.pnoe-recs-col.lifestyle {{ background: #FEF3F2; border-color: #FECDD3; }}
.pnoe-recs-head {{
  display: flex; align-items: center; gap: 9px;
  margin-bottom: 10px;
}}
.pnoe-recs-icon {{
  width: 28px; height: 28px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 14px; flex-shrink: 0;
}}
.pnoe-recs-col.exercise  .pnoe-recs-icon {{ background: #DBEAFE; }}
.pnoe-recs-col.nutrition .pnoe-recs-icon {{ background: #D1FAE5; }}
.pnoe-recs-col.lifestyle .pnoe-recs-icon {{ background: #FEE2E2; }}
.pnoe-recs-label {{
  font-size: 11px; font-weight: 800;
  letter-spacing: 0.10em; text-transform: uppercase;
  color: #1F2937;
}}
.pnoe-recs-body {{
  font-size: 13px; color: #374151; line-height: 1.6;
}}
.pnoe-refs {{
  margin-top: 12px; padding-top: 10px;
  border-top: 1px dashed rgba(0,0,0,0.10);
}}
.pnoe-refs-lbl {{
  font-size: 10px; font-weight: 700; letter-spacing: 0.10em;
  color: #6B7280; text-transform: uppercase; margin-bottom: 5px;
}}
.pnoe-refs ul {{
  list-style: none; padding: 0; margin: 0;
}}
.pnoe-refs li {{
  font-size: 11.5px; line-height: 1.45; margin-bottom: 5px;
  color: #374151;
}}
.pnoe-refs a:hover {{ text-decoration: underline !important; }}
@media (max-width: 768px) {{
  .pnoe-recs-grid {{ grid-template-columns: 1fr; gap: 11px; }}
  .pnoe-recs {{ padding: 20px 18px; }}
}}
</style>
<div class="pnoe-recs">
<div class="pnoe-recs-title">{tx['title']}</div>
<div class="pnoe-recs-grid">
<div class="pnoe-recs-col exercise">
<div class="pnoe-recs-head"><span class="pnoe-recs-icon">🏃</span><span class="pnoe-recs-label">{tx['ex_lbl']}</span></div>
<div class="pnoe-recs-body">{ex}</div>
{_refs_html("exercise")}
</div>
<div class="pnoe-recs-col nutrition">
<div class="pnoe-recs-head"><span class="pnoe-recs-icon">🥗</span><span class="pnoe-recs-label">{tx['nu_lbl']}</span></div>
<div class="pnoe-recs-body">{nu}</div>
{_refs_html("nutrition")}
</div>
<div class="pnoe-recs-col lifestyle">
<div class="pnoe-recs-head"><span class="pnoe-recs-icon">🌿</span><span class="pnoe-recs-label">{tx['li_lbl']}</span></div>
<div class="pnoe-recs-body">{li}</div>
{_refs_html("lifestyle")}
</div>
</div>
</div>
""", unsafe_allow_html=True)


def _compute_health_pillars(profile, vitals, status_map, report_text, lang):
    """Return (pillars_list, overall_score). Each pillar has a score 0-100 or
    None when no data was available. The overall is the mean of pillars that
    had data — never invented for missing inputs."""
    age = profile.get("age", 0) or 0
    # Accent-stripped so Greek history like "Υπέρταση" matches the "υπερτασ" pattern
    history = _strip_accents((profile.get("history") or "").lower())
    rep_low = _strip_accents((report_text or "").lower())

    def _ss(k):
        s = status_map.get(k)
        if s == "green":  return 100
        if s == "yellow": return 60
        if s == "red":    return 25
        return None

    # 1) 🫀 Cardiovascular: HR + BP + age/hypertension penalty
    cardio, cfact = [], []
    if _ss("hr") is not None:
        cardio.append(_ss("hr"));  cfact.append(f"HR {vitals.get('hr')}")
    if _ss("bp") is not None:
        cardio.append(_ss("bp"));  cfact.append(f"BP {vitals.get('bp_sys')}/{vitals.get('bp_dia')}")
    if cardio:
        sc = sum(cardio) / len(cardio)
        if age >= 75: sc = max(20, sc - 12); cfact.append("ηλικία ≥75" if lang=="el" else "age ≥75")
        elif age >= 65: sc = max(25, sc - 6)
        if any(w in history for w in ("υπερτασ","hypertens")):
            sc = max(20, sc - 8); cfact.append("ιστ. υπέρτασης" if lang=="el" else "hypertension hx")
        c_score = int(round(sc))
    else:
        c_score = None

    # 2) 🫁 Respiratory: SpO2 + BR + smoking/asthma flags
    resp, rfact = [], []
    if _ss("spo2") is not None:
        resp.append(_ss("spo2")); rfact.append(f"SpO₂ {vitals.get('spo2')}%")
    if _ss("br") is not None:
        resp.append(_ss("br"));   rfact.append(f"BR {vitals.get('br')}/min")
    if resp:
        sc = sum(resp) / len(resp)
        if any(w in history for w in ("καπν","smok","τσιγαρ")):
            sc = max(20, sc - 15); rfact.append("κάπνισμα" if lang=="el" else "smoking")
        if any(w in history for w in ("ασθμ","asthm","copd","χαπ")):
            sc = max(20, sc - 8);  rfact.append("ασθματικός" if lang=="el" else "asthma")
        r_score = int(round(sc))
    else:
        r_score = None

    # 3) ⚖️ Metabolic: BMI + temp + diabetes flag
    meta, mfact = [], []
    if _ss("bmi") is not None:
        meta.append(_ss("bmi"));  mfact.append(f"ΔΜΣ {vitals.get('bmi')}" if lang=="el" else f"BMI {vitals.get('bmi')}")
    if _ss("temp") is not None:
        meta.append(_ss("temp")); mfact.append(f"T {vitals.get('temp')}°C")
    if meta:
        sc = sum(meta) / len(meta)
        if any(w in history for w in ("διαβητ","diabet","τ2","t2")):
            sc = max(20, sc - 12); mfact.append("διαβήτης" if lang=="el" else "diabetes")
        m_score = int(round(sc))
    else:
        m_score = None

    # 4) 🩺 Symptom burden: from report content (red flags + severity terms)
    sb_score = 100
    sb_fact = []
    urgent = [_strip_accents(w) for w in
              ["επείγον","emergency","stroke","εγκεφαλικ","heart attack","έμφραγμα",
               "anaphylax","αναφυλαξ","unconscious","αναίσθητ","166","112"]]
    if any(w in rep_low for w in urgent):
        sb_score -= 50
        sb_fact.append("κόκκινες σημαίες" if lang=="el" else "red flags")
    severity = [_strip_accents(w) for w in
                ("σοβαρ","οξύς","έντον","severe","intense","acute")]
    if any(w in rep_low for w in severity):
        sb_score -= 12
        sb_fact.append("έντονα συμπτώματα" if lang=="el" else "intense symptoms")
    # Many differentials = more diagnostic uncertainty
    diff_rows = rep_low.count("|")
    if diff_rows >= 16:  # ≥4 rows in the markdown table
        sb_score -= 8
        sb_fact.append("πολλαπλές διαφορικές" if lang=="el" else "multiple differentials")
    sb_score = max(20, sb_score)
    if not sb_fact:
        sb_fact.append("ήπιο προφίλ" if lang=="el" else "mild profile")

    pillars = [
        {"key":"cardio","icon":"🫀",
         "label_el":"Καρδιαγγειακή","label_en":"Cardiovascular",
         "score":c_score,"factors":cfact,"available":c_score is not None},
        {"key":"resp","icon":"🫁",
         "label_el":"Αναπνευστική","label_en":"Respiratory",
         "score":r_score,"factors":rfact,"available":r_score is not None},
        {"key":"meta","icon":"⚖️",
         "label_el":"Μεταβολική","label_en":"Metabolic",
         "score":m_score,"factors":mfact,"available":m_score is not None},
        {"key":"symp","icon":"🩺",
         "label_el":"Συμπτωματικό φορτίο","label_en":"Symptom burden",
         "score":sb_score,"factors":sb_fact,"available":True},
    ]
    avail = [p for p in pillars if p["available"]]
    overall = int(round(sum(p["score"] for p in avail) / len(avail))) if avail else None
    return pillars, overall


def _grade_label(score, lang):
    """Map a 0-100 score to the PNOE 5-level grade label."""
    if score is None:
        return ("Δεν υπάρχουν δεδομένα", "#9CA3AF") if lang=="el" else ("No data", "#9CA3AF")
    if score >= 80: return (("Άριστο" if lang=="el" else "Excellent"), "#059669")
    if score >= 60: return (("Καλό"   if lang=="el" else "Good"),      "#10B981")
    if score >= 40: return (("Μέτριο" if lang=="el" else "Neutral"),   "#3B82F6")
    if score >= 20: return (("Χαμηλό" if lang=="el" else "Limited"),   "#F97316")
    return            (("Πολύ χαμηλό" if lang=="el" else "Severe limit."), "#DC2626")


def _pillar_scale_html(score):
    """A clean 5-segment scale (PNOE-style) for a pillar score, on white bg."""
    if score is None:
        return '<div style="height:10px;background:#F3F4F6;border-radius:5px;margin-top:6px"></div>'
    seg = max(0, min(4, int(score) // 20))
    colors = ["#DC2626","#F97316","#3B82F6","#10B981","#059669"]
    out = '<div style="display:flex;gap:4px;margin-top:6px">'
    for i in range(5):
        bg = colors[i] if i <= seg else "#E5E7EB"
        marker = "box-shadow:0 0 0 2px white inset" if i == seg else ""
        out += f'<div style="flex:1;height:10px;background:{bg};border-radius:5px;{marker}"></div>'
    out += '</div>'
    return out


def _render_health_pillars(profile, vitals, status_map, report_text, lang):
    """4-Pillar Health Profile card — replaces the placeholder wellness score
    with a transparent, factor-explained breakdown (PNOE 'Overview' inspired).
    Only shown when at least ONE measurement-based pillar (cardio/resp/meta)
    has data — symptom burden alone is not a 'health profile'."""
    pillars, overall = _compute_health_pillars(profile, vitals, status_map, report_text, lang)
    # Require objective measurements — don't fabricate a "wellness score" from
    # symptom-burden alone. If no vitals were taken, this card stays hidden.
    has_measurements = any(p["available"] for p in pillars if p["key"] in ("cardio","resp","meta"))
    if not has_measurements:
        return
    if lang == "el":
        title    = "📊 ΠΡΟΦΙΛ ΥΓΕΙΑΣ"
        ov_lbl   = "Συνολικό σκορ"
        no_data  = "δεν μετρήθηκε"
        method   = ("Υπολογίζεται από ζωτικά + ιστορικό + ευρήματα εκτίμησης. "
                    "Δεν αντικαθιστά εργαστηριακή μέτρηση.")
        factors_lbl = "Παράγοντες"
    else:
        title    = "📊 HEALTH PROFILE"
        ov_lbl   = "Overall score"
        no_data  = "not measured"
        method   = ("Computed from vitals + history + assessment findings. "
                    "Not a substitute for lab measurements.")
        factors_lbl = "Factors"
    ov_grade, ov_color = _grade_label(overall, lang)
    overall_disp = f"{overall}" if overall is not None else "—"

    # Build pillar rows
    rows_html = ""
    for p in pillars:
        label = p["label_el"] if lang == "el" else p["label_en"]
        grade, gcolor = _grade_label(p["score"], lang)
        score_disp = f"{p['score']}" if p["score"] is not None else "—"
        factors_disp = (" · ".join(p["factors"][:3])) if p["factors"] else no_data
        opacity = "1" if p["available"] else "0.55"
        rows_html += (
            f'<div style="padding:12px 0;border-top:1px solid #F3F4F6;opacity:{opacity}">'
            f'<div style="display:flex;align-items:center;justify-content:space-between;gap:12px">'
            f'<div style="display:flex;align-items:center;gap:10px;min-width:0;flex:1">'
            f'<span style="font-size:20px;flex-shrink:0">{p["icon"]}</span>'
            f'<span style="font-size:13.5px;font-weight:700;color:#1F2937">{label}</span>'
            f'</div>'
            f'<div style="display:flex;align-items:center;gap:10px;flex-shrink:0">'
            f'<span style="font-size:18px;font-weight:800;color:{gcolor};font-variant-numeric:tabular-nums">{score_disp}<span style="font-size:11px;color:#9CA3AF;font-weight:600">%</span></span>'
            f'<span style="background:{gcolor}15;color:{gcolor};font-size:10.5px;font-weight:700;padding:3px 9px;border-radius:99px;letter-spacing:0.04em;text-transform:uppercase">{grade}</span>'
            f'</div>'
            f'</div>'
            f'{_pillar_scale_html(p["score"])}'
            f'<div style="font-size:11px;color:#6B7280;margin-top:6px;line-height:1.5">'
            f'<span style="font-weight:700;letter-spacing:0.08em;text-transform:uppercase">{factors_lbl}:</span> {factors_disp}'
            f'</div>'
            f'</div>'
        )
    st.markdown(f"""
<style>
.hp-card {{
  background: white; border: 1px solid #E0E5FF; border-radius: 22px;
  padding: 24px 24px 22px; margin: 18px 0;
  font-family: 'Inter', system-ui, sans-serif;
  box-shadow: 0 2px 10px rgba(45,63,231,0.05);
}}
.hp-title {{
  font-size: 13.5px; font-weight: 800; letter-spacing: 0.01em;
  color: #1A1A2E;
  border-bottom: 1px solid #EEF1FC; padding-bottom: 14px; margin-bottom: 14px;
}}
.hp-overall {{
  display: flex; align-items: center; gap: 16px;
  background: #F7F8FC;
  border-radius: 16px; padding: 14px 18px; margin-bottom: 4px;
}}
.hp-overall .ov-num {{
  font-size: 38px; font-weight: 800; line-height: 1;
  color: {ov_color}; font-variant-numeric: tabular-nums;
}}
.hp-overall .ov-meta {{ flex: 1; min-width: 0; }}
.hp-overall .ov-lbl {{
  font-size: 10.5px; font-weight: 700; letter-spacing: 0.12em;
  color: #6B7280; text-transform: uppercase;
}}
.hp-overall .ov-grade {{
  font-size: 16px; font-weight: 700; color: {ov_color}; margin-top: 2px;
}}
.hp-method {{
  font-size: 10.5px; color: #9CA3AF; margin-top: 12px;
  padding-top: 10px; border-top: 1px dashed #E5E7EB; line-height: 1.5;
}}
</style>
<div class="hp-card">
<div class="hp-title">{title}</div>
<div class="hp-overall">
<div class="ov-num">{overall_disp}<span style="font-size:18px;color:#9CA3AF;font-weight:600">%</span></div>
<div class="ov-meta"><div class="ov-lbl">{ov_lbl}</div><div class="ov-grade">{ov_grade}</div></div>
</div>
{rows_html}
<div class="hp-method">ℹ️ {method}</div>
</div>
""", unsafe_allow_html=True)


def render_emergency_resources(lang):
    """'Πού να απευθυνθώ' card with:
      - Emergency numbers (166 EKAB, 112 EU, 1066 fire dept) as click-to-call
      - Vrisko.gr links for ΕΟΠΥΥ doctors, on-duty hospitals, on-duty pharmacies
        (these are public Greek directories — same ones nextdeal.gr / vrisko link to)
      - Google Maps quick-search buttons for nearby facilities
    Rendered on the report screen so the user knows their next step after triage."""
    if lang == "el":
        tx = {
            "title":       "📍 ΠΟΥ ΝΑ ΑΠΕΥΘΥΝΘΩ",
            "subtitle":    "Επόμενα βήματα — εφημερεύοντα, ΕΟΠΥΥ γιατροί, φαρμακεία",
            "emerg_title": "🚨 ΣΕ ΕΠΕΙΓΟΥΣΑ ΑΝΑΓΚΗ",
            "ekab":        "ΕΚΑΒ Ασθενοφόρο",
            "eu_112":      "Ευρωπαϊκή Γραμμή Έκτακτης Ανάγκης",
            "pfy":         "Πρωτοβάθμια Φροντίδα Υγείας (1135)",
            "find_doc":    "🩺 Γιατρός ΕΟΠΥΥ",
            "find_doc_sub":"Συμβεβλημένοι γιατροί",
            "find_hosp":   "🚑 Εφημερεύοντα νοσοκομεία",
            "find_hosp_sub":"Σήμερα",
            "find_pharm":  "💊 Εφημερεύοντα φαρμακεία",
            "find_pharm_sub":"Διανυκτερεύοντα",
            "maps_title":  "Άνοιξε στο Google Maps",
            "maps_hosp":   "Νοσοκομείο κοντά μου",
            "maps_doc":    "Ιατρείο κοντά μου",
            "maps_pharm":  "Φαρμακείο κοντά μου",
            "gov_title":   "Ηλεκτρονικές Υπηρεσίες Υγείας (gov.gr)",
            "gov_note":    "Ανοίγουν σε νέα καρτέλα στο gov.gr — δεν αποθηκεύουμε δεδομένα.",
            "gov_links": [
                ("📂", "Ηλεκτρονικός Φάκελος Υγείας",
                 "https://www.gov.gr/ipiresies/ugeia-kai-pronoia/phakelos-ugeias"),
                ("💊", "e-Συνταγογράφηση",
                 "https://www.e-prescription.gr"),
                ("🩺", "Γιατροί ΕΟΠΥΥ",
                 "https://www.eopyy.gov.gr"),
                ("📋", "ΑΜΚΑ",
                 "https://www.gov.gr/ipiresies/apasxolisi-kai-syntaxiodotisi/amka"),
                ("🔔", "ΕΟΔΥ — Εθνικός Οργανισμός Δημόσιας Υγείας",
                 "https://eody.gov.gr"),
            ],
        }
        # Greek Google Maps queries (browser geolocates from device)
        maps_q = {
            "hosp":  "νοσοκομείο",
            "doc":   "ιατρείο",
            "pharm": "φαρμακείο",
        }
    else:
        tx = {
            "title":       "📍 NEXT STEPS — WHERE TO GO",
            "subtitle":    "On-duty facilities, EOPYY doctors, pharmacies",
            "emerg_title": "🚨 IN AN EMERGENCY",
            "ekab":        "EKAB Ambulance",
            "eu_112":      "European Emergency Line",
            "pfy":         "Primary Care Helpline (1135)",
            "find_doc":    "🩺 EOPYY Doctor",
            "find_doc_sub":"Affiliated physicians",
            "find_hosp":   "🚑 On-duty hospitals",
            "find_hosp_sub":"Today",
            "find_pharm":  "💊 On-duty pharmacies",
            "find_pharm_sub":"Night/weekend",
            "maps_title":  "Open in Google Maps",
            "maps_hosp":   "Hospital near me",
            "maps_doc":    "Doctor's office near me",
            "maps_pharm":  "Pharmacy near me",
            "gov_title":   "Digital Health Services (gov.gr)",
            "gov_note":    "Open in a new tab on gov.gr — we store no data.",
            "gov_links": [
                ("📂", "Electronic Health Record",
                 "https://www.gov.gr/ipiresies/ugeia-kai-pronoia/phakelos-ugeias"),
                ("💊", "e-Prescription",
                 "https://www.e-prescription.gr"),
                ("🩺", "EOPYY Doctors",
                 "https://www.eopyy.gov.gr"),
                ("📋", "AMKA",
                 "https://www.gov.gr/ipiresies/apasxolisi-kai-syntaxiodotisi/amka"),
                ("🔔", "EODY — Public Health",
                 "https://eody.gov.gr"),
            ],
        }
        maps_q = {
            "hosp":  "hospital",
            "doc":   "doctor",
            "pharm": "pharmacy",
        }
    import urllib.parse as _up
    def _maps(q):
        return f"https://www.google.com/maps/search/?api=1&query={_up.quote(q)}"
    # Official Ministry of Health page for Athens hospital duty schedule.
    # vrisko.gr was a 3rd-party aggregator; moh.gov.gr is the authoritative source.
    URL_DOC   = "https://www.vrisko.gr/dir/giatroi-eopyy"
    URL_HOSP  = "https://www.moh.gov.gr/articles/citizen/efhmeries-nosokomeiwn/68-efhmeries-nosokomeiwn-attikhs"
    URL_PHARM = "https://www.vrisko.gr/efimeries-farmakeion"
    st.markdown(f"""
<style>
.er-card {{
  background: white; border: 1px solid #E0E5FF; border-radius: 22px;
  padding: 24px 24px 22px; margin: 18px 0;
  font-family: 'Inter', system-ui, sans-serif;
  box-shadow: 0 2px 10px rgba(45,63,231,0.05);
}}
.er-title {{
  font-size: 13.5px; font-weight: 800; letter-spacing: 0.01em;
  color: #1A1A2E;
  border-bottom: 1px solid #EEF1FC; padding-bottom: 14px; margin-bottom: 4px;
}}
.er-subtitle {{
  font-size: 12px; color: #9CA3AF; margin-bottom: 16px;
}}
.er-emerg {{
  background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
  border: 1px solid #FCA5A5; border-radius: 12px;
  padding: 14px 16px; margin-bottom: 16px;
}}
.er-emerg-title {{
  font-size: 10.5px; font-weight: 800; color: #991B1B;
  letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 10px;
}}
.er-emerg-row {{
  display: flex; justify-content: space-between; align-items: center;
  padding: 8px 0; border-top: 1px dashed rgba(220,38,38,0.20);
  gap: 12px;
}}
.er-emerg-row:first-of-type {{ border-top: none; padding-top: 4px; }}
.er-emerg-label {{ font-size: 13px; color: #7F1D1D; font-weight: 600; flex: 1; min-width: 0; }}
.er-call-btn {{
  background: #DC2626; color: white; padding: 7px 16px; border-radius: 8px;
  font-weight: 700; font-size: 14px; text-decoration: none;
  font-variant-numeric: tabular-nums; white-space: nowrap; flex-shrink: 0;
}}
.er-call-btn:hover {{ background: #B91C1C; color: white; text-decoration: none; }}

.er-grid {{
  display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px;
}}
.er-link {{
  display: block;
  background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 10px;
  padding: 14px; text-decoration: none; color: inherit;
  transition: all 0.15s;
}}
.er-link:hover {{
  background: white; border-color: #2D3FE7; text-decoration: none; color: inherit;
  transform: translateY(-1px); box-shadow: 0 2px 6px rgba(45,63,231,0.10);
}}
.er-link-title {{ font-size: 13.5px; font-weight: 700; color: #1F2937; margin-bottom: 3px; }}
.er-link-sub {{ font-size: 11px; color: #6B7280; }}

.er-maps {{
  margin-top: 16px; padding-top: 14px; border-top: 1px dashed #E5E7EB;
}}
.er-maps-title {{
  font-size: 10.5px; font-weight: 700; letter-spacing: 0.12em;
  color: #6B7280; text-transform: uppercase; margin-bottom: 10px;
}}
.er-maps-row {{ display: flex; gap: 8px; flex-wrap: wrap; }}
.er-maps-btn {{
  flex: 1; min-width: 140px;
  display: inline-block; padding: 9px 14px;
  background: white; border: 1px solid #C7D2FE; border-radius: 8px;
  color: #2D3FE7; font-size: 12.5px; font-weight: 600;
  text-decoration: none; text-align: center;
}}
.er-maps-btn:hover {{ background: #EFF6FF; color: #2D3FE7; text-decoration: none; }}

.er-gov-section {{
  margin-top: 20px; padding-top: 16px; border-top: 1px solid #E5E7EB;
}}
.er-gov-title {{
  font-size: 11px; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase;
  color: #059669; margin-bottom: 10px;
}}
.er-gov-grid {{
  display: flex; flex-wrap: wrap; gap: 8px;
}}
.er-gov-link {{
  background: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 8px;
  padding: 8px 14px; font-size: 12.5px; font-weight: 600; color: #065F46;
  text-decoration: none; display: inline-flex; align-items: center; gap: 6px;
}}
.er-gov-link:hover {{ background: #D1FAE5; color: #065F46; text-decoration: none; }}
.er-gov-note {{ font-size: 11px; color: #9CA3AF; margin-top: 8px; }}

@media (max-width: 640px) {{
  .er-grid {{ grid-template-columns: 1fr; }}
  .er-emerg-row {{ flex-wrap: wrap; }}
  .er-maps-btn {{ min-width: 100%; }}
  .er-gov-link {{ font-size: 11.5px; }}
}}
</style>
<div class="er-card">
  <div class="er-title">{tx['title']}</div>
  <div class="er-subtitle">{tx['subtitle']}</div>

  <div class="er-emerg">
    <div class="er-emerg-title">{tx['emerg_title']}</div>
    <div class="er-emerg-row">
      <span class="er-emerg-label">{tx['ekab']}</span>
      <a class="er-call-btn" href="tel:166">📞 166</a>
    </div>
    <div class="er-emerg-row">
      <span class="er-emerg-label">{tx['eu_112']}</span>
      <a class="er-call-btn" href="tel:112">📞 112</a>
    </div>
    <div class="er-emerg-row">
      <span class="er-emerg-label">{tx['pfy']}</span>
      <a class="er-call-btn" href="tel:1135">📞 1135</a>
    </div>
  </div>

  <div class="er-grid">
    <a class="er-link" href="{URL_DOC}" target="_blank" rel="noopener">
      <div class="er-link-title">{tx['find_doc']}</div>
      <div class="er-link-sub">{tx['find_doc_sub']} ↗</div>
    </a>
    <a class="er-link" href="{URL_HOSP}" target="_blank" rel="noopener">
      <div class="er-link-title">{tx['find_hosp']}</div>
      <div class="er-link-sub">{tx['find_hosp_sub']} ↗</div>
    </a>
    <a class="er-link" href="{URL_PHARM}" target="_blank" rel="noopener">
      <div class="er-link-title">{tx['find_pharm']}</div>
      <div class="er-link-sub">{tx['find_pharm_sub']} ↗</div>
    </a>
  </div>

  <div class="er-maps">
    <div class="er-maps-title">📍 {tx['maps_title']}</div>
    <div class="er-maps-row">
      <a class="er-maps-btn" href="{_maps(maps_q['hosp'])}" target="_blank" rel="noopener">🚑 {tx['maps_hosp']}</a>
      <a class="er-maps-btn" href="{_maps(maps_q['doc'])}" target="_blank" rel="noopener">🩺 {tx['maps_doc']}</a>
      <a class="er-maps-btn" href="{_maps(maps_q['pharm'])}" target="_blank" rel="noopener">💊 {tx['maps_pharm']}</a>
    </div>
  </div>

  <div class="er-gov-section">
    <div class="er-gov-title">🇬🇷 {tx['gov_title']}</div>
    <div class="er-gov-grid">
      {"".join(f'<a class="er-gov-link" href="{url}" target="_blank" rel="noopener">{icon} {label}</a>' for icon,label,url in tx['gov_links'])}
    </div>
    <div class="er-gov-note">{tx['gov_note']}</div>
  </div>
</div>
""", unsafe_allow_html=True)


def render_report():
    render_stepper("report")
    p=st.session_state.profile; lang=st.session_state.lang
    nm = p.get("name","")
    age = p.get("age")
    sub_el = f"για {nm}" + (f", {age} ετών" if age else "")
    sub_en = f"for {nm}" + (f", {age} years old" if age else "")
    render_doc_header(
        "Η εκτίμηση σου", "Your assessment",
        icon="📋",
        sub_el=(sub_el if nm else "Κλινική εκτίμηση με τεκμηρίωση"),
        sub_en=(sub_en if nm else "Clinical assessment with references"),
    )
    render_vitals_summary()
    if not st.session_state.report:
        conversation="\n".join(f"{'Patient' if m['role']=='user' else 'Asklepios'}: {m['content']}" for m in st.session_state.triage_chat)
        vitals_text="\n".join(f"- {k}: {v}" for k,v in st.session_state.vitals.items()) if st.session_state.vitals else "Not provided"
        vitals_analysis=st.session_state.vitals_analysis or "Not available"
        last_user=next((m["content"] for m in reversed(st.session_state.triage_chat) if m["role"]=="user"),"")
        search_query=last_user[:80]+" diagnosis management" if last_user else "symptom assessment management"

        # ── Loading banner overlay (full-viewport, animated) ─────────────────
        # The report header pushes the inline spinner below the fold — users
        # stare at an apparently frozen page for 20-40s. This overlay sits on
        # top via position:fixed and keeps animating during the blocking API call.
        _nm_disp = p.get("name") or ("τον/την ασθενή" if lang=="el" else "the patient")
        _overlay = st.empty()
        _overlay.markdown(f"""
<style>
@keyframes ask-float{{0%,100%{{transform:translateY(0)}}50%{{transform:translateY(-10px)}}}}
@keyframes ask-bounce{{0%,80%,100%{{transform:scale(0.55);opacity:.35}}40%{{transform:scale(1.1);opacity:1}}}}
@keyframes ask-spin{{from{{transform:rotate(0deg)}}to{{transform:rotate(360deg)}}}}
@keyframes ask-fadein{{from{{opacity:0}}to{{opacity:1}}}}
@keyframes ask-msg{{0%{{opacity:0;transform:translateY(6px)}}8%{{opacity:1;transform:translateY(0)}}88%{{opacity:1;transform:translateY(0)}}96%{{opacity:0;transform:translateY(-4px)}}100%{{opacity:0}}}}
.ask-overlay{{position:fixed;inset:0;z-index:2147483600;background:rgba(15,23,42,.55);
  backdrop-filter:blur(6px);display:flex;align-items:center;justify-content:center;
  animation:ask-fadein 220ms ease-out;font-family:Inter,system-ui,sans-serif;padding:16px}}
.ask-card{{background:linear-gradient(160deg,#EFF6FF 0%,#fff 100%);border-radius:28px;
  border:2px solid #BFDBFE;padding:36px 28px 28px;width:min(94vw,400px);text-align:center;
  box-shadow:0 30px 80px rgba(37,99,235,.25)}}
.ask-avatar-wrap{{position:relative;width:100px;height:100px;margin:0 auto;
  animation:ask-float 2.6s ease-in-out infinite}}
.ask-aura{{position:absolute;inset:-8px;border-radius:50%;
  background:conic-gradient(from 0deg,#2563EB,#7C3AED,#2563EB,#1D4ED8,#2563EB);
  animation:ask-spin 8s linear infinite;opacity:.8}}
.ask-avatar{{position:relative;width:100px;height:100px;border-radius:50%;background:#fff;
  border:4px solid #3B82F6;display:flex;align-items:center;justify-content:center;
  font-size:48px;overflow:hidden}}
.ask-super{{font-size:11px;font-weight:700;letter-spacing:5px;color:#2563EB;margin-top:18px}}
.ask-name{{font-size:20px;font-weight:800;color:#1E3A5F;margin-top:2px;line-height:1.2;word-break:break-word;max-width:300px;margin-left:auto;margin-right:auto;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.ask-bubble{{position:relative;margin-top:18px;background:#DBEAFE;border:2px solid #93C5FD;
  border-radius:18px 18px 18px 4px;padding:13px 15px;min-height:50px;
  display:flex;align-items:center;justify-content:center;
  color:#1E40AF;font-size:14px;font-weight:600}}
.ask-msg-stack{{position:relative;width:100%;min-height:20px}}
.ask-msg{{position:absolute;left:0;right:0;opacity:0;
  animation:ask-msg 18s infinite ease-in-out}}
.ask-msg:nth-child(1){{animation-delay:0s}}
.ask-msg:nth-child(2){{animation-delay:3s}}
.ask-msg:nth-child(3){{animation-delay:6s}}
.ask-msg:nth-child(4){{animation-delay:9s}}
.ask-msg:nth-child(5){{animation-delay:12s}}
.ask-msg:nth-child(6){{animation-delay:15s}}
.ask-dots{{display:flex;gap:8px;justify-content:center;margin-top:18px}}
.ask-dot{{width:10px;height:10px;border-radius:50%;background:#3B82F6;animation:ask-bounce 1s infinite}}
.ask-dot:nth-child(2){{animation-delay:.15s}}
.ask-dot:nth-child(3){{animation-delay:.30s}}
.ask-foot{{font-size:12px;color:#475569;margin-top:18px;line-height:1.55}}
</style>
<div class="ask-overlay">
<div class="ask-card">
  <div class="ask-avatar-wrap">
    <div class="ask-aura"></div>
    <div class="ask-avatar">🩺</div>
  </div>
  <div class="ask-super">{"ASKLEPIOS AI NURSE" if lang=="el" else "ASKLEPIOS AI NURSE"}</div>
  <div class="ask-name">{_nm_disp}</div>
  <div class="ask-bubble">
    <div class="ask-msg-stack">
      <span class="ask-msg">{"🔬 Αναζήτηση PubMed…" if lang=="el" else "🔬 Searching PubMed…"}</span>
      <span class="ask-msg">{"📚 Ανάλυση βιβλιογραφίας…" if lang=="el" else "📚 Analysing literature…"}</span>
      <span class="ask-msg">{"🩺 Σύνταξη κλινικής εκτίμησης…" if lang=="el" else "🩺 Writing clinical assessment…"}</span>
      <span class="ask-msg">{"💊 Έλεγχος φαρμάκων & αντενδείξεων…" if lang=="el" else "💊 Checking medications…"}</span>
      <span class="ask-msg">{"📍 Εξατομικευμένες συστάσεις…" if lang=="el" else "📍 Personalised recommendations…"}</span>
      <span class="ask-msg">{"✨ Σχεδόν έτοιμο!" if lang=="el" else "✨ Almost ready!"}</span>
    </div>
  </div>
  <div class="ask-dots">
    <div class="ask-dot"></div><div class="ask-dot"></div><div class="ask-dot"></div>
  </div>
  <div class="ask-foot">{"Μην κλείσεις τη σελίδα — η αναφορά δημιουργείται (20–40″)." if lang=="el" else "Don't close the page — report is being generated (20–40s)."}</div>
</div>
</div>
""", unsafe_allow_html=True)

        with st.spinner("🔬 PubMed..." if lang=="el" else "🔬 Searching PubMed..."):
            refs=pubmed_search(search_query,n=3); st.session_state.report_pubmed=refs
        pubmed_ctx="\n".join(f"- {a['title']} ({a['journal']}, {a['date']}) {a['url']}" for a in refs) if refs else "None found."
        pp=p.get
        # Special-population flags that the report MUST respect
        _rep_flags = []
        if pp("pregnancy"):
            _rep_flags.append("PREGNANT — exclude Category D/X drugs; flag teratogenic risks; recommend OB-GYN consultation")
        if pp("for_whom") == "other":
            _rep_flags.append("Caregiver-mode: report addresses the caregiver; use third person for the patient")
        _age_r = pp("age", 0) or 0
        if _age_r < 18:
            _rep_flags.append(f"PEDIATRIC ({_age_r} yo) — use pediatric vital ranges, dosing by weight, age-appropriate red flags")
        _rep_flags_str = ("\nSPECIAL CONSIDERATIONS: " + " | ".join(_rep_flags)) if _rep_flags else ""
        report_prompt=f"""Generate a concise clinical assessment for:
PATIENT: {pp('name')}, {pp('age')}yo {pp('sex')}{_rep_flags_str}
HISTORY: {pp('history','none')} | ALLERGIES: {pp('allergies','none')} | MEDS: {pp('meds_raw','none')}
VITALS: {vitals_text}
VITALS ANALYSIS: {vitals_analysis}
CONSULTATION: {conversation}
PUBMED: {pubmed_ctx}
Write these sections IN THIS ORDER, using EXACTLY these headers as written (do not abbreviate or drop letters):
{"1. ΚΥΡΙΟ ΠΑΡΑΠΟΝΟ  2. ΙΣΤΟΡΙΚΟ  3. ΕΚΤΙΜΗΣΗ (Πρωτεύουσα Διάγνωση + Διαφορικές Διαγνώσεις)  4. ΘΕΡΑΠΕΥΤΙΚΟ ΠΛΑΝΟ  5. ΚΟΚΚΙΝΕΣ ΣΗΜΑΙΕΣ  6. ΒΙΒΛΙΟΓΡΑΦΙΑ" if lang=="el" else "1. CHIEF COMPLAINT  2. HISTORY  3. ASSESSMENT (Primary Diagnosis + Differentials)  4. TREATMENT PLAN  5. RED FLAGS  6. REFERENCES"}
For the differentials use a markdown table with EXACTLY 3 columns and these short headers: {"| Διάγνωση | % | Σχόλιο |" if lang=="el" else "| Diagnosis | % | Comment |"} (keep the probability header as just "%", and put values like "~8%"). Keep cell text short.

After section 6 (References), append EXACTLY this delimited block — same format, no extra text inside the delimiters:
<<<RECS
CONDITION: [the primary clinical condition in 2-4 ENGLISH words, MeSH-friendly — e.g. "Hypertension", "Migraine", "Type 2 Diabetes", "Gastroesophageal Reflux", "Anxiety Disorder". Just the noun phrase, no extra text. This is used to fetch matching guideline literature.]
EXERCISE: [2-3 sentences of PERSONALISED exercise advice for this specific patient — based on age, conditions, symptoms. Direct and actionable. {"Σε Ελληνικά." if lang=="el" else "In English."} No generic platitudes.]
NUTRITION: [2-3 sentences of personalised nutrition advice for this patient — specific foods/changes that target the assessed conditions. {"Σε Ελληνικά." if lang=="el" else "In English."}]
LIFESTYLE: [2-3 sentences on sleep, stress, smoking, alcohol — tailored to this case. {"Σε Ελληνικά." if lang=="el" else "In English."}]
RECS>>>

Language: {"Greek" if lang=="el" else "English"}. Be direct. End with a one-line AI disclaimer.{output_language_directive()}"""
        with st.spinner("Δημιουργία αναφοράς..." if lang=="el" else "Generating report..."):
            result=claude([{"role":"user","content":report_prompt}],system=kira_system(),max_tokens=4000,timeout=120)
            if result.startswith("⚠️"):
                _overlay.empty()
                st.error(result)
                if st.button("🔄 Retry"): st.rerun()
                return
            # Parse out the PNOE-style RECS block ONCE on generation. The cleaned
            # report (without delimiters) is what shows on-screen and in exports;
            # the recs dict drives the 3-column visual card.
            _clean, _recs = _extract_recs(result)
            st.session_state.report = _clean
            st.session_state.report_recs = _recs
            # Fetch high-evidence PubMed refs PER PILLAR (Exercise/Nutrition/Lifestyle)
            # using MeSH + Practice-Guideline/Systematic-Review/Meta-Analysis filters.
            # Also fetch Physiotherapy (PEDro-equivalent via PubMed) and Psychology
            # (PubMed/MEDLINE, peer-reviewed) evidence in the same parallel batch.
            _condition = (_recs or {}).get("condition", "").strip()
            if _condition:
                from concurrent.futures import ThreadPoolExecutor as _TPE
                with st.spinner("📚 " + ("Αναζήτηση οδηγιών ανά πυλώνα..." if lang=="el"
                                          else "Searching guideline-level evidence per pillar...")):
                    with _TPE(max_workers=6) as _ex:
                        _futs = {p: _ex.submit(pubmed_pillar_search, _condition, p, 2)
                                 for p in ("exercise","nutrition","lifestyle")}
                        _futs["physio"]     = _ex.submit(pedro_pillar_search, _condition, 3)
                        _futs["psychology"] = _ex.submit(psychology_pillar_search, _condition, 3)
                        # Re-fetch the MAIN bibliography using the clean MeSH-friendly
                        # condition name (e.g. "Aortic Stenosis TAVI") instead of the
                        # original search_query, which was built from the raw last chat
                        # message (often non-English/conversational) and could return
                        # zero PubMed results even when the condition itself has plenty
                        # of literature. This only refreshes the references shown in the
                        # "🔬 PubMed" expander + PDF export — it does not rewrite the
                        # "6. ΒΙΒΛΙΟΓΡΑΦΙΑ" text the AI already wrote in the report body.
                        _futs["bibliography"] = _ex.submit(pubmed_search, _condition, 3)
                        _all_refs = {p: f.result() for p,f in _futs.items()}
                if _all_refs.get("bibliography"):
                    st.session_state.report_pubmed = _all_refs["bibliography"]
                st.session_state.report_recs_refs = {p: _all_refs[p] for p in ("exercise","nutrition","lifestyle")}
                st.session_state.report_physio_refs = _all_refs.get("physio", [])
                st.session_state.report_psych_refs  = _all_refs.get("psychology", [])
            else:
                st.session_state.report_recs_refs = {}
                st.session_state.report_physio_refs = []
                st.session_state.report_psych_refs  = []
        # Clear the loading overlay now that we have data — rerun will render the report
        _overlay.empty()
        st.rerun()
    if not st.session_state.report:
        if st.button("🔄 "+("Δοκιμή ξανά" if lang=="el" else "Retry"),type="primary"): st.rerun()
        return
    # ── Doctor's-report style: PATIENT INFO doc-card + CLINICAL ASSESSMENT header ──
    # Inspired by the medical-report template (USGH-style): blue/red accent boxes
    # for allergies + medications side-by-side, with medical history above. The
    # actual Claude assessment renders below with restyled markdown section headers.
    history_raw = (p.get("history") or "").strip()
    history = history_raw if history_raw else "—"
    allergies_raw = (p.get("allergies") or "").strip()
    allergies = allergies_raw if allergies_raw else "—"
    meds_raw = (p.get("meds_raw") or "").strip()
    meds_list = [m.strip() for m in meds_raw.split(",") if m.strip()]
    meds_html = "<br>".join(f"• {m}" for m in meds_list) if meds_list else "—"
    if lang == "el":
        TX = {
            "patient_info": "Στοιχεία Ασθενή",
            "history_lbl": "Ιατρικό Ιστορικό",
            "allergies_lbl": "Αλλεργίες",
            "meds_lbl": "Φάρμακα",
            "assessment_title": "Κλινική Αξιολόγηση",
        }
    else:
        TX = {
            "patient_info": "Patient Information",
            "history_lbl": "Medical History",
            "allergies_lbl": "Allergies",
            "meds_lbl": "Medications",
            "assessment_title": "Clinical Assessment",
        }
    st.markdown(f"""
<style>
.report-card {{
  background: white;
  border: 1px solid #E0E5FF;
  border-radius: 22px;
  padding: 24px 24px 22px;
  margin: 6px 0 16px;
  font-family: 'Inter', system-ui, sans-serif;
  box-shadow: 0 2px 10px rgba(45,63,231,0.05);
}}
.report-card-title {{
  font-size: 13.5px; font-weight: 800; letter-spacing: 0.01em;
  color: #1A1A2E;
  padding-bottom: 14px; margin-bottom: 16px;
  border-bottom: 1px solid #EEF1FC;
  display: flex; align-items: center; gap: 10px;
}}
.report-card-title .rct-icon {{
  width: 32px; height: 32px; border-radius: 50%;
  background: #E8ECFE; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  font-size: 15px;
}}
.history-block {{
  background: #F7F8FC; border: 1px solid #EEF1FC;
  border-radius: 14px; padding: 14px 16px; margin-bottom: 14px;
  font-size: 13.5px; color: #374151; line-height: 1.55;
}}
.history-block .hb-lbl {{
  font-size: 10.5px; font-weight: 700; color: #6B7280;
  text-transform: uppercase; letter-spacing: 0.10em; margin-bottom: 6px;
}}
.aller-meds {{
  display: grid; grid-template-columns: 1fr 1fr; gap: 14px;
}}
.aller-box, .meds-box {{
  border-radius: 14px; padding: 16px 18px;
  font-size: 13.5px; line-height: 1.55;
}}
.aller-box {{ background: #FEF2F2; border: 1px solid #FECACA; }}
.meds-box  {{ background: #E8ECFE; border: 1px solid #C7D2FE; }}
.aller-box .am-lbl, .meds-box .am-lbl {{
  font-size: 10.5px; font-weight: 700; letter-spacing: 0.12em;
  text-transform: uppercase; margin-bottom: 8px;
}}
.aller-box .am-lbl {{ color: #991B1B; }}
.meds-box  .am-lbl {{ color: #2D3FE7; }}
.aller-box .am-val {{ color: #7F1D1D; font-weight: 500; }}
.meds-box  .am-val {{ color: #1A1A2E; font-weight: 500; }}

/* Clinical Assessment section header (separates patient info from Claude content) */
.assessment-section-header {{
  background: #F4F6FF;
  border: 1px solid #E0E5FF;
  border-radius: 16px;
  padding: 16px 20px;
  margin: 16px 0 14px;
  display: flex; align-items: center; gap: 12px;
  font-family: 'Inter', system-ui, sans-serif;
}}
.assessment-section-header .ash-icon {{
  width: 38px; height: 38px; border-radius: 50%;
  background: #E8ECFE; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  font-size: 18px;
}}
.assessment-section-header .ash-title {{
  font-size: 12.5px; font-weight: 800; letter-spacing: 0.08em;
  color: #2D3FE7; text-transform: uppercase;
}}

/* Style markdown section headers inside the Claude report so each section
 * ("ΚΥΡΙΟ ΠΑΡΑΠΟΝΟ", "ΙΣΤΟΡΙΚΟ", "ΕΚΤΙΜΗΣΗ" ...) reads like a medical report block */
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2 {{
  color: #2D3FE7 !important;
  font-size: 14px !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.10em !important;
  border-bottom: 1.5px solid #E0E5FF !important;
  padding-bottom: 6px !important;
  margin: 22px 0 12px !important;
}}
[data-testid="stMarkdownContainer"] h3 {{
  color: #2D3FE7 !important;
  font-size: 13.5px !important;
  font-weight: 700 !important;
  margin: 16px 0 8px !important;
}}
@media (max-width: 640px) {{
  .report-card {{ padding: 20px 18px; }}
  .aller-meds {{ grid-template-columns: 1fr; gap: 10px; }}
  .assessment-section-header {{ padding: 12px 16px; }}
}}
</style>
<div class="report-card">
  <div class="report-card-title"><span class="rct-icon">📑</span>{TX['patient_info']}</div>
  <div class="history-block">
    <div class="hb-lbl">📋 {TX['history_lbl']}</div>
    <div>{history}</div>
  </div>
  <div class="aller-meds">
    <div class="aller-box">
      <div class="am-lbl">🔴 {TX['allergies_lbl']}</div>
      <div class="am-val">{allergies}</div>
    </div>
    <div class="meds-box">
      <div class="am-lbl">💊 {TX['meds_lbl']}</div>
      <div class="am-val">{meds_html}</div>
    </div>
  </div>
</div>
<div class="assessment-section-header">
  <span class="ash-icon">📋</span>
  <span class="ash-title">{TX['assessment_title']}</span>
</div>
""", unsafe_allow_html=True)
    st.markdown(st.session_state.report)
    # Photo findings card — if the user uploaded any photos during triage, the
    # AI vision analyses become visible evidence in the final report. Each card
    # shows scan type + Claude's interpretation. (Florence-2 description is
    # already woven into the analysis so we don't duplicate it.)
    _pfs = st.session_state.get("photo_findings") or []
    if isinstance(_pfs, list) and _pfs:
        _pf_title = ("📷 ΕΥΡΗΜΑΤΑ ΑΠΟ ΦΩΤΟΓΡΑΦΙΕΣ" if lang=="el"
                     else "📷 PHOTO FINDINGS")
        _pf_count = len(_pfs)
        import html as _html_pf, re as _re_pf
        def _flat_pf(t): return _re_pf.sub(r"\s+", " ", (t or "").strip())
        _cards_html = ""
        for i, pf in enumerate(_pfs, 1):
            _label = _html_pf.escape(pf.get("scan_label","—"))
            _analysis = _flat_pf(pf.get("analysis",""))
            # Keep markdown bold/headers in the analysis readable inside the card —
            # convert ** ** → <strong>, leave the rest as text after escape.
            _analysis = _html_pf.escape(_analysis)
            _analysis = _re_pf.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", _analysis)
            _cards_html += (
                f'<div class="pf-item">'
                f'<div class="pf-head"><span class="pf-num">{i}</span><span class="pf-label">{_label}</span></div>'
                f'<div class="pf-body">{_analysis}</div>'
                f'</div>'
            )
        st.markdown(
            f'<style>'
            f'.pf-card{{background:white;border:1px solid #E0E5FF;border-radius:22px;padding:24px 24px 22px;margin:18px 0;font-family:Inter,system-ui,sans-serif;box-shadow:0 2px 10px rgba(45,63,231,0.05)}}'
            f'.pf-title{{font-size:13.5px;font-weight:800;letter-spacing:0.01em;color:#1A1A2E;border-bottom:1px solid #EEF1FC;padding-bottom:14px;margin-bottom:16px}}'
            f'.pf-item{{padding:14px 0;border-bottom:1px solid #F3F4F6}}'
            f'.pf-item:last-child{{border-bottom:none;padding-bottom:0}}'
            f'.pf-head{{display:flex;align-items:center;gap:10px;margin-bottom:8px}}'
            f'.pf-num{{background:#E8ECFE;color:#2D3FE7;font-size:11px;font-weight:700;width:24px;height:24px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;flex-shrink:0}}'
            f'.pf-label{{font-size:13.5px;font-weight:700;color:#111827}}'
            f'.pf-body{{font-size:13px;color:#374151;line-height:1.6}}'
            f'.pf-body strong{{color:#1F2937}}'
            f'</style>'
            f'<div class="pf-card"><div class="pf-title">{_pf_title} · {_pf_count}</div>{_cards_html}</div>',
            unsafe_allow_html=True,
        )
    # Lab findings card — mirrors the photo card style with a green accent for
    # laboratory data. Same in-memory-only privacy: contents are session-only.
    _lfs = st.session_state.get("lab_findings") or []
    if isinstance(_lfs, list) and _lfs:
        _lf_title = ("🧪 ΕΥΡΗΜΑΤΑ ΕΡΓΑΣΤΗΡΙΑΚΩΝ ΕΞΕΤΑΣΕΩΝ" if lang=="el"
                     else "🧪 LAB FINDINGS")
        _lf_count = len(_lfs)
        import html as _html_lf, re as _re_lf
        def _flat_lf(t): return _re_lf.sub(r"\s+", " ", (t or "").strip())
        _lf_cards = ""
        for i, lf in enumerate(_lfs, 1):
            _fname = _html_lf.escape(lf.get("file_name","—"))
            _an = _html_lf.escape(_flat_lf(lf.get("analysis","")))
            _an = _re_lf.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", _an)
            _lf_cards += (
                f'<div class="lf-item">'
                f'<div class="lf-head"><span class="lf-num">{i}</span>'
                f'<span class="lf-label">📄 {_fname}</span></div>'
                f'<div class="lf-body">{_an}</div>'
                f'</div>'
            )
        st.markdown(
            f'<style>'
            f'.lf-card{{background:white;border:1px solid #E0E5FF;border-radius:22px;padding:24px 24px 22px;margin:18px 0;font-family:Inter,system-ui,sans-serif;box-shadow:0 2px 10px rgba(45,63,231,0.05)}}'
            f'.lf-title{{font-size:13.5px;font-weight:800;letter-spacing:0.01em;color:#1A1A2E;border-bottom:1px solid #EEF1FC;padding-bottom:14px;margin-bottom:16px}}'
            f'.lf-item{{padding:14px 0;border-bottom:1px solid #F3F4F6}}'
            f'.lf-item:last-child{{border-bottom:none;padding-bottom:0}}'
            f'.lf-head{{display:flex;align-items:center;gap:10px;margin-bottom:8px}}'
            f'.lf-num{{background:#D1FAE5;color:#065F46;font-size:11px;font-weight:700;width:24px;height:24px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;flex-shrink:0}}'
            f'.lf-label{{font-size:13.5px;font-weight:700;color:#111827}}'
            f'.lf-body{{font-size:13px;color:#374151;line-height:1.6}}'
            f'.lf-body strong{{color:#1F2937}}'
            f'</style>'
            f'<div class="lf-card"><div class="lf-title">{_lf_title} · {_lf_count}</div>{_lf_cards}</div>',
            unsafe_allow_html=True,
        )
    # PNOE-style 3-pillar Recommendations card (Exercise / Nutrition / Lifestyle)
    if st.session_state.get("report_recs"):
        _render_recs_card(st.session_state.report_recs, lang,
                          refs=st.session_state.get("report_recs_refs") or {})

    # (Physio and psychology cards removed — no dedicated API available.)

    # Where-to-go card: emergency numbers + nearby clinics/pharmacies finder.
    # Placed right after the personalised recs so the user has all the info
    # needed to take the next step.
    render_emergency_resources(lang)
    if st.session_state.report_pubmed:
        with st.expander(f"🔬 {t('pubmed')} ({len(st.session_state.report_pubmed)})"):
            for a in st.session_state.report_pubmed:
                st.markdown(f"**[{a['title']}]({a['url']})**  \n*{a['authors']} — {a['journal']}, {a['date']}*")
    if get_openai_key():
        with st.expander(f"🤖 {t('second_opinion')}"):
            if not st.session_state.report_gpt:
                if st.button(("Λάβε δεύτερη γνώμη GPT-4o" if lang=="el" else "Get GPT-4o second opinion"),
                             type="secondary", key="gpt_get"):
                    with st.spinner("GPT-4o reviewing..."):
                        _gpt_prompt = (
                            f"Patient: {p.get('name')}, {p.get('age')}yo {p.get('sex','')}\n"
                            f"History: {p.get('history','none')} | Allergies: {p.get('allergies','none')} | Meds: {p.get('meds_raw','none')}\n\n"
                            f"Claude clinical assessment:\n{st.session_state.report}\n\n"
                            f"As an independent clinical reviewer: do you AGREE with this assessment? "
                            f"What specific ADDITIONS or CORRECTIONS would you make (differentials missed, "
                            f"treatment refinements, red flags overlooked, drug-interaction concerns)? "
                            f"Be concise — bullet points OK. Respond in {'Greek' if lang=='el' else 'English'}."
                        )
                        st.session_state.report_gpt = gpt4o(prompt=_gpt_prompt, system=kira_system(), max_tokens=900)
                    st.rerun()
            else:
                st.markdown(st.session_state.report_gpt)
                # Integration: if the second opinion adds value, the user can fold
                # it into the main report so it shows up in the on-screen assessment
                # AND in every downstream export (PDF/HTML/TXT/WhatsApp).
                st.divider()
                if st.session_state.get("_gpt_integrated"):
                    st.success("✓ " + ("Ενσωματώθηκε στην τελική εκτίμηση παραπάνω και στα exports."
                                       if lang=="el" else
                                       "Integrated into the final assessment above and in all exports."))
                else:
                    if st.button(("➕ Ενσωμάτωση στην τελική εκτίμηση" if lang=="el"
                                  else "➕ Integrate into final assessment"),
                                 type="primary", use_container_width=True, key="gpt_integrate"):
                        _hdr = "## " + ("ΔΕΥΤΕΡΗ ΓΝΩΜΗ (GPT-4o)" if lang=="el"
                                        else "SECOND OPINION (GPT-4o)")
                        st.session_state.report = (
                            (st.session_state.report or "").rstrip()
                            + "\n\n---\n\n" + _hdr + "\n\n"
                            + (st.session_state.report_gpt or "").strip()
                        )
                        st.session_state["_gpt_integrated"] = True
                        st.rerun()
                    st.caption(("💡 Προσθέτει τη δεύτερη γνώμη ως ξεχωριστή ενότητα στην αναφορά "
                                "και σε όλα τα exports (PDF/TXT/WhatsApp)."
                                if lang=="el" else
                                "💡 Adds the second opinion as a separate section in the report "
                                "and in every export (PDF/TXT/WhatsApp)."))
    if len(st.session_state.medications)>=2:
        with st.expander("💊 RxNorm" + (" — Έλεγχος Αλληλεπιδράσεων" if lang=="el" else " — Interactions")):
            with st.spinner("RxNorm..."): rxr=rxnorm_interactions([m["name"] for m in st.session_state.medications])
            if rxr: st.markdown(rxr)
    # ── 4-Pillar Health Profile (replaces the old placeholder wellness score).
    # Honest, factor-explained — Cardiovascular / Respiratory / Metabolic /
    # Symptom burden — each backed by the vitals + history items that drove it.
    v=st.session_state.vitals
    _status_map = classify_vitals(dict(v), age=st.session_state.profile.get("age")) if v else {}
    _render_health_pillars(st.session_state.profile, v, _status_map,
                           st.session_state.report, lang)
    urgent_kw=["chest pain","πόνος στήθους","stroke","εγκεφαλικό","anaphylaxis","αναφυλαξία","166","112","emergency","επείγον","unconscious","αναίσθητος"]
    if any(kw in st.session_state.report.lower() for kw in urgent_kw):
        st.markdown('<div class="red-flags-urgent">🚨 Η αναφορά περιέχει <b>επείγουσες ενδείξεις</b>. Καλέστε <b>166</b> ή <b>112</b> αμέσως αν ισχύουν.</div>',unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="emergency">{t("emergency")}</div>',unsafe_allow_html=True)
    st.markdown('<div class="disclaimer-red">AI-generated. Δεν αντικαθιστά ιατρική γνώμη.</div>',unsafe_allow_html=True)

    # ── Feedback (👍/👎) — quality signal only, no medical data stored ──────────
    st.markdown("---")
    if st.session_state.get("fb_sent"):
        st.success("Ευχαριστούμε για το feedback!" if lang=="el" else "Thanks for your feedback!")
    else:
        st.caption("Σου φάνηκε χρήσιμη η εκτίμηση; (μας βοηθάει να βελτιωνόμαστε)" if lang=="el" else "Was this assessment helpful? (helps us improve)")
        rating = st.session_state.get("fb_rating", "")
        fc1, fc2 = st.columns(2)
        with fc1:
            if st.button(("👍 Χρήσιμη" if lang=="el" else "👍 Helpful"), key="fb_up", use_container_width=True,
                         type=("primary" if rating=="up" else "secondary")):
                st.session_state["fb_rating"]="up"; st.rerun()
        with fc2:
            if st.button(("👎 Όχι χρήσιμη" if lang=="el" else "👎 Not helpful"), key="fb_down", use_container_width=True,
                         type=("primary" if rating=="down" else "secondary")):
                st.session_state["fb_rating"]="down"; st.rerun()
        if rating:
            comment = st.text_area(("Τι θα βελτίωνες; (προαιρετικό)" if lang=="el" else "What would you improve? (optional)"),
                                   key="fb_comment", height=80)
            if st.button(("Αποστολή" if lang=="el" else "Submit"), key="fb_submit", type="primary"):
                save_feedback(rating, comment)
                st.session_state["fb_sent"]=True; st.rerun()

    fname=f"asklepios_report_{p.get('name','patient')}_{datetime.now().strftime('%Y%m%d')}"
    c1,c2,c3,c4=st.columns(4)
    with c1:
        if st.button("← "+("Νέα Αξιολόγηση" if lang=="el" else "New Assessment"),use_container_width=True):
            delete_draft(st.session_state.get("auth_user",""))
            for k,vv in defaults.items(): st.session_state[k]=vv
            for fbk in ("fb_comment","fb_rating","fb_sent","photo_added","photo_findings",
                        "_draft_hash","_from_facescan","_scan_injected","_vitals_nudge_off",
                        "_gpt_integrated","_photo_preview",
                        "lab_added","lab_findings","_lab_preview"): st.session_state.pop(fbk, None)
            st.rerun()
    with c2:
        # TXT: report + recs (plain text) so the file is self-contained
        _txt_parts = [st.session_state.report or ""]
        _r = st.session_state.get("report_recs")
        if _r and any(_r.get(k) for k in ("exercise","nutrition","lifestyle")):
            _hdr = ("ΕΞΑΤΟΜΙΚΕΥΜΕΝΕΣ ΣΥΣΤΑΣΕΙΣ" if lang=="el" else "PERSONALISED RECOMMENDATIONS")
            _lbls = (("Φυσική Δραστηριότητα","Διατροφή","Τρόπος Ζωής") if lang=="el"
                     else ("Exercise","Nutrition","Lifestyle"))
            _txt_parts += [
                "", "", "## " + _hdr,
                f"🏃 {_lbls[0]}: " + _r.get("exercise","—"),
                f"🥗 {_lbls[1]}: " + _r.get("nutrition","—"),
                f"🌿 {_lbls[2]}: " + _r.get("lifestyle","—"),
            ]
        _txt_full = "\n".join(_txt_parts)
        st.download_button("📄 TXT",data=_txt_full,file_name=fname+".txt",mime="text/plain",use_container_width=True)
    with c3:
        _recs_for_html = dict(st.session_state.get("report_recs") or {})
        if _recs_for_html:
            _recs_for_html["_refs"] = st.session_state.get("report_recs_refs") or {}
        _pf_for_html = st.session_state.get("photo_findings") or []
        if not isinstance(_pf_for_html, list):
            _pf_for_html = []
        _lf_for_html = st.session_state.get("lab_findings") or []
        if not isinstance(_lf_for_html, list):
            _lf_for_html = []
        st.download_button("📄 PDF/HTML",data=generate_html_report(st.session_state.profile,st.session_state.vitals,st.session_state.report,st.session_state.report_pubmed,lang=lang,recs=_recs_for_html,photo_findings=_pf_for_html,lab_findings=_lf_for_html),file_name=fname+".html",mime="text/html",use_container_width=True,help="Open in browser → Ctrl+P → Save as PDF")
    with c4:
        import re as _re_wa
        wa_lines=[f"🩺 Asklepios AI Nurse",
                  f"Ασθενής: {p.get('name','')} {p.get('age','')}y · {p.get('sex','')}"]
        vbits=[]
        if v.get("hr"):     vbits.append(f"HR {v['hr']}bpm")
        if v.get("bp_sys"): vbits.append(f"BP {v['bp_sys']}/{v.get('bp_dia','?')}mmHg")
        if v.get("br"):     vbits.append(f"BR {v['br']}/min")
        if v.get("spo2"):   vbits.append(f"SpO2 {v['spo2']}%")
        if v.get("temp"):   vbits.append(f"T {v['temp']}°C")
        if v.get("bmi"):    vbits.append(f"ΔΜΣ {v['bmi']}")
        if vbits: wa_lines.append("Ζωτικά: "+", ".join(vbits))
        # Clean markdown from report so it reads well in WhatsApp
        rep=_re_wa.sub(r"[#*>`|]", "", st.session_state.report or "").strip()
        rep=_re_wa.sub(r"\n{3,}", "\n\n", rep)
        # Cap length — wa.me pre-fill fails on very long URLs
        if len(rep)>1500:
            rep=rep[:1500].rsplit("\n",1)[0].rstrip()+"\n…(πλήρης αναφορά στο PDF)"
        if rep:
            wa_lines+=["", rep]
        # PNOE-style recs in WhatsApp (plain emoji-prefixed lines)
        _r2 = st.session_state.get("report_recs")
        if _r2 and any(_r2.get(k) for k in ("exercise","nutrition","lifestyle")):
            _lbls2 = (("Άσκηση","Διατροφή","Τρόπος ζωής") if lang=="el"
                      else ("Exercise","Nutrition","Lifestyle"))
            wa_lines += ["", ("📍 Συστάσεις:" if lang=="el" else "📍 Recommendations:")]
            if _r2.get("exercise"):  wa_lines.append(f"🏃 {_lbls2[0]}: {_r2['exercise']}")
            if _r2.get("nutrition"): wa_lines.append(f"🥗 {_lbls2[1]}: {_r2['nutrition']}")
            if _r2.get("lifestyle"): wa_lines.append(f"🌿 {_lbls2[2]}: {_r2['lifestyle']}")
        wa_lines+=["", "---", "⚠️ AI-generated. asklepiosainurse.up.railway.app"]
        msg="\n".join(wa_lines)
        wa_url="https://wa.me/?text="+urllib.parse.quote(msg)
        st.markdown(f'<a href="{wa_url}" target="_blank" style="display:block;text-align:center;padding:8px;border-radius:8px;text-decoration:none;font-weight:600;font-size:13px;color:white;background:#25D366">WhatsApp</a>',unsafe_allow_html=True)

# ── COOKIE MANAGER (once) — persistent login + in-progress profile draft ──────
if _STX_OK and auth_enabled():
    try:
        CM = stx.CookieManager()
    except Exception:
        CM = None

# Read the login cookie (single component call).
_all_cookies = {}
if CM is not None:
    try:
        _all_cookies = CM.get_all() or {}
    except Exception:
        _all_cookies = {}

# Restore login from the signed cookie (keeps the user signed in across reloads /
# new tabs — e.g. the tab returning from the external face scan).
if auth_enabled() and not is_logged_in():
    _ctok = _all_cookies.get(COOKIE_NAME)
    _em = _read_token(_ctok) if _ctok else None
    if _em:
        st.session_state["auth_user"] = _em
        # Cookie restore = returning user; skip the hero landing for this session.
        st.session_state.setdefault("_hero_seen", True)

# Restore the in-progress assessment from the ENCRYPTED server-side draft ONLY
# when returning from the face scan (which sets _from_facescan on the very first
# run of the new tab). General re-opens of the app stay clean — the user does NOT
# pick up an old conversation just because they opened the app again.
#
# Order matters: on run 1 (URL has facescan param), this block sees _from_facescan
# still False (set later by the facescan block below), so it skips. The facescan
# block sets _from_facescan=True and st.rerun() → on run 2 this block fires.
if (auth_enabled() and is_logged_in()
        and st.session_state.get("_from_facescan")
        and not st.session_state.profile.get("name")
        and not st.session_state.get("_draft_loaded")):
    st.session_state["_draft_loaded"] = True
    _dd = load_draft(st.session_state.get("auth_user", ""))
    if _dd and (_dd.get("profile") or {}).get("name"):
        st.session_state.profile = _dd["profile"]
        if _dd.get("lang"):
            st.session_state.lang = _dd["lang"]
        if _dd.get("triage_chat"):
            st.session_state.triage_chat = _dd["triage_chat"]
        if _dd.get("vitals_analysis"):
            st.session_state.vitals_analysis = _dd["vitals_analysis"]
        if _dd.get("medications"):
            st.session_state.medications = _dd["medications"]
        else:
            _mr = st.session_state.profile.get("meds_raw", "")
            st.session_state.medications = [{"name": m.strip(), "freq": "", "notes": ""}
                                            for m in _mr.split(",") if m.strip()] if _mr else []
        # One-shot: the draft has served its purpose, delete it so it does NOT
        # resurrect on later re-opens.
        delete_draft(st.session_state.get("auth_user", ""))

# If we came back from the face scan during an ongoing conversation, drop the
# measurement into the chat so Asklepios continues the SAME assessment with it
# (instead of the result just sitting silently in the vitals badge).
if (st.session_state.get("_from_facescan") and st.session_state.triage_chat
        and not st.session_state.get("_scan_injected")):
    _v = st.session_state.vitals
    _bits = []
    if _v.get("hr"):
        _bits.append((f"καρδιακός ρυθμός {_v['hr']} bpm" if st.session_state.lang=="el"
                      else f"heart rate {_v['hr']} bpm"))
    if _v.get("br"):
        _bits.append((f"αναπνοές {_v['br']}/min" if st.session_state.lang=="el"
                      else f"breathing {_v['br']}/min"))
    if _bits:
        _m = (("Μέτρησα τα ζωτικά μου με τη σάρωση: " if st.session_state.lang=="el"
               else "I measured my vitals with the scan: ") + ", ".join(_bits) + ".")
        st.session_state.triage_chat.append({"role": "user", "content": _m})
        st.session_state["_scan_injected"] = True
        st.session_state["_scan_reply_pending"] = True

# ── FACESCAN INTERCEPTION ─────────────────────────────────────────────────────
try:
    _raw = st.query_params.get("facescan","")
    if _raw:
        _scanned = json.loads(urllib.parse.unquote(_raw))
        if _scanned and isinstance(_scanned, dict):
            # Filter out null values — only keep actual measurements
            _clean = {k: v for k, v in _scanned.items()
                      if v is not None and v != 0 and k not in ("quality","wellness")}
            # Keep quality and wellness as-is even if 0
            if "quality"  in _scanned: _clean["quality"]  = _scanned["quality"]
            if "wellness" in _scanned: _clean["wellness"] = _scanned["wellness"]
            if _clean:
                st.session_state.vitals = _clean
                st.session_state["_from_facescan"] = True
                st.session_state["_fs_banner"] = True
                st.session_state["_scan_injected"] = False
                st.session_state.screen = "triage" if st.session_state.profile.get("name") else "intake"
            st.query_params.clear()
            st.rerun()
except Exception:
    pass

# If we returned from the face scan and the profile draft has since been restored
# (the cookie can take a render to arrive), skip the now-prefilled intake form and
# jump straight to the assessment.
if (st.session_state.get("_from_facescan") and st.session_state.vitals
        and st.session_state.profile.get("name") and st.session_state.screen == "intake"):
    st.session_state.screen = "triage"
    st.session_state["_from_facescan"] = False

# ── ADMIN ROUTING ────────────────────────────────────────────────────────────
# Reached via asklepiosainurse.up.railway.app/?admin=1. Completely separate
# from the patient OTP login below — gated by its own password
# (render_admin_gate), never touches auth_enabled()/is_logged_in().
if st.query_params.get("admin") == "1":
    if render_admin_gate():
        render_admin_panel()
    st.stop()

# ── HERO LANDING — shown once per session to every visitor ───────────────────
# Mirrors the pet app pattern (_hero_seen). The hero is shown:
#   • to non-logged-in visitors  → replaces the bare login screen
#   • to logged-in users on first open of the session → before home
# Once the CTA button is clicked inside render_login_screen(), it sets
# _hero_seen = True and calls st.rerun() — this block is then skipped.
# Admin and face-scan round-trips skip it via st.stop() above.
if not st.session_state.get("_hero_seen"):
    render_login_screen()
    st.stop()

# ── LOGIN GATE ────────────────────────────────────────────────────────────────
# Login-first: every visitor is identified in Supabase → Authentication → Users.
if auth_enabled() and not is_logged_in():
    render_login_screen()
    st.stop()

# ── PERSIST login cookie on a CLEAN render pass ───────────────────────────────
# The login cookie write on the verify *click* + immediate st.rerun() is unreliable
# (the rerun aborts the stx browser write); a normal render that completes lands it.
if CM is not None and is_logged_in() and not st.session_state.get("_cookie_synced"):
    _save_login_cookie(st.session_state.get("auth_user", ""))
    st.session_state["_cookie_synced"] = True
# NOTE: the encrypted draft is NOT saved on every clean render. It is saved only
# when about to leave for an external page (face scan) via
# _save_session_for_external_nav(). After the round-trip it is one-shot deleted.
# This way the user does NOT accumulate a saved conversation across general
# re-opens — re-entering the app starts clean.

screen=st.session_state.screen
render_topbar()
if st.session_state.pop("_fs_banner", False):
    v_loaded = st.session_state.vitals
    metrics  = [f"HR:{v_loaded['hr']}bpm" if "hr" in v_loaded else "",
                f"BR:{v_loaded['br']}/min" if "br" in v_loaded else "",
                f"HRV:{v_loaded['hrv']}ms" if "hrv" in v_loaded else ""]
    metrics_str = " · ".join(m for m in metrics if m)
    lang = st.session_state.lang
    msg = (f"✅ Σάρωση φορτώθηκε! {metrics_str}" if lang=="el"
           else f"✅ Face scan loaded! {metrics_str}")
    st.success(msg)
if   screen=="home":   render_home()
elif screen=="intake": render_intake()
elif screen=="vitals": render_vitals()
elif screen=="triage": render_triage()
elif screen=="report": render_report()
elif screen=="history": render_history()
else: render_home()

# Bottom tab bar — spacer first (keeps the screen's own last element from
# being hidden behind the fixed bar), then the nav itself.
st.markdown('<div class="bottom-nav-spacer"></div>', unsafe_allow_html=True)
render_bottom_nav()
