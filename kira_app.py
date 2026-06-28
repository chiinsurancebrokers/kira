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
.kira-step-label { font-size: 10px; color: #94A3B8; text-align: center; letter-spacing: .02em; word-break: break-word; overflow-wrap: break-word; }
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
    """Inline email→OTP login. Returns True once the user is logged in.

    Security hardening vs original:
    - Strict email regex (not just '@' check)
    - otp_sent_to from URL only accepted if valid email format
    - _log_user_login() upserts email into user_logins table after verify
      (email, last_seen, lang) — no medical data, purely for user counting.
    - st.error (blocking) instead of st.warning on bad input
    """
    import re as _re
    lang = st.session_state.lang
    if is_logged_in():
        return True

    st.markdown(f'''<div style="background:rgba(45,63,231,0.06);border:1px solid rgba(45,63,231,0.15);border-radius:14px;padding:20px 22px;text-align:center;margin:10px 0">
        <div style="font-size:34px;margin-bottom:6px">🔒</div>
        <div style="font-size:16px;font-weight:700;color:#1A1A2E">{"Σύνδεση" if lang=="el" else "Sign in"}</div>
        <div style="font-size:13px;color:#6B7280;margin-top:4px">{"Email + κωδικός μίας χρήσης. Χωρίς password." if lang=="el" else "Email + one-time code. No password."}</div>
    </div>''', unsafe_allow_html=True)

    def _valid_email(e):
        return bool(_re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', (e or "").strip()))

    def _log_user_login(email):
        """Upsert into user_logins for analytics. No medical data stored."""
        try:
            sb = _supabase_client()
            if not sb: return
            from datetime import datetime as _dt
            sb.table("user_logins").upsert({
                "email":     email,
                "last_seen": _dt.utcnow().isoformat(),
                "lang":      st.session_state.get("lang", "el"),
            }, on_conflict="email").execute()
        except Exception:
            pass  # analytics failure must never block login

    sent_to = st.session_state.get("otp_sent_to")
    if not sent_to:
        pe = st.query_params.get("pe")
        if pe and _valid_email(pe):          # only trust URL param if it looks like an email
            st.session_state["otp_sent_to"] = pe
            sent_to = pe

    if not sent_to:
        # ── STAGE 1: enter email ────────────────────────────────────────────
        email = st.text_input("Email", key="otp_email", placeholder="you@example.com")
        if st.button(("📩 " + ("Στείλε μου τον κωδικό" if lang=="el" else "Send me the code")),
                     type="primary", use_container_width=True, key="otp_send"):
            email_clean = (email or "").strip().lower()
            if not _valid_email(email_clean):
                st.error("Παρακαλώ βάλε ένα έγκυρο email (π.χ. name@example.com)."
                         if lang=="el" else
                         "Please enter a valid email address (e.g. name@example.com).")
            else:
                ok, err = send_otp(email_clean)
                st.session_state["otp_sent_to"] = email_clean
                st.query_params["pe"] = email_clean
                if not ok:
                    st.session_state["_otp_send_warning"] = (err or "")[:140]
                st.rerun()
    else:
        # ── STAGE 2: enter OTP code ─────────────────────────────────────────
        warn = st.session_state.pop("_otp_send_warning", None)
        if warn:
            st.warning("⚠️ Πιθανό πρόβλημα αποστολής — έλεγξε inbox & spam, βάλε τον κωδικό παρακάτω."
                       if lang=="el" else
                       "⚠️ Send had an issue — check inbox & spam, then enter the code below.")
        else:
            st.success("📧 " + (f"Σου στείλαμε κωδικό στο **{sent_to}**"
                                 if lang=="el" else f"We sent a code to **{sent_to}**"))
        st.caption("Έλεγξε inbox & spam. Ο κωδικός φτάνει σε λίγα δευτερόλεπτα."
                   if lang=="el" else
                   "Check inbox & spam. The code arrives within a few seconds.")

        code = st.text_input(
            ("Κωδικός από το email" if lang=="el" else "Code from your email"),
            key="otp_code", placeholder="12345678", max_chars=8,
        )
        if st.button(("✓ " + ("Επιβεβαίωση & Σύνδεση" if lang=="el" else "Verify & Sign in")),
                     type="primary", use_container_width=True, key="otp_verify"):
            _code_clean = str(code or "").strip().replace(" ", "")
            if not _code_clean or not _code_clean.isdigit() or len(_code_clean) < 6:
                st.error("Βάλε τον 6-8ψήφιο κωδικό από το email."
                         if lang=="el" else
                         "Enter the 6-8 digit code from your email.")
            else:
                ok, err = verify_otp(sent_to, _code_clean)
                if ok:
                    _log_user_login(sent_to)        # ← analytics upsert
                    st.session_state.pop("otp_sent_to", None)
                    if "pe" in st.query_params: del st.query_params["pe"]
                    st.rerun()
                else:
                    st.error("Λάθος ή ληγμένος κωδικός — δοκίμασε ξανά ή πάτα «Νέος κωδικός»."
                             if lang=="el" else
                             "Wrong or expired code — try again or press 'New code'.")

        c1, c2 = st.columns(2)
        with c1:
            if st.button(("📩 " + ("Νέος κωδικός" if lang=="el" else "New code")),
                         use_container_width=True, key="otp_resend"):
                ok2, _ = send_otp(sent_to)
                if ok2:
                    st.success("Νέος κωδικός στάλθηκε." if lang=="el" else "New code sent.")
                else:
                    st.info("Αν δεν λάβεις σε 60'', χρησιμοποίησε τον προηγούμενο."
                            if lang=="el" else
                            "If no new code in 60s, use the previous one.")
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
    """Hero landing + login — fully multilingual via t() keys.
    Language picker at top sets st.session_state.lang which drives all t() calls.
    RTL handled via is_rtl() global CSS injection in the main router.
    """
    lang = st.session_state.lang
    rtl  = is_rtl()
    _dir = 'rtl' if rtl else 'ltr'
    _ta  = 'right' if rtl else 'center'

    # ── Language selector — full dropdown, first thing the user sees ──────────
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
.ask-hero-nav{display:flex;align-items:center;justify-content:space-between;
  padding:10px 2px 14px;font-family:'Inter',system-ui,sans-serif;}
.ask-hero-logo{font-size:17px;font-weight:800;color:#1A1A2E;}
.ask-hero-logo span{color:#2D3FE7;}
</style>
""", unsafe_allow_html=True)
    lc1, lc2 = st.columns([5, 2])
    with lc1:
        st.markdown('<div class="ask-hero-nav"><div class="ask-hero-logo">⚕ <span>Asklepios</span></div></div>', unsafe_allow_html=True)
    with lc2:
        _all_codes = list(OUTPUT_LANGUAGES.keys())
        try:    _idx = _all_codes.index(lang)
        except: _idx = 0
        _chosen = st.selectbox("", _all_codes, index=_idx,
                               format_func=lambda c: OUTPUT_LANGUAGES[c][0],
                               key="hero_lang_select", label_visibility="collapsed")
        if _chosen != lang:
            st.session_state.lang = _chosen
            st.session_state["output_lang"] = _chosen
            st.rerun()

    # RTL wrapper for the whole hero when needed
    _rtl_open  = f'<div dir="{_dir}" style="text-align:{_ta};">' if rtl else ""
    _rtl_close = "</div>" if rtl else ""

    # ── HERO CARD ─────────────────────────────────────────────────────────────
    st.markdown(f"""
<style>
.ask-hr-hero{{background:#F4F6FF;border:1px solid #E0E5FF;border-radius:24px;
  padding:30px 22px 22px;margin:0 0 16px;text-align:{_ta};
  font-family:'Inter',system-ui,sans-serif;direction:{_dir};}}
.ask-hr-kicker{{font-size:10px;font-weight:700;letter-spacing:0.14em;color:#2D3FE7;
  margin-bottom:12px;text-transform:uppercase;}}
.ask-hr-h1{{font-size:28px;font-weight:800;line-height:1.18;color:#1A1A2E;
  letter-spacing:-0.5px;margin-bottom:12px;}}
.ask-hr-sub{{font-size:14px;color:#4B5563;line-height:1.6;max-width:400px;
  margin:0 auto 20px;}}
.ask-hr-fcards{{display:flex;flex-direction:column;gap:8px;text-align:{"right" if rtl else "left"};margin-bottom:14px;}}
.ask-hr-fc{{background:white;border:1px solid #E0E5FF;border-radius:12px;
  padding:10px 13px;display:flex;align-items:center;gap:10px;flex-direction:{"row-reverse" if rtl else "row"};}}
.ask-hr-fc-ic{{width:30px;height:30px;border-radius:50%;display:flex;
  align-items:center;justify-content:center;font-size:14px;flex-shrink:0;}}
.ask-hr-fc-ic.blue{{background:#E8ECFE;}}
.ask-hr-fc-ic.red{{background:#FEE2E2;}}
.ask-hr-fc-ic.grn{{background:#ECFDF5;}}
.ask-hr-fc-ic.pur{{background:#EDE9FE;}}
.ask-hr-fc-txt{{font-size:12.5px;font-weight:600;color:#1A1A2E;flex:1;min-width:0;line-height:1.3;text-align:{"right" if rtl else "left"};word-break:break-word;}}
.ask-hr-fc-txt small{{font-weight:400;color:#6B7280;display:block;font-size:11px;}}
.ask-hr-fc-badge{{background:#E8ECFE;color:#2D3FE7;font-size:10.5px;font-weight:700;
  padding:2px 8px;border-radius:999px;flex-shrink:0;white-space:nowrap;}}
.ask-hr-fc-badge.ok{{background:#ECFDF5;color:#059669;}}
.ask-hr-power{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:13px;}}
.ask-hr-pow{{background:white;border:1.5px solid #2D3FE7;border-radius:12px;
  padding:10px 11px;display:flex;align-items:flex-start;gap:8px;text-align:{"right" if rtl else "left"};
  flex-direction:{"row-reverse" if rtl else "row"};}}
.ask-hr-pow.gov{{border-color:#059669;}}
.ask-hr-pow.lang{{border-color:#7C3AED;}}
.ask-hr-pow-ic{{font-size:16px;flex-shrink:0;margin-top:1px;}}
.ask-hr-pow-t{{font-size:11px;font-weight:700;color:#1A1A2E;margin-bottom:2px;}}
.ask-hr-pow-s{{font-size:10px;color:#6B7280;line-height:1.4;}}
.ask-hr-pow-tag{{display:inline-block;background:#E8ECFE;color:#2D3FE7;
  font-size:10px;font-weight:700;padding:2px 6px;border-radius:999px;margin-top:3px;}}
.ask-hr-pow.gov .ask-hr-pow-tag{{background:#ECFDF5;color:#059669;}}
.ask-hr-pow.lang .ask-hr-pow-tag{{background:#EDE9FE;color:#7C3AED;}}
.ask-hr-disc{{background:white;border:1px solid #E5E7EB;border-radius:10px;
  padding:9px 12px;font-size:11px;color:#6B7280;line-height:1.5;
  display:flex;gap:8px;align-items:flex-start;text-align:{"right" if rtl else "left"};
  flex-direction:{"row-reverse" if rtl else "row"};}}
</style>
<div class="ask-hr-hero">
  <div class="ask-hr-kicker">ASKLEPIOS · {"AI ΝΟΣΗΛΕΥΤΗΣ" if lang=="el" else "AI NURSE"}</div>
  <div class="ask-hr-h1">{t("hero_h1")}</div>
  <div class="ask-hr-sub">{t("hero_sub")}</div>
  <div class="ask-hr-fcards">
    <div class="ask-hr-fc">
      <div class="ask-hr-fc-ic blue">💬</div>
      <div class="ask-hr-fc-txt">{t("hero_f1t")}<small>{t("hero_f1s")}</small></div>
      <div class="ask-hr-fc-badge">{"Βήμα" if lang=="el" else "Step"} 1</div>
    </div>
    <div class="ask-hr-fc">
      <div class="ask-hr-fc-ic red">❤️</div>
      <div class="ask-hr-fc-txt">{t("hero_f2t")}<small>{t("hero_f2s")}</small></div>
      <div class="ask-hr-fc-badge">{"Βήμα" if lang=="el" else "Step"} 2</div>
    </div>
    <div class="ask-hr-fc">
      <div class="ask-hr-fc-ic pur">🧬</div>
      <div class="ask-hr-fc-txt">{t("hero_f3t")}<small>{t("hero_f3s")}</small></div>
      <div class="ask-hr-fc-badge">{"Βήμα" if lang=="el" else "Step"} 3</div>
    </div>
    <div class="ask-hr-fc">
      <div class="ask-hr-fc-ic grn">📋</div>
      <div class="ask-hr-fc-txt">{t("hero_f4t")}<small>{t("hero_f4s")}</small></div>
      <div class="ask-hr-fc-badge ok">✓ {"Αναφορά" if lang=="el" else "Report"}</div>
    </div>
  </div>
  <div class="ask-hr-power">
    <div class="ask-hr-pow">
      <div class="ask-hr-pow-ic">🤖</div>
      <div>
        <div class="ask-hr-pow-t">{t("hero_p1t")}</div>
        <div class="ask-hr-pow-s">{t("hero_p1s")}</div>
        <div class="ask-hr-pow-tag">{t("hero_p1b")}</div>
      </div>
    </div>
    <div class="ask-hr-pow gov">
      <div class="ask-hr-pow-ic">🇬🇷</div>
      <div>
        <div class="ask-hr-pow-t">{t("hero_p2t")}</div>
        <div class="ask-hr-pow-s">{t("hero_p2s")}</div>
        <div class="ask-hr-pow-tag">{t("hero_p2b")}</div>
      </div>
    </div>
    <div class="ask-hr-pow lang">
      <div class="ask-hr-pow-ic">🌍</div>
      <div>
        <div class="ask-hr-pow-t">{t("hero_p3t")}</div>
        <div class="ask-hr-pow-s">{t("hero_p3s")}</div>
        <div class="ask-hr-pow-tag">{t("hero_p3b")}</div>
      </div>
    </div>
  </div>
  <div class="ask-hr-disc">
    <span style="font-size:14px;flex-shrink:0">ℹ️</span>
    <span>{t("hero_disc")}</span>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── gov.gr QUICK ACCESS ───────────────────────────────────────────────────
    _gov_links = [
        ("📂", "Ηλεκτρονικός Φάκελος Υγείας" if lang=="el" else "Health Record",
         "https://www.gov.gr/ipiresies/ugeia-kai-pronoia/phakelos-ugeias"),

        ("👨\u200d⚕️", "Ιατροί ΕΟΠΥΥ" if lang=="el" else "EOPYY Doctors",
         "https://www.eopyy.gov.gr"),
        ("📋", "ΑΜΚΑ" if lang=="el" else "AMKA",
         "https://www.gov.gr/ipiresies/apasxolisi-kai-syntaxiodotisi/amka"),
        ("🔔", "ΕΟΔΥ" if lang=="el" else "EODY",
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
  padding:15px 17px;margin:0 0 18px;font-family:'Inter',system-ui,sans-serif;direction:{_dir};">
  <div style="font-size:11px;font-weight:700;letter-spacing:0.1em;color:#065F46;
    text-transform:uppercase;margin-bottom:11px;">{t("hero_gov_title")}</div>
  <div style="line-height:2;">{_gov_btn_html}</div>
  <div style="font-size:10.5px;color:#6B7280;margin-top:9px;">{t("hero_gov_note")}</div>
</div>
""", unsafe_allow_html=True)

    # ── HOW IT WORKS ─────────────────────────────────────────────────────────
    # NOTE: previously this truncated labels with hard character counts
    # (text[:14]+"…") tuned by eyeballing the English strings. That cut Greek/
    # Hindi/Urdu/Arabic words mid-glyph (e.g. "φωτογραφία" → "φω…") since other
    # scripts don't share English's chars-per-pixel ratio. Now we pass the full
    # text through and let CSS line-clamp (below) wrap/truncate visually,
    # which adapts correctly to any language and any font.
    _steps_data = [
        ("1","👤", t("stepper_profile").split(" ",1)[1] if " " in t("stepper_profile") else t("name"), t("history")),
        ("2","💬", t("triage_title"), t("triage_sub")),
        ("3","❤️", t("vitals_title"), "HR, BP, SpO₂"),
        ("4","🧬", t("hero_f3t"), t("hero_f3s")),
        ("5","🧠", "Triage AI", "Claude + GPT-4o"),
        ("6","📄", t("stepper_report").split(" ",1)[1] if " " in t("stepper_report") else "Report", "PubMed"),
    ]
    _arrow = f'<div style="flex:0 0 auto;align-self:flex-start;padding-top:14px;font-size:11px;color:#C7D2FE;">{"‹" if rtl else "›"}</div>'
    _steps_html = _arrow.join(f"""<div style="flex:1 1 0;min-width:0;text-align:{_ta};padding:0 2px;">
  <div style="width:28px;height:28px;border-radius:50%;background:#2D3FE7;color:white;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:12px;margin:0 auto 8px;">{n}</div>
  <div style="font-size:20px;margin-bottom:5px;">{ic}</div>
  <div style="font-size:12px;font-weight:700;color:#1A1A2E;margin-bottom:2px;line-height:1.25;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;word-break:break-word;">{tt}</div>
  <div style="font-size:10px;color:#6B7280;line-height:1.35;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden;word-break:break-word;">{ss}</div>
</div>""" for n, ic, tt, ss in _steps_data)
    st.markdown(f"""
<div style="font-family:'Inter',system-ui,sans-serif;margin:0 0 20px;direction:{_dir};">
  <div style="font-size:18px;font-weight:800;color:#1A1A2E;text-align:{_ta};margin-bottom:16px;">{t("hero_how")}</div>
  <div style="display:flex;align-items:flex-start;gap:0;width:100%;flex-direction:{"row-reverse" if rtl else "row"};">{_steps_html}</div>
</div>
""", unsafe_allow_html=True)

    # ── STATS BAND ────────────────────────────────────────────────────────────
    st.markdown(f"""
<div style="display:flex;border:1px solid #E0E5FF;border-radius:14px;overflow:hidden;
  background:white;margin:0 0 20px;font-family:'Inter',system-ui,sans-serif;direction:{_dir};">
  <div style="flex:1;text-align:{_ta};padding:13px 6px;border-{"left" if rtl else "right"}:1px solid #E0E5FF;">
    <div style="font-size:19px;font-weight:800;color:#2D3FE7;">88.9%</div>
    <div style="font-size:10.5px;color:#6B7280;margin-top:3px;line-height:1.3;">{t("hero_s1l")}</div>
  </div>
  <div style="flex:1;text-align:{_ta};padding:13px 6px;border-{"left" if rtl else "right"}:1px solid #E0E5FF;">
    <div style="font-size:19px;font-weight:800;color:#2D3FE7;">0%</div>
    <div style="font-size:10.5px;color:#6B7280;margin-top:3px;line-height:1.3;">{t("hero_s2l")}</div>
  </div>
  <div style="flex:1;text-align:{_ta};padding:13px 6px;">
    <div style="font-size:19px;font-weight:800;color:#2D3FE7;">2 AI</div>
    <div style="font-size:10.5px;color:#6B7280;margin-top:3px;line-height:1.3;">{t("hero_s3l")}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── FOR WHOM ─────────────────────────────────────────────────────────────
    _aud = [
        ("👨\u200d👩\u200d👧","#E8ECFE","#2D3FE7", t("hero_aud1t"), t("hero_aud1d"), t("hero_aud1b")),
        ("🤝","#FFFBEB","#92400E", t("hero_aud2t"), t("hero_aud2d"), t("hero_aud2b")),
        ("👨\u200d⚕️","#ECFDF5","#065F46", t("hero_aud3t"), t("hero_aud3d"), t("hero_aud3b")),
    ]
    _aud_cards = "".join(f"""<div style="flex:1 1 160px;background:white;border:1px solid #E0E5FF;border-radius:14px;padding:14px 14px 12px;direction:{_dir};text-align:{"right" if rtl else "left"};">
  <div style="width:32px;height:32px;border-radius:50%;background:{bg};display:flex;align-items:center;justify-content:center;font-size:16px;margin-bottom:9px;{"margin-right:0;margin-left:auto;" if rtl else ""}">{ic}</div>
  <div style="font-size:13px;font-weight:800;color:#1A1A2E;margin-bottom:4px;">{tt}</div>
  <div style="font-size:11.5px;color:#4B5563;line-height:1.5;margin-bottom:7px;">{dd}</div>
  <div style="display:inline-block;background:{bg};color:{tc};font-size:10px;font-weight:700;padding:2px 9px;border-radius:999px;">{badge}</div>
</div>""" for ic, bg, tc, tt, dd, badge in _aud)
    st.markdown(f"""
<div style="font-family:'Inter',system-ui,sans-serif;margin:0 0 22px;">
  <div style="font-size:18px;font-weight:800;color:#1A1A2E;text-align:{_ta};margin-bottom:16px;">{t("hero_for_whom")}</div>
  <div style="display:flex;gap:10px;flex-wrap:wrap;flex-direction:{"row-reverse" if rtl else "row"};">{_aud_cards}</div>
</div>
""", unsafe_allow_html=True)

    # ── LOGIN FORM / CONTINUE ─────────────────────────────────────────────────
    st.markdown(f"""
<div style="font-size:16px;font-weight:800;color:#1A1A2E;text-align:{_ta};
  margin:4px 0 12px;font-family:'Inter',system-ui,sans-serif;direction:{_dir};">{t("hero_login_title")}</div>
""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if is_logged_in():
            # Already authenticated — show CTA to proceed to home
            if st.button(t("hero_cta"), type="primary", use_container_width=True, key="hero_cta_loggedin"):
                st.session_state["_hero_seen"] = True
                st.rerun()
        elif auth_enabled():
            # Supabase is configured — require OTP login
            render_login_gate()
            # render_login_gate calls st.rerun() on success, which will
            # hit is_logged_in() above on the next render.
        else:
            # No Supabase (local dev) — CTA proceeds directly
            if st.button(t("hero_cta"), type="primary", use_container_width=True, key="hero_cta_noauth"):
                st.session_state["_hero_seen"] = True
                st.rerun()

    st.markdown(f'<div class="disclaimer" dir="{_dir}">{t("disclaimer_main")}</div>', unsafe_allow_html=True)


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
        # NOTE: eutils' esearch defaults to sorting by most-recent-publication-
        # date when no `sort` param is given — NOT by relevance, even though
        # that's what the pubmed.ncbi.nlm.nih.gov website shows by default.
        # We use "most+cited" rather than "relevance": relevance only scores
        # keyword-match strength in title/abstract/MeSH, so it can still rank
        # a niche case report above a well-established, heavily-cited review
        # just because the case report's text happens to match more closely.
        # Citation count is a much closer proxy for "this is strong, well-
        # validated literature" — combined with the existing _PILLAR_PTYPE
        # filter (Practice Guideline/Systematic Review/Meta-Analysis/Review),
        # this surfaces the most-cited paper *within* the high-evidence-type
        # subset, rather than just the most recent or most keyword-matched one.
        p = urllib.parse.urlencode({"db":"pubmed","term":query,"retmax":n,"retmode":"json","sort":"most+cited","api_key":get_ncbi_key()})
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

# pubmed_search()/esummary only gives title+journal+date — no abstract text.
# This pulls the actual abstract via efetch (XML) so the report-generation
# prompt can ground the clinical write-up in what these specific papers say,
# rather than the model filling in plan/citations purely from its own training
# knowledge with the title as a label. Best-effort: any PMID that fails to
# parse is just omitted, never raises.
def pubmed_fetch_abstracts(pmids, timeout=10):
    """Fetch abstract text for a list of PMIDs via NCBI efetch.
    Returns {pmid: abstract_text}. PMIDs with no abstract (e.g. some letters/
    editorials) or that fail to parse are simply absent from the result —
    callers should treat a missing key the same as 'no abstract available'."""
    if not pmids:
        return {}
    try:
        import xml.etree.ElementTree as _ET
        p = urllib.parse.urlencode({"db":"pubmed","id":",".join(pmids),"retmode":"xml","api_key":get_ncbi_key()})
        with urllib.request.urlopen(f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?{p}", timeout=timeout) as r:
            root = _ET.fromstring(r.read())
        out = {}
        for art in root.findall(".//PubmedArticle"):
            pmid_el = art.find(".//PMID")
            if pmid_el is None or not pmid_el.text:
                continue
            pmid = pmid_el.text.strip()
            # AbstractText can appear multiple times (structured abstracts:
            # Background/Methods/Results/Conclusion) — join them in order.
            parts = []
            for ab in art.findall(".//Abstract/AbstractText"):
                label = ab.get("Label")
                txt = (ab.text or "").strip()
                if not txt:
                    continue
                parts.append(f"{label}: {txt}" if label else txt)
            if parts:
                out[pmid] = " ".join(parts)
        return out
    except Exception:
        return {}

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

# ── MAIN BIBLIOGRAPHY (diagnosis-level evidence) ────────────────────────────
# This feeds the "🔬 PubMed" expander + PDF export — the section the user sees
# as the report's general references. It used to be a bare pubmed_search() on
# the condition name with no field/quality scoping, which let NCBI's relevance
# ranking surface tangential hits (e.g. a pediatric case report or an unrelated
# surgical-technique paper that merely shares a MeSH term with the diagnosis).
# Fix: search the condition in the Title field specifically (so the diagnosis
# has to be a primary subject, not an incidental mention) and prefer
# guideline/review-quality literature, same as the pillar searches do.
def bibliography_search(condition, n=3):
    """Diagnosis-level PubMed search for the main report bibliography.
    Returns the same list-of-dicts shape as pubmed_search. Tries, in order:
    1) condition in Title + high-evidence ptype (most specific)
    2) condition in Title, any ptype (still on-topic, just not guideline-tier)
    3) plain keyword search (last-resort, old behaviour) so we never show
       nothing when NCBI genuinely has no closely-titled paper."""
    if not condition:
        return []
    cond_q = condition.strip()
    title_scoped = f"{cond_q}[Title] AND {_PILLAR_PTYPE}"
    res = pubmed_search(title_scoped, n=n)
    if res:
        return res
    res = pubmed_search(f"{cond_q}[Title]", n=n)
    if res:
        return res
    return pubmed_search(cond_q, n=n)

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
        "subtitle": "Ο AI Νοσηλευτής σου",
        "tagline": "Έγκυρη ιατρική πληροφόρηση · Πάντα δίπλα σου",
        "start": "Ξεκίνα Εκτίμηση",
        "disclaimer_main": "⚠️ Ο Asklepios παρέχει πληροφορίες υγείας αποκλειστικά για ενημερωτικούς σκοπούς. Δεν αντικαθιστά ιατρική διάγνωση ή θεραπεία. Σε επείγουσα ανάγκη καλέστε **166** (ΕΚΑΒ) ή **112**.",
        "emergency": "🚨 ΣΕ ΕΠΕΙΓΟΥΣΑ ΑΝΑΓΚΗ: ΚΑΛΕΣΤΕ 166 (ΕΚΑΒ) ή 112",
        "name": "Όνομα", "age": "Ηλικία", "sex": "Φύλο",
        "male": "Άνδρας", "female": "Γυναίκα", "other": "Άλλο",
        "history": "Ιατρικό ιστορικό (προηγούμενες παθήσεις, χειρουργεία)",
        "allergies": "Αλλεργίες",
        "meds": "Τρέχοντα φάρμακα / συμπληρώματα",
        "next": "Επόμενο →",
        "back": "← Πίσω",
        "vitals_title": "Ζωτικές Ενδείξεις",
        "vitals_sub": "Εισάγετε τις μετρήσεις σας.",
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
        "triage_sub": "Περιγράψτε τα συμπτώματά σας. Ο Asklepios θα σας κάνει κατευθυνόμενες ερωτήσεις.",
        "triage_placeholder": "Π.χ. Έχω πονοκέφαλο τριών ημερών με ναυτία...",
        "generate_report": "Δημιουργία Πλήρους Αναφοράς",
        "report_title": "Λεπτομερής Εκτίμηση Υγείας",
        "second_opinion": "Δεύτερη Γνώμη GPT-4o",
        "pubmed": "Επιστημονικές Αναφορές PubMed",
        "skip_vitals": "Παράλειψη (χωρίς μετρήσεις)",
        "stepper_profile": '1 Στοιχεία',
        "stepper_vitals": '2 Ζωτικές',
        "stepper_symptoms": '3 Συμπτώματα',
        "stepper_report": '4 Αναφορά',
        "please_enter_name": 'Παρακαλώ βάλε το όνομά σου.',
        "bp_risk_title": 'Εκτίμηση Κινδύνου Αρτηριακής Πίεσης',
        "articles_label": 'Άρθρα',
        "read_more": 'Διάβασε περισσότερα',
        "triage_explainer": '👇 Βήμα 3 — Περίγραψε εδώ τι σε απασχολεί (π.χ. «πόνος στο μάτι 2 μέρες»). Ο Asklepios θα σου κάνει ερωτήσεις και στο τέλος θα δημιουργήσει αναφορά.',
        "triage_quick_select": 'Γρήγορη επιλογή',
        "triage_send_selected": 'Αποστολή επιλεγμένων',
        "triage_main_symptoms": 'Κύρια συμπτώματα: ',
        "vitals_core_title": 'Βασικές Μετρήσεις',
        "vitals_optional": 'Αναπνοή, βάρος, ύψος (προαιρετικά)',
        "vitals_skip_continue": 'Δεν έχω μετρήσεις — Συνέχεια στα συμπτώματα →',
        "voice_input_label": '🎤 Φωνητική εισαγωγή (μίλα αντί να πληκτρολογείς)',
        "home_explainer_title": 'Από πού ξεκινάω;',
        "home_explainer_body": 'Πάτα <strong>Έλεγχος Συμπτωμάτων</strong> για να ξεκινήσεις. Το Asklepios θα σε ρωτήσει για το προφίλ σου και μετά θα αξιολογήσει τα συμπτώματά σου βήμα-βήμα.',
        "home_symptoms_btn": 'Έλεγχος Συμπτωμάτων',
        "home_vitals_btn": 'Ζωτικά Σημεία',
        "home_emergency": 'Για πόνο στο στήθος, δυσκολία αναπνοής, σοβαρή αιμορραγία, απώλεια συνείδησης ή συμπτώματα εγκεφαλικού, καλέστε αμέσως 166 (ΕΚΑΒ) ή 112.',
        "home_emergency_label": 'Επείγον',
        "intake_for_whom": 'Για ποιον είναι αυτή η αξιολόγηση;',
        "intake_for_me": 'Για μένα',
        "intake_for_other": 'Για άλλο άτομο που φροντίζω',
        "intake_tell_us": 'Πες μας λίγα για σένα',
        "intake_tell_us_sub": 'Όνομα, ηλικία, ιατρικό ιστορικό',
        "nav_home": 'Αρχική',
        "nav_vitals": 'Ζωτικά',
        "nav_symptoms": 'Συμπτώματα',
        "nav_history": 'Ιστορικό',
        "hero_h1": "Περίγραψε τι νιώθεις.<br><span style='color:#2D3FE7'>Λάβε κλινική εκτίμηση.</span>",
        "hero_sub": 'Τεκμηριωμένη αξιολόγηση με αναφορές PubMed + δεύτερη γνώμη GPT-4o. Για τον <strong>ιατρό</strong> σου.',
        "hero_f1t": 'Περιγραφή συμπτωμάτων',
        "hero_f1s": 'Μιλάς φυσικά — το AI ρωτά & οργανώνει',
        "hero_f2t": 'Καταγραφή ζωτικών',
        "hero_f2s": 'HR, BP, SpO₂, θερμοκρασία',
        "hero_f3t": 'Εξετάσεις & φωτογραφία',
        "hero_f3s": 'Ανέβασε αιματολογικά, PDF ή φωτογραφία',
        "hero_f4t": 'Κλινική αναφορά για τον ιατρό',
        "hero_f4s": 'PubMed + δεύτερη γνώμη GPT-4o',
        "hero_p1t": 'Δεύτερη ιατρική γνώμη',
        "hero_p1s": 'Claude + GPT-4o ανεξάρτητα',
        "hero_p1b": 'Μοναδικό',
        "hero_p2t": 'Σύνδεση gov.gr',
        "hero_p2s": 'Ηλεκτρ. Φάκελος, ΑΜΚΑ, e-Συνταγ.',
        "hero_p2b": 'Ελληνικό ΣΥ',
        "hero_p3t": '16 γλώσσες',
        "hero_p3s": 'Ελληνικά, Αγγλικά, Χίντι, Αραβικά κ.ά.',
        "hero_p3b": 'Για όλους',
        "hero_disc": 'Δεν αντικαθιστά τον <strong>ιατρό</strong>. Επείγον: <strong>166</strong> ή <strong>112</strong>. Δεν αποθηκεύουμε ιατρικά δεδομένα. 🔒 GDPR',
        "hero_how": 'Πώς λειτουργεί',
        "hero_cta": '✦ Ξεκίνα αξιολόγηση & αναφορά',
        "hero_login_title": 'Ξεκίνα — δωρεάν, χωρίς password',
        "hero_for_whom": 'Για ποιον είναι',
        "hero_aud1t": 'Για όλη την οικογένεια',
        "hero_aud1d": 'Για σένα, τα παιδιά σου, τους γονείς σου. Πήγαινε στον ιατρό έτοιμος.',
        "hero_aud1b": 'Ενήλικες · Παιδιά · Ηλικιωμένοι',
        "hero_aud2t": 'Για φροντιστές υγείας',
        "hero_aud2d": 'Φροντίζεις κάποιον άλλο; Το Asklepios δουλεύει σε caregiver mode.',
        "hero_aud2b": 'Caregiver mode',
        "hero_aud3t": 'Για ιατρούς & ιατρεία',
        "hero_aud3d": 'Ο ασθενής φτάνει με οργανωμένο ιστορικό. Λιγότερα τηλεφωνήματα, ποιοτικότερος χρόνος.',
        "hero_aud3b": 'Εξοικονόμηση χρόνου',
        "hero_s1l": 'Ακρίβεια triage (Semigran-45)',
        "hero_s2l": 'Unsafe undertriage',
        "hero_s3l": 'Claude + GPT-4o δεύτερη γνώμη',
        "hero_gov_title": '🔒 Ηλεκτρονικές Υπηρεσίες Υγείας (gov.gr)',
        "hero_gov_note": 'Ανοίγουν σε νέα καρτέλα στο gov.gr — δεν αποθηκεύουμε δεδομένα.',
    },
    "en": {
        "title": "Asklepios",
        "subtitle": "Your AI Nurse",
        "tagline": "Evidence-based health guidance · Always by your side",
        "start": "Start Assessment",
        "disclaimer_main": "⚠️ Asklepios provides health information for informational purposes only. It does not replace medical diagnosis or treatment. In an emergency call **166** (EKAB) or **112**.",
        "emergency": "🚨 EMERGENCY: CALL 166 (EKAB) or 112",
        "name": "Name", "age": "Age", "sex": "Biological Sex",
        "male": "Male", "female": "Female", "other": "Other",
        "history": "Medical history (conditions, surgeries)",
        "allergies": "Allergies",
        "meds": "Current medications / supplements",
        "next": "Next →",
        "back": "← Back",
        "vitals_title": "Your Vitals",
        "vitals_sub": "Enter your measurements.",
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
        "triage_sub": "Describe your symptoms. Asklepios will ask targeted follow-up questions.",
        "triage_placeholder": "E.g. I have had a headache for three days with nausea...",
        "generate_report": "Generate Full Clinical Report",
        "report_title": "Detailed Health Assessment",
        "second_opinion": "GPT-4o Second Opinion",
        "pubmed": "PubMed Evidence",
        "skip_vitals": "Skip (no measurements)",
        "stepper_profile": '1 Profile',
        "stepper_vitals": '2 Vitals',
        "stepper_symptoms": '3 Symptoms',
        "stepper_report": '4 Report',
        "please_enter_name": 'Please enter your name.',
        "bp_risk_title": 'Blood Pressure Risk Estimate',
        "articles_label": 'Articles',
        "read_more": 'Read more',
        "triage_explainer": "👇 Step 3 — Describe what's bothering you (e.g. 'eye pain for 2 days'). Asklepios will ask follow-up questions and then generate a report.",
        "triage_quick_select": 'Quick select',
        "triage_send_selected": 'Send selected',
        "triage_main_symptoms": 'Main symptoms: ',
        "vitals_core_title": 'Core Vitals',
        "vitals_optional": 'Breathing, weight, height (optional)',
        "vitals_skip_continue": "I don't need vitals — Continue to symptoms →",
        "voice_input_label": '🎤 Voice input (speak instead of typing)',
        "home_explainer_title": 'Where do I start?',
        "home_explainer_body": 'Tap <strong>Check Symptoms</strong> to begin. Asklepios will ask for your profile and then assess your symptoms step by step.',
        "home_symptoms_btn": 'Symptom Check',
        "home_vitals_btn": 'Vital Signs',
        "home_emergency": 'For chest pain, difficulty breathing, severe bleeding, loss of consciousness, or stroke symptoms, call 166 (EKAB) or 112 immediately.',
        "home_emergency_label": 'Emergency',
        "intake_for_whom": 'Who is this assessment for?',
        "intake_for_me": 'For me',
        "intake_for_other": 'For someone I care for',
        "intake_tell_us": 'Tell us about yourself',
        "intake_tell_us_sub": 'Name, age, medical history',
        "nav_home": 'Home',
        "nav_vitals": 'Vitals',
        "nav_symptoms": 'Symptoms',
        "nav_history": 'History',
        "hero_h1": "Describe what you feel.<br><span style='color:#2D3FE7'>Get a clinical assessment.</span>",
        "hero_sub": 'Evidence-based assessment with PubMed references + GPT-4o second opinion. For your <strong>doctor</strong>.',
        "hero_f1t": 'Symptom description',
        "hero_f1s": 'Speak naturally — AI asks & organises',
        "hero_f2t": 'Vital signs',
        "hero_f2s": 'HR, BP, SpO₂, temperature',
        "hero_f3t": 'Lab results & photos',
        "hero_f3s": 'Upload blood tests, PDF results or a photo',
        "hero_f4t": 'Clinical report for your doctor',
        "hero_f4s": 'PubMed + GPT-4o second opinion',
        "hero_p1t": 'Second medical opinion',
        "hero_p1s": 'Claude + GPT-4o independently',
        "hero_p1b": 'Unique',
        "hero_p2t": 'gov.gr integration',
        "hero_p2s": 'Health record, AMKA, e-Prescription',
        "hero_p2b": 'Greek NHS',
        "hero_p3t": '16 languages',
        "hero_p3s": 'Greek, English, Hindi, Arabic, Hebrew & more',
        "hero_p3b": 'For everyone',
        "hero_disc": 'Does not replace your <strong>doctor</strong>. Emergency: <strong>112</strong>. We store no medical data. 🔒 GDPR',
        "hero_how": 'How it works',
        "hero_cta": '✦ Start assessment & report',
        "hero_login_title": 'Get started — free, no password',
        "hero_for_whom": 'Who is it for',
        "hero_aud1t": 'For the whole family',
        "hero_aud1d": 'For you, your children, your parents. Go to your doctor prepared.',
        "hero_aud1b": 'Adults · Children · Elderly',
        "hero_aud2t": 'For caregivers',
        "hero_aud2d": 'Caring for someone else? Asklepios works in caregiver mode.',
        "hero_aud2b": 'Caregiver mode',
        "hero_aud3t": 'For doctors & clinics',
        "hero_aud3d": 'Patients arrive with organised history. Fewer routine calls, better quality time.',
        "hero_aud3b": 'Save time',
        "hero_s1l": 'Triage accuracy (Semigran-45)',
        "hero_s2l": 'Unsafe undertriage',
        "hero_s3l": 'Claude + GPT-4o second opinion',
        "hero_gov_title": '🔒 Digital Health Services (gov.gr)',
        "hero_gov_note": 'Open in a new tab on gov.gr — we store no data.',
    },
    "hi": {
        "title": "Asklepios",
        "subtitle": "आपकी AI नर्स",
        "tagline": "विश्वसनीय स्वास्थ्य जानकारी · हमेशा आपके साथ",
        "start": "मूल्यांकन शुरू करें",
        "disclaimer_main": "⚠️ Asklepios केवल जानकारी के लिए स्वास्थ्य सूचना प्रदान करता है। यह चिकित्सा निदान या उपचार का विकल्प नहीं है। आपात स्थिति में **166** (EKAB) या **112** पर कॉल करें।",
        "emergency": "🚨 आपातकाल: 166 (EKAB) या 112 पर कॉल करें",
        "name": "नाम", "age": "उम्र", "sex": "लिंग",
        "male": "पुरुष", "female": "महिला", "other": "अन्य",
        "history": "चिकित्सा इतिहास (बीमारियाँ, ऑपरेशन)",
        "allergies": "एलर्जी",
        "meds": "वर्तमान दवाएं / सप्लीमेंट",
        "next": "अगला →",
        "back": "← वापस",
        "vitals_title": "महत्वपूर्ण संकेत",
        "vitals_sub": "अपनी माप दर्ज करें।",
        "hr": "हृदय गति (bpm)",
        "bp_sys": "रक्तचाप — सिस्टोलिक (mmHg)",
        "bp_dia": "रक्तचाप — डायस्टोलिक (mmHg)",
        "br": "श्वसन दर (/min)",
        "spo2": "SpO2 (%)",
        "temp": "तापमान (°C)",
        "weight": "वजन (kg)",
        "height": "ऊंचाई (cm)",
        "analyse_vitals": "संकेतों का विश्लेषण करें",
        "triage_title": "लक्षण मूल्यांकन",
        "triage_sub": "अपने लक्षण बताएं। Asklepios आपसे प्रश्न पूछेगा।",
        "triage_placeholder": "जैसे: तीन दिनों से सिरदर्द और मतली है...",
        "generate_report": "पूरी रिपोर्ट बनाएं",
        "report_title": "विस्तृत स्वास्थ्य मूल्यांकन",
        "second_opinion": "GPT-4o दूसरी राय",
        "pubmed": "PubMed शोध संदर्भ",
        "skip_vitals": "छोड़ें (माप के बिना)",
        "stepper_profile": '1 प्रोफ़ाइल',
        "stepper_vitals": '2 संकेत',
        "stepper_symptoms": '3 लक्षण',
        "stepper_report": '4 रिपोर्ट',
        "please_enter_name": 'कृपया अपना नाम दर्ज करें।',
        "bp_risk_title": 'रक्तचाप जोखिम अनुमान',
        "articles_label": 'लेख',
        "read_more": 'और पढ़ें',
        "triage_explainer": "👇 चरण 3 — यहाँ बताएं क्या परेशान कर रहा है (जैसे 'आँख में दर्द 2 दिन से')। Asklepios सवाल पूछेगा और फिर रिपोर्ट बनाएगा।",
        "triage_quick_select": 'त्वरित चयन',
        "triage_send_selected": 'चुने हुए भेजें',
        "triage_main_symptoms": 'मुख्य लक्षण: ',
        "vitals_core_title": 'मुख्य संकेत',
        "vitals_optional": 'श्वास, वजन, ऊंचाई (वैकल्पिक)',
        "vitals_skip_continue": 'मुझे संकेत नहीं चाहिए — लक्षणों पर जारी रखें →',
        "voice_input_label": '🎤 आवाज़ इनपुट (टाइप करने की बजाय बोलें)',
        "home_explainer_title": 'मैं कहाँ से शुरू करूँ?',
        "home_explainer_body": '<strong>लक्षण जाँच</strong> पर टैप करें। Asklepios आपका प्रोफ़ाइल पूछेगा और फिर आपके लक्षणों का मूल्यांकन करेगा।',
        "home_symptoms_btn": 'लक्षण जाँच',
        "home_vitals_btn": 'महत्वपूर्ण संकेत',
        "home_emergency": 'सीने में दर्द, सांस लेने में कठिनाई, गंभीर रक्तस्राव या बेहोशी के लिए तुरंत 166 (EKAB) या 112 पर कॉल करें।',
        "home_emergency_label": 'आपातकाल',
        "intake_for_whom": 'यह मूल्यांकन किसके लिए है?',
        "intake_for_me": 'मेरे लिए',
        "intake_for_other": 'मेरी देखभाल में किसी के लिए',
        "intake_tell_us": 'अपने बारे में बताएं',
        "intake_tell_us_sub": 'नाम, उम्र, चिकित्सा इतिहास',
        "nav_home": 'होम',
        "nav_vitals": 'संकेत',
        "nav_symptoms": 'लक्षण',
        "nav_history": 'इतिहास',
        "hero_h1": "बताएं आप क्या महसूस कर रहे हैं।<br><span style='color:#2D3FE7'>पाएं नैदानिक मूल्यांकन।</span>",
        "hero_sub": 'PubMed संदर्भों के साथ + GPT-4o दूसरी राय। आपके <strong>डॉक्टर</strong> के लिए।',
        "hero_f1t": 'लक्षणों का विवरण',
        "hero_f1s": 'स्वाभाविक रूप से बोलें — AI पूछता और व्यवस्थित करता है',
        "hero_f2t": 'महत्वपूर्ण संकेत',
        "hero_f2s": 'HR, BP, SpO₂, तापमान',
        "hero_f3t": 'परीक्षण और फोटो',
        "hero_f3s": 'रक्त परीक्षण, PDF या फोटो अपलोड करें',
        "hero_f4t": 'डॉक्टर के लिए रिपोर्ट',
        "hero_f4s": 'PubMed + GPT-4o दूसरी राय',
        "hero_p1t": 'दूसरी चिकित्सा राय',
        "hero_p1s": 'Claude + GPT-4o स्वतंत्र रूप से',
        "hero_p1b": 'अनोखा',
        "hero_p2t": 'gov.gr से जुड़ाव',
        "hero_p2s": 'स्वास्थ्य रिकॉर्ड, AMKA, e-Prescription',
        "hero_p2b": 'Greek NHS',
        "hero_p3t": '16 भाषाएं',
        "hero_p3s": 'हिंदी, उर्दू, अरबी, बंगाली सहित',
        "hero_p3b": 'सबके लिए',
        "hero_disc": 'डॉक्टर का विकल्प नहीं। आपातकाल: <strong>166</strong> या <strong>112</strong>। 🔒 GDPR',
        "hero_how": 'यह कैसे काम करता है',
        "hero_cta": '✦ मूल्यांकन शुरू करें',
        "hero_login_title": 'शुरू करें — मुफ्त, बिना पासवर्ड',
        "hero_for_whom": 'किसके लिए है',
        "hero_aud1t": 'पूरे परिवार के लिए',
        "hero_aud1d": 'आपके, बच्चों और माता-पिता के लिए। डॉक्टर के पास तैयार जाएं।',
        "hero_aud1b": 'वयस्क · बच्चे · बुजुर्ग',
        "hero_aud2t": 'देखभाल करने वालों के लिए',
        "hero_aud2d": 'किसी और की देखभाल? Caregiver mode में काम करता है।',
        "hero_aud2b": 'Caregiver mode',
        "hero_aud3t": 'डॉक्टरों के लिए',
        "hero_aud3d": 'मरीज संगठित इतिहास के साथ आते हैं। कम routine calls।',
        "hero_aud3b": 'समय की बचत',
        "hero_s1l": 'Triage सटीकता (Semigran-45)',
        "hero_s2l": 'Unsafe undertriage',
        "hero_s3l": 'Claude + GPT-4o दूसरी राय',
        "hero_gov_title": '🔒 Digital Health Services (gov.gr)',
        "hero_gov_note": 'gov.gr पर नई टैब में खुलता है।',
    },
    "ur": {
        "title": "Asklepios",
        "subtitle": "آپ کی AI نرس",
        "tagline": "قابل اعتماد صحت کی معلومات · ہمیشہ آپ کے ساتھ",
        "start": "تشخیص شروع کریں",
        "disclaimer_main": "⚠️ Asklepios صرف معلوماتی مقاصد کے لیے صحت کی معلومات فراہم کرتا ہے۔ یہ طبی تشخیص یا علاج کا متبادل نہیں ہے۔ ہنگامی صورت میں **166** (EKAB) یا **112** پر کال کریں۔",
        "emergency": "🚨 ہنگامی صورتحال: 166 یا 112 پر کال کریں",
        "name": "نام", "age": "عمر", "sex": "جنس",
        "male": "مرد", "female": "عورت", "other": "دیگر",
        "history": "طبی تاریخ (بیماریاں، آپریشن)",
        "allergies": "الرجی",
        "meds": "موجودہ دوائیں / سپلیمنٹ",
        "next": "اگلا →",
        "back": "← واپس",
        "vitals_title": "اہم علامات",
        "vitals_sub": "اپنی پیمائش درج کریں۔",
        "hr": "دل کی دھڑکن (bpm)",
        "bp_sys": "بلڈ پریشر — سیسٹولک (mmHg)",
        "bp_dia": "بلڈ پریشر — ڈائاسٹولک (mmHg)",
        "br": "سانس کی شرح (/min)",
        "spo2": "SpO2 (%)",
        "temp": "درجہ حرارت (°C)",
        "weight": "وزن (kg)",
        "height": "قد (cm)",
        "analyse_vitals": "علامات کا تجزیہ کریں",
        "triage_title": "علامات کی تشخیص",
        "triage_sub": "اپنی علامات بیان کریں۔ Asklepios سوالات پوچھے گا۔",
        "triage_placeholder": "مثلاً: تین دن سے سردرد اور متلی ہے...",
        "generate_report": "مکمل رپورٹ بنائیں",
        "report_title": "تفصیلی صحت کی تشخیص",
        "second_opinion": "GPT-4o دوسری رائے",
        "pubmed": "PubMed تحقیقی حوالہ جات",
        "skip_vitals": "چھوڑیں (پیمائش کے بغیر)",
        "stepper_profile": '1 پروفائل',
        "stepper_vitals": '2 علامات',
        "stepper_symptoms": '3 علامات',
        "stepper_report": '4 رپورٹ',
        "please_enter_name": 'براہ کرم اپنا نام درج کریں۔',
        "bp_risk_title": 'بلڈ پریشر خطرہ تخمینہ',
        "articles_label": 'مضامین',
        "read_more": 'مزید پڑھیں',
        "triage_explainer": "👇 مرحلہ 3 — یہاں بتائیں کیا پریشان کر رہا ہے (جیسے 'آنکھ میں درد 2 دن سے')۔ Asklepios سوال پوچھے گا اور پھر رپورٹ بنائے گا۔",
        "triage_quick_select": 'فوری انتخاب',
        "triage_send_selected": 'منتخب بھیجیں',
        "triage_main_symptoms": 'اہم علامات: ',
        "vitals_core_title": 'اہم علامات',
        "vitals_optional": 'سانس، وزن، قد (اختیاری)',
        "vitals_skip_continue": 'مجھے علامات نہیں چاہییں — علامات پر جاری رکھیں →',
        "voice_input_label": '🎤 آواز ان پٹ (ٹائپ کرنے کی بجائے بولیں)',
        "home_explainer_title": 'میں کہاں سے شروع کروں؟',
        "home_explainer_body": '<strong>علامات جانچ</strong> پر ٹیپ کریں۔ Asklepios آپ کا پروفائل پوچھے گا اور پھر آپ کی علامات کا جائزہ لے گا۔',
        "home_symptoms_btn": 'علامات جانچ',
        "home_vitals_btn": 'اہم علامات',
        "home_emergency": 'سینے میں درد، سانس لینے میں دشواری یا بے ہوشی کے لیے فوری 166 یا 112 پر کال کریں۔',
        "home_emergency_label": 'ہنگامی',
        "intake_for_whom": 'یہ تشخیص کس کے لیے ہے؟',
        "intake_for_me": 'میرے لیے',
        "intake_for_other": 'جس کی میں دیکھ بھال کرتا ہوں',
        "intake_tell_us": 'اپنے بارے میں بتائیں',
        "intake_tell_us_sub": 'نام، عمر، طبی تاریخ',
        "nav_home": 'ہوم',
        "nav_vitals": 'علامات',
        "nav_symptoms": 'علامات',
        "nav_history": 'تاریخ',
        "hero_h1": "بتائیں آپ کیا محسوس کر رہے ہیں۔<br><span style='color:#2D3FE7'>طبی تشخیص حاصل کریں۔</span>",
        "hero_sub": 'PubMed + GPT-4o دوسری رائے۔ آپ کے <strong>ڈاکٹر</strong> کے لیے۔',
        "hero_f1t": 'علامات کی وضاحت',
        "hero_f1s": 'قدرتی طریقے سے بولیں — AI پوچھتا اور منظم کرتا ہے',
        "hero_f2t": 'اہم علامات',
        "hero_f2s": 'HR, BP, SpO₂, درجہ حرارت',
        "hero_f3t": 'ٹیسٹ اور تصاویر',
        "hero_f3s": 'خون کے ٹیسٹ، PDF یا تصویر اپلوڈ کریں',
        "hero_f4t": 'ڈاکٹر کے لیے رپورٹ',
        "hero_f4s": 'PubMed + GPT-4o دوسری رائے',
        "hero_p1t": 'دوسری طبی رائے',
        "hero_p1s": 'Claude + GPT-4o آزادانہ',
        "hero_p1b": 'منفرد',
        "hero_p2t": 'gov.gr سے تعلق',
        "hero_p2s": 'صحت کا ریکارڈ، AMKA، e-نسخہ',
        "hero_p2b": 'Greek NHS',
        "hero_p3t": '16 زبانیں',
        "hero_p3s": 'اردو، عربی، ہندی، بنگالی سمیت',
        "hero_p3b": 'سب کے لیے',
        "hero_disc": 'ڈاکٹر کا متبادل نہیں۔ ہنگامی: <strong>166</strong> یا <strong>112</strong>۔ 🔒 GDPR',
        "hero_how": 'یہ کیسے کام کرتا ہے',
        "hero_cta": '✦ تشخیص شروع کریں',
        "hero_login_title": 'شروع کریں — مفت، بغیر پاس ورڈ',
        "hero_for_whom": 'کس کے لیے ہے',
        "hero_aud1t": 'پورے خاندان کے لیے',
        "hero_aud1d": 'آپ، بچوں اور والدین کے لیے۔ ڈاکٹر کے پاس تیار جائیں۔',
        "hero_aud1b": 'بالغ · بچے · بزرگ',
        "hero_aud2t": 'دیکھ بھال کرنے والوں کے لیے',
        "hero_aud2d": 'کسی اور کی دیکھ بھال؟ Caregiver mode میں کام کرتا ہے۔',
        "hero_aud2b": 'Caregiver mode',
        "hero_aud3t": 'ڈاکٹروں کے لیے',
        "hero_aud3d": 'مریض منظم تاریخ کے ساتھ آتے ہیں۔ کم routine calls۔',
        "hero_aud3b": 'وقت کی بچت',
        "hero_s1l": 'Triage درستگی (Semigran-45)',
        "hero_s2l": 'Unsafe undertriage',
        "hero_s3l": 'Claude + GPT-4o دوسری رائے',
        "hero_gov_title": '🔒 Digital Health Services (gov.gr)',
        "hero_gov_note": 'gov.gr پر نئی ٹیب میں کھلتا ہے۔',
    },
    "ar": {
        "title": "Asklepios",
        "subtitle": "ممرضتك الذكية",
        "tagline": "معلومات صحية موثوقة · دائماً بجانبك",
        "start": "ابدأ التقييم",
        "disclaimer_main": "⚠️ يقدم Asklepios معلومات صحية لأغراض إعلامية فقط. لا يُعوِّض التشخيص الطبي أو العلاج. في حالات الطوارئ اتصل بـ **166** (EKAB) أو **112**.",
        "emergency": "🚨 طوارئ: اتصل بـ 166 أو 112",
        "name": "الاسم", "age": "العمر", "sex": "الجنس",
        "male": "ذكر", "female": "أنثى", "other": "آخر",
        "history": "التاريخ الطبي (الأمراض، العمليات)",
        "allergies": "الحساسية",
        "meds": "الأدوية الحالية / المكملات",
        "next": "التالي →",
        "back": "← رجوع",
        "vitals_title": "العلامات الحيوية",
        "vitals_sub": "أدخل قياساتك.",
        "hr": "معدل ضربات القلب (bpm)",
        "bp_sys": "ضغط الدم — الانقباضي (mmHg)",
        "bp_dia": "ضغط الدم — الانبساطي (mmHg)",
        "br": "معدل التنفس (/min)",
        "spo2": "SpO2 (%)",
        "temp": "درجة الحرارة (°C)",
        "weight": "الوزن (kg)",
        "height": "الطول (cm)",
        "analyse_vitals": "تحليل العلامات الحيوية",
        "triage_title": "تقييم الأعراض",
        "triage_sub": "صف أعراضك. سيطرح عليك Asklepios أسئلة موجهة.",
        "triage_placeholder": "مثلاً: أعاني من صداع منذ ثلاثة أيام مع غثيان...",
        "generate_report": "إنشاء تقرير طبي كامل",
        "report_title": "تقييم صحي مفصل",
        "second_opinion": "رأي ثانٍ من GPT-4o",
        "pubmed": "مراجع PubMed العلمية",
        "skip_vitals": "تخطَّ (بدون قياسات)",
        "stepper_profile": '1 الملف',
        "stepper_vitals": '2 الحيوية',
        "stepper_symptoms": '3 الأعراض',
        "stepper_report": '4 التقرير',
        "please_enter_name": 'يرجى إدخال اسمك.',
        "bp_risk_title": 'تقدير خطر ضغط الدم',
        "articles_label": 'مقالات',
        "read_more": 'اقرأ أكثر',
        "triage_explainer": "👇 الخطوة 3 — صف هنا ما يزعجك (مثل 'ألم في العين منذ يومين'). سيطرح عليك Asklepios أسئلة ثم ينشئ تقريراً.",
        "triage_quick_select": 'اختيار سريع',
        "triage_send_selected": 'إرسال المختار',
        "triage_main_symptoms": 'الأعراض الرئيسية: ',
        "vitals_core_title": 'العلامات الحيوية الأساسية',
        "vitals_optional": 'التنفس، الوزن، الطول (اختياري)',
        "vitals_skip_continue": 'لا أحتاج علامات حيوية — المتابعة للأعراض →',
        "voice_input_label": '🎤 إدخال صوتي (تحدث بدلاً من الكتابة)',
        "home_explainer_title": 'من أين أبدأ؟',
        "home_explainer_body": 'اضغط على <strong>فحص الأعراض</strong>. سيسألك Asklepios عن ملفك الشخصي ثم يقيّم أعراضك خطوة بخطوة.',
        "home_symptoms_btn": 'فحص الأعراض',
        "home_vitals_btn": 'العلامات الحيوية',
        "home_emergency": 'لألم الصدر أو صعوبة التنفس أو فقدان الوعي، اتصل فوراً بـ 166 أو 112.',
        "home_emergency_label": 'طوارئ',
        "intake_for_whom": 'لمن هذا التقييم؟',
        "intake_for_me": 'لي',
        "intake_for_other": 'لشخص أرعاه',
        "intake_tell_us": 'أخبرنا عن نفسك',
        "intake_tell_us_sub": 'الاسم، العمر، التاريخ الطبي',
        "nav_home": 'الرئيسية',
        "nav_vitals": 'الحيوية',
        "nav_symptoms": 'الأعراض',
        "nav_history": 'التاريخ',
        "hero_h1": "صف ما تشعر به.<br><span style='color:#2D3FE7'>احصل على تقييم طبي.</span>",
        "hero_sub": 'تقييم مبني على الأدلة + رأي ثانٍ من GPT-4o. لـ<strong>طبيبك</strong>.',
        "hero_f1t": 'وصف الأعراض',
        "hero_f1s": 'تكلم بشكل طبيعي — AI يسأل وينظّم',
        "hero_f2t": 'العلامات الحيوية',
        "hero_f2s": 'HR, BP, SpO₂, درجة الحرارة',
        "hero_f3t": 'فحوصات وصور',
        "hero_f3s": 'ارفع تحاليل الدم، PDF أو صورة',
        "hero_f4t": 'تقرير للطبيب',
        "hero_f4s": 'PubMed + رأي ثانٍ GPT-4o',
        "hero_p1t": 'رأي طبي ثانٍ',
        "hero_p1s": 'Claude + GPT-4o بشكل مستقل',
        "hero_p1b": 'فريد',
        "hero_p2t": 'تكامل مع gov.gr',
        "hero_p2s": 'السجل الصحي، AMKA، وصفات إلكترونية',
        "hero_p2b": 'NHS اليوناني',
        "hero_p3t": '16 لغة',
        "hero_p3s": 'العربية، الهندية، الأردية، العبرية وغيرها',
        "hero_p3b": 'للجميع',
        "hero_disc": 'لا يُعوِّض الطبيب. طوارئ: <strong>166</strong> أو <strong>112</strong>. 🔒 GDPR',
        "hero_how": 'كيف يعمل',
        "hero_cta": '✦ ابدأ التقييم',
        "hero_login_title": 'ابدأ — مجاناً، بدون كلمة مرور',
        "hero_for_whom": 'لمن هو',
        "hero_aud1t": 'للعائلة كاملة',
        "hero_aud1d": 'لك ولأطفالك ووالديك. اذهب إلى الطبيب مستعداً.',
        "hero_aud1b": 'بالغون · أطفال · مسنون',
        "hero_aud2t": 'لمقدمي الرعاية',
        "hero_aud2d": 'ترعى شخصاً آخر؟ يعمل في Caregiver mode.',
        "hero_aud2b": 'Caregiver mode',
        "hero_aud3t": 'للأطباء والعيادات',
        "hero_aud3d": 'المريض يصل بتاريخ منظم. مكالمات روتينية أقل.',
        "hero_aud3b": 'توفير الوقت',
        "hero_s1l": 'دقة الفرز (Semigran-45)',
        "hero_s2l": 'Unsafe undertriage',
        "hero_s3l": 'Claude + GPT-4o رأي ثانٍ',
        "hero_gov_title": '🔒 الخدمات الصحية الرقمية (gov.gr)',
        "hero_gov_note": 'تفتح في تبويب جديد على gov.gr.',
    },
    "bn": {
        "title": "Asklepios",
        "subtitle": "আপনার AI নার্স",
        "tagline": "নির্ভরযোগ্য স্বাস্থ্য তথ্য · সবসময় আপনার পাশে",
        "start": "মূল্যায়ন শুরু করুন",
        "disclaimer_main": "⚠️ Asklepios শুধুমাত্র তথ্যমূলক উদ্দেশ্যে স্বাস্থ্য তথ্য প্রদান করে। এটি চিকিৎসা নির্ণয় বা চিকিৎসার বিকল্প নয়। জরুরি অবস্থায় **166** (EKAB) বা **112** তে কল করুন।",
        "emergency": "🚨 জরুরি অবস্থা: 166 বা 112 তে কল করুন",
        "name": "নাম", "age": "বয়স", "sex": "লিঙ্গ",
        "male": "পুরুষ", "female": "মহিলা", "other": "অন্যান্য",
        "history": "চিকিৎসা ইতিহাস (রোগ, অপারেশন)",
        "allergies": "অ্যালার্জি",
        "meds": "বর্তমান ওষুধ / সাপ্লিমেন্ট",
        "next": "পরবর্তী →",
        "back": "← ফিরে যান",
        "vitals_title": "গুরুত্বপূর্ণ লক্ষণ",
        "vitals_sub": "আপনার পরিমাপ লিখুন।",
        "hr": "হৃদস্পন্দন (bpm)",
        "bp_sys": "রক্তচাপ — সিস্টোলিক (mmHg)",
        "bp_dia": "রক্তচাপ — ডায়াস্টোলিক (mmHg)",
        "br": "শ্বাসের হার (/min)",
        "spo2": "SpO2 (%)",
        "temp": "তাপমাত্রা (°C)",
        "weight": "ওজন (kg)",
        "height": "উচ্চতা (cm)",
        "analyse_vitals": "লক্ষণ বিশ্লেষণ করুন",
        "triage_title": "লক্ষণ মূল্যায়ন",
        "triage_sub": "আপনার লক্ষণ বর্ণনা করুন। Asklepios প্রশ্ন করবে।",
        "triage_placeholder": "যেমন: তিন দিন ধরে মাথাব্যথা এবং বমি ভাব...",
        "generate_report": "সম্পূর্ণ রিপোর্ট তৈরি করুন",
        "report_title": "বিস্তারিত স্বাস্থ্য মূল্যায়ন",
        "second_opinion": "GPT-4o দ্বিতীয় মতামত",
        "pubmed": "PubMed গবেষণা তথ্যসূত্র",
        "skip_vitals": "এড়িয়ে যান (পরিমাপ ছাড়া)",
        "stepper_profile": '1 প্রোফাইল',
        "stepper_vitals": '2 লক্ষণ',
        "stepper_symptoms": '3 উপসর্গ',
        "stepper_report": '4 রিপোর্ট',
        "please_enter_name": 'অনুগ্রহ করে আপনার নাম লিখুন।',
        "bp_risk_title": 'রক্তচাপ ঝুঁকি অনুমান',
        "articles_label": 'নিবন্ধ',
        "read_more": 'আরও পড়ুন',
        "triage_explainer": "👇 ধাপ 3 — এখানে বলুন কী সমস্যা হচ্ছে (যেমন 'চোখে ব্যথা ২ দিন')। Asklepios প্রশ্ন করবে এবং রিপোর্ট তৈরি করবে।",
        "triage_quick_select": 'দ্রুত নির্বাচন',
        "triage_send_selected": 'নির্বাচিত পাঠান',
        "triage_main_symptoms": 'প্রধান লক্ষণ: ',
        "vitals_core_title": 'মূল গুরুত্বপূর্ণ লক্ষণ',
        "vitals_optional": 'শ্বাস, ওজন, উচ্চতা (ঐচ্ছিক)',
        "vitals_skip_continue": 'আমার ভাইটাল দরকার নেই — লক্ষণে চালিয়ে যান →',
        "voice_input_label": '🎤 ভয়েস ইনপুট (টাইপের বদলে বলুন)',
        "home_explainer_title": 'আমি কোথা থেকে শুরু করব?',
        "home_explainer_body": '<strong>লক্ষণ পরীক্ষা</strong> ট্যাপ করুন। Asklepios আপনার প্রোফাইল জিজ্ঞেস করবে এবং তারপর ধাপে ধাপে লক্ষণ মূল্যায়ন করবে।',
        "home_symptoms_btn": 'লক্ষণ পরীক্ষা',
        "home_vitals_btn": 'গুরুত্বপূর্ণ লক্ষণ',
        "home_emergency": 'বুকে ব্যথা, শ্বাস নিতে কষ্ট বা চেতনা হারানোর জন্য অবিলম্বে 166 বা 112 কল করুন।',
        "home_emergency_label": 'জরুরি',
        "intake_for_whom": 'এই মূল্যায়ন কার জন্য?',
        "intake_for_me": 'আমার জন্য',
        "intake_for_other": 'যার যত্ন নিচ্ছি তার জন্য',
        "intake_tell_us": 'আপনার সম্পর্কে বলুন',
        "intake_tell_us_sub": 'নাম, বয়স, চিকিৎসা ইতিহাস',
        "nav_home": 'হোম',
        "nav_vitals": 'লক্ষণ',
        "nav_symptoms": 'উপসর্গ',
        "nav_history": 'ইতিহাস',
        "hero_h1": "বলুন আপনি কী অনুভব করছেন।<br><span style='color:#2D3FE7'>ক্লিনিক্যাল মূল্যায়ন পান।</span>",
        "hero_sub": 'PubMed রেফারেন্স + GPT-4o দ্বিতীয় মতামত সহ প্রমাণ-ভিত্তিক মূল্যায়ন। আপনার <strong>ডাক্তারের</strong> জন্য।',
        "hero_f1t": 'লক্ষণের বিবরণ',
        "hero_f1s": 'স্বাভাবিকভাবে বলুন — AI জিজ্ঞেস করে ও সাজায়',
        "hero_f2t": 'গুরুত্বপূর্ণ লক্ষণ',
        "hero_f2s": 'HR, BP, SpO₂, তাপমাত্রা',
        "hero_f3t": 'পরীক্ষা ও ছবি',
        "hero_f3s": 'রক্ত পরীক্ষা, PDF বা ছবি আপলোড করুন',
        "hero_f4t": 'ডাক্তারের জন্য রিপোর্ট',
        "hero_f4s": 'PubMed + GPT-4o দ্বিতীয় মতামত',
        "hero_p1t": 'দ্বিতীয় চিকিৎসা মতামত',
        "hero_p1s": 'Claude + GPT-4o স্বাধীনভাবে',
        "hero_p1b": 'অনন্য',
        "hero_p2t": 'gov.gr সংযোগ',
        "hero_p2s": 'স্বাস্থ্য রেকর্ড, AMKA, e-প্রেসক্রিপশন',
        "hero_p2b": 'Greek NHS',
        "hero_p3t": '16টি ভাষা',
        "hero_p3s": 'বাংলা, হিন্দি, আরবি এবং আরও',
        "hero_p3b": 'সবার জন্য',
        "hero_disc": 'ডাক্তারের বিকল্প নয়। জরুরি: <strong>166</strong> বা <strong>112</strong>। 🔒 GDPR',
        "hero_how": 'এটি কীভাবে কাজ করে',
        "hero_cta": '✦ মূল্যায়ন শুরু করুন',
        "hero_login_title": 'শুরু করুন — বিনামূল্যে, পাসওয়ার্ড ছাড়া',
        "hero_for_whom": 'কার জন্য',
        "hero_aud1t": 'পুরো পরিবারের জন্য',
        "hero_aud1d": 'আপনার, সন্তান ও বাবা-মার জন্য। ডাক্তারের কাছে প্রস্তুত যান।',
        "hero_aud1b": 'প্রাপ্তবয়স্ক · শিশু · বৃদ্ধ',
        "hero_aud2t": 'যত্নশীলদের জন্য',
        "hero_aud2d": 'অন্য কারো যত্ন নিচ্ছেন? Caregiver mode-এ কাজ করে।',
        "hero_aud2b": 'Caregiver mode',
        "hero_aud3t": 'ডাক্তার ও ক্লিনিকের জন্য',
        "hero_aud3d": 'রোগী সংগঠিত ইতিহাস নিয়ে আসে। কম রুটিন কল।',
        "hero_aud3b": 'সময় সাশ্রয়',
        "hero_s1l": 'Triage নির্ভুলতা (Semigran-45)',
        "hero_s2l": 'Unsafe undertriage',
        "hero_s3l": 'Claude + GPT-4o দ্বিতীয় মতামত',
        "hero_gov_title": '🔒 ডিজিটাল স্বাস্থ্য সেবা (gov.gr)',
        "hero_gov_note": 'gov.gr-এ নতুন ট্যাবে খোলে।',
    },
    "bg": {
        "title": "Asklepios", "subtitle": "Вашата AI медицинска сестра",
        "tagline": "Надеждна здравна информация · Винаги до вас",
        "start": "Започни оценка",
        "disclaimer_main": "⚠️ Asklepios предоставя здравна информация само за информационни цели. Не замества медицинска диагноза или лечение. При спешност се обадете на **166** (EKAB) или **112**.",
        "emergency": "🚨 СПЕШНО: ОБАДЕТЕ СЕ НА 166 ИЛИ 112",
        "name": "Име", "age": "Възраст", "sex": "Пол",
        "male": "Мъж", "female": "Жена", "other": "Друго",
        "history": "Медицинска история (заболявания, операции)", "allergies": "Алергии",
        "meds": "Текущи лекарства / добавки", "next": "Напред →", "back": "← Назад",
        "vitals_title": "Жизнени показатели", "vitals_sub": "Въведете измерванията си.",
        "hr": "Сърдечен ритъм (bpm)", "bp_sys": "Кръвно налягане — Систолично (mmHg)",
        "bp_dia": "Кръвно налягане — Диастолично (mmHg)", "br": "Дихателна честота (/min)",
        "spo2": "SpO2 (%)", "temp": "Температура (°C)", "weight": "Тегло (kg)", "height": "Ръст (cm)",
        "analyse_vitals": "Анализ на показателите", "triage_title": "Оценка на симптомите",
        "triage_sub": "Опишете симптомите си. Asklepios ще задава въпроси.",
        "triage_placeholder": "Напр. Имам главоболие от три дни с гадене...",
        "generate_report": "Генерирай пълен доклад", "report_title": "Подробна здравна оценка",
        "second_opinion": "Второ мнение GPT-4o", "pubmed": "PubMed научни референции",
        "skip_vitals": "Пропусни (без измервания)",
        "stepper_profile": '1 Профил',
        "stepper_vitals": '2 Показатели',
        "stepper_symptoms": '3 Симптоми',
        "stepper_report": '4 Доклад',
        "please_enter_name": 'Моля, въведете вашето име.',
        "bp_risk_title": 'Оценка на риска от кръвно налягане',
        "articles_label": 'Статии',
        "read_more": 'Прочети повече',
        "triage_explainer": "👇 Стъпка 3 — Опишете какво ви притеснява (напр. 'болка в окото 2 дни'). Asklepios ще задава въпроси и ще изготви доклад.",
        "triage_quick_select": 'Бърз избор',
        "triage_send_selected": 'Изпрати избраните',
        "triage_main_symptoms": 'Основни симптоми: ',
        "vitals_core_title": 'Основни жизнени показатели',
        "vitals_optional": 'Дишане, тегло, ръст (незадължително)',
        "vitals_skip_continue": 'Нямам измервания — Продължи към симптоми →',
        "voice_input_label": '🎤 Гласов вход (говорете вместо да пишете)',
        "home_explainer_title": 'Откъде да започна?',
        "home_explainer_body": 'Натиснете <strong>Оценка на симптоми</strong>. Asklepios ще попита за вашия профил и след това ще оцени симптомите ви стъпка по стъпка.',
        "home_symptoms_btn": 'Оценка на симптоми',
        "home_vitals_btn": 'Жизнени показатели',
        "home_emergency": 'При болка в гърдите, затруднено дишане или загуба на съзнание, незабавно се обадете на 166 или 112.',
        "home_emergency_label": 'Спешно',
        "intake_for_whom": 'За кого е тази оценка?',
        "intake_for_me": 'За мен',
        "intake_for_other": 'За някой, за когото се грижа',
        "intake_tell_us": 'Разкажете ни за себе си',
        "intake_tell_us_sub": 'Ime, възраст, медицинска история',
        "nav_home": 'Начало',
        "nav_vitals": 'Показатели',
        "nav_symptoms": 'Симптоми',
        "nav_history": 'История',
        "hero_h1": "Опишете какво чувствате.<br><span style='color:#2D3FE7'>Получете клинична оценка.</span>",
        "hero_sub": 'Оценка, основана на доказателства с PubMed референции + второ мнение от GPT-4o. За вашия <strong>лекар</strong>.',
        "hero_f1t": 'Описание на симптомите',
        "hero_f1s": 'Говорете естествено — AI пита и организира',
        "hero_f2t": 'Жизнени показатели',
        "hero_f2s": 'HR, BP, SpO₂, температура',
        "hero_f3t": 'Изследвания и снимки',
        "hero_f3s": 'Качете кръвни изследвания, PDF или снимка',
        "hero_f4t": 'Клиничен доклад за лекаря',
        "hero_f4s": 'PubMed + второ мнение GPT-4o',
        "hero_p1t": 'Второ медицинско мнение',
        "hero_p1s": 'Claude + GPT-4o независимо',
        "hero_p1b": 'Уникално',
        "hero_p2t": 'Интеграция с gov.gr',
        "hero_p2s": 'Здравно досие, AMKA, е-рецепта',
        "hero_p2b": 'Гръцки NHS',
        "hero_p3t": '16 езика',
        "hero_p3s": 'Гръцки, Английски, Арабски, Хинди и др.',
        "hero_p3b": 'За всички',
        "hero_disc": 'Не замества лекаря. Спешни: <strong>166</strong> или <strong>112</strong>. Не съхраняваме медицински данни. 🔒 GDPR',
        "hero_how": 'Как работи',
        "hero_cta": '✦ Започни оценка и доклад',
        "hero_login_title": 'Започни — безплатно, без парола',
        "hero_for_whom": 'За кого е',
        "hero_aud1t": 'За цялото семейство',
        "hero_aud1d": 'За вас, децата и родителите ви. Отидете при лекаря подготвени.',
        "hero_aud1b": 'Възрастни · Деца · Възрастни хора',
        "hero_aud2t": 'За болногледачи',
        "hero_aud2d": 'Грижите се за някой друг? Работи в caregiver mode.',
        "hero_aud2b": 'Caregiver mode',
        "hero_aud3t": 'За лекари и клиники',
        "hero_aud3d": 'Пациентът идва с организирана история. По-малко рутинни обаждания.',
        "hero_aud3b": 'Спестяване на време',
        "hero_s1l": 'Точност на триажа (Semigran-45)',
        "hero_s2l": 'Unsafe undertriage',
        "hero_s3l": 'Claude + GPT-4o второ мнение',
        "hero_gov_title": '🔒 Дигитални здравни услуги (gov.gr)',
        "hero_gov_note": 'Отваря в нов раздел на gov.gr — не съхраняваме данни.',
    },
    "ro": {
        "title": "Asklepios", "subtitle": "Asistenta ta AI",
        "tagline": "Informații medicale de încredere · Mereu alături de tine",
        "start": "Începe evaluarea",
        "disclaimer_main": "⚠️ Asklepios oferă informații de sănătate exclusiv în scop informativ. Nu înlocuiește diagnosticul sau tratamentul medical. În urgențe sunați **166** (EKAB) sau **112**.",
        "emergency": "🚨 URGENȚĂ: SUNAȚI 166 SAU 112",
        "name": "Nume", "age": "Vârstă", "sex": "Sex",
        "male": "Masculin", "female": "Feminin", "other": "Altul",
        "history": "Istoric medical (afecțiuni, operații)", "allergies": "Alergii",
        "meds": "Medicamente curente / suplimente", "next": "Înainte →", "back": "← Înapoi",
        "vitals_title": "Semne vitale", "vitals_sub": "Introduceți măsurătorile.",
        "hr": "Ritm cardiac (bpm)", "bp_sys": "Tensiune arterială — Sistolică (mmHg)",
        "bp_dia": "Tensiune arterială — Diastolică (mmHg)", "br": "Frecvență respiratorie (/min)",
        "spo2": "SpO2 (%)", "temp": "Temperatură (°C)", "weight": "Greutate (kg)", "height": "Înălțime (cm)",
        "analyse_vitals": "Analizați semnele vitale", "triage_title": "Evaluarea simptomelor",
        "triage_sub": "Descrieți simptomele. Asklepios va pune întrebări.",
        "triage_placeholder": "De ex. Am dureri de cap de trei zile cu greață...",
        "generate_report": "Generați raport complet", "report_title": "Evaluare detaliată a sănătății",
        "second_opinion": "A doua opinie GPT-4o", "pubmed": "Referințe PubMed",
        "skip_vitals": "Omiteți (fără măsurători)",
        "stepper_profile": '1 Profil',
        "stepper_vitals": '2 Vitale',
        "stepper_symptoms": '3 Simptome',
        "stepper_report": '4 Raport',
        "please_enter_name": 'Vă rugăm să introduceți numele dvs.',
        "bp_risk_title": 'Estimarea riscului tensiunii arteriale',
        "articles_label": 'Articole',
        "read_more": 'Citește mai mult',
        "triage_explainer": "👇 Pasul 3 — Descrieți ce vă deranjează (de ex. 'durere la ochi 2 zile'). Asklepios va pune întrebări și va genera un raport.",
        "triage_quick_select": 'Selecție rapidă',
        "triage_send_selected": 'Trimite selectate',
        "triage_main_symptoms": 'Simptome principale: ',
        "vitals_core_title": 'Semne vitale de bază',
        "vitals_optional": 'Respirație, greutate, înălțime (opțional)',
        "vitals_skip_continue": 'Nu am măsurători — Continuă la simptome →',
        "voice_input_label": '🎤 Introducere vocală (vorbiți în loc să tastați)',
        "home_explainer_title": 'De unde încep?',
        "home_explainer_body": 'Apăsați <strong>Verificare Simptome</strong>. Asklepios va întreba despre profilul dvs. și apoi va evalua simptomele pas cu pas.',
        "home_symptoms_btn": 'Verificare Simptome',
        "home_vitals_btn": 'Semne Vitale',
        "home_emergency": 'Pentru dureri în piept, dificultăți de respirație sau pierderea cunoștinței, sunați imediat la 166 sau 112.',
        "home_emergency_label": 'Urgență',
        "intake_for_whom": 'Pentru cine este această evaluare?',
        "intake_for_me": 'Pentru mine',
        "intake_for_other": 'Pentru cineva de care am grijă',
        "intake_tell_us": 'Spuneți-ne despre dvs.',
        "intake_tell_us_sub": 'Nume, vârstă, istoric medical',
        "nav_home": 'Acasă',
        "nav_vitals": 'Vitale',
        "nav_symptoms": 'Simptome',
        "nav_history": 'Istoric',
        "hero_h1": "Descrieți ce simțiți.<br><span style='color:#2D3FE7'>Obțineți o evaluare clinică.</span>",
        "hero_sub": 'Evaluare bazată pe dovezi cu referințe PubMed + a doua opinie GPT-4o. Pentru <strong>medicul</strong> dumneavoastră.',
        "hero_f1t": 'Descrierea simptomelor',
        "hero_f1s": 'Vorbiți natural — AI întreabă și organizează',
        "hero_f2t": 'Semne vitale',
        "hero_f2s": 'HR, BP, SpO₂, temperatură',
        "hero_f3t": 'Analize și fotografii',
        "hero_f3s": 'Încărcați analize de sânge, PDF sau o fotografie',
        "hero_f4t": 'Raport clinic pentru medic',
        "hero_f4s": 'PubMed + a doua opinie GPT-4o',
        "hero_p1t": 'A doua opinie medicală',
        "hero_p1s": 'Claude + GPT-4o independent',
        "hero_p1b": 'Unic',
        "hero_p2t": 'Integrare gov.gr',
        "hero_p2s": 'Dosar medical, AMKA, e-Prescripție',
        "hero_p2b": 'NHS Grec',
        "hero_p3t": '16 limbi',
        "hero_p3s": 'Română, Greacă, Arabă, Hindi și altele',
        "hero_p3b": 'Pentru toți',
        "hero_disc": 'Nu înlocuiește medicul. Urgențe: <strong>166</strong> sau <strong>112</strong>. Nu stocăm date medicale. 🔒 GDPR',
        "hero_how": 'Cum funcționează',
        "hero_cta": '✦ Începe evaluarea și raportul',
        "hero_login_title": 'Începe — gratuit, fără parolă',
        "hero_for_whom": 'Pentru cine este',
        "hero_aud1t": 'Pentru toată familia',
        "hero_aud1d": 'Pentru dumneavoastră, copii și părinți. Mergeți la medic pregătiți.',
        "hero_aud1b": 'Adulți · Copii · Vârstnici',
        "hero_aud2t": 'Pentru îngrijitori',
        "hero_aud2d": 'Îngrijiți pe cineva? Funcționează în caregiver mode.',
        "hero_aud2b": 'Caregiver mode',
        "hero_aud3t": 'Pentru medici și clinici',
        "hero_aud3d": 'Pacientul vine cu istoricul organizat. Mai puține apeluri de rutină.',
        "hero_aud3b": 'Economie de timp',
        "hero_s1l": 'Acuratețe triage (Semigran-45)',
        "hero_s2l": 'Unsafe undertriage',
        "hero_s3l": 'Claude + GPT-4o a doua opinie',
        "hero_gov_title": '🔒 Servicii digitale de sănătate (gov.gr)',
        "hero_gov_note": 'Se deschide într-un tab nou pe gov.gr — nu stocăm date.',
    },
    "al": {
        "title": "Asklepios", "subtitle": "Infermierja juaj AI",
        "tagline": "Informacion shëndetësor i besueshëm · Gjithmonë pranë jush",
        "start": "Fillo vlerësimin",
        "disclaimer_main": "⚠️ Asklepios ofron informacion shëndetësor vetëm për qëllime informative. Nuk zëvendëson diagnozën ose trajtimin mjekësor. Në raste urgjence telefononi **166** (EKAB) ose **112**.",
        "emergency": "🚨 URGJENCË: TELEFONONI 166 OSE 112",
        "name": "Emri", "age": "Mosha", "sex": "Gjinia",
        "male": "Mashkull", "female": "Femër", "other": "Tjetër",
        "history": "Historia mjekësore (sëmundje, operacione)", "allergies": "Alergji",
        "meds": "Medikamente aktuale / suplemente", "next": "Tjetër →", "back": "← Kthehu",
        "vitals_title": "Shenjat vitale", "vitals_sub": "Fusni matjet tuaja.",
        "hr": "Ritmi i zemrës (bpm)", "bp_sys": "Presioni i gjakut — Sistolik (mmHg)",
        "bp_dia": "Presioni i gjakut — Diastolik (mmHg)", "br": "Shkalla e frymëmarrjes (/min)",
        "spo2": "SpO2 (%)", "temp": "Temperatura (°C)", "weight": "Pesha (kg)", "height": "Gjatësia (cm)",
        "analyse_vitals": "Analizoni shenjat vitale", "triage_title": "Vlerësimi i simptomave",
        "triage_sub": "Përshkruani simptomat tuaja. Asklepios do të bëjë pyetje.",
        "triage_placeholder": "P.sh. Kam dhimbje koke prej tre ditësh me të përziera...",
        "generate_report": "Gjenero raport të plotë", "report_title": "Vlerësim i detajuar shëndetësor",
        "second_opinion": "Mendim i dytë GPT-4o", "pubmed": "Referenca PubMed",
        "skip_vitals": "Kalo (pa matje)",
        "stepper_profile": '1 Profili',
        "stepper_vitals": '2 Vitale',
        "stepper_symptoms": '3 Simptoma',
        "stepper_report": '4 Raport',
        "please_enter_name": 'Ju lutemi vendosni emrin tuaj.',
        "bp_risk_title": 'Vlerësimi i rrezikut të presionit të gjakut',
        "articles_label": 'Artikuj',
        "read_more": 'Lexo më shumë',
        "triage_explainer": "👇 Hapi 3 — Përshkruani çfarë ju shqetëson (p.sh. 'dhimbje syri 2 ditë'). Asklepios do të bëjë pyetje dhe do të gjenerojë raport.",
        "triage_quick_select": 'Zgjidhje e shpejtë',
        "triage_send_selected": 'Dërgoni të zgjedhurat',
        "triage_main_symptoms": 'Simptomat kryesore: ',
        "vitals_core_title": 'Shenjat vitale bazë',
        "vitals_optional": 'Frymëmarrje, peshë, gjatësi (opsional)',
        "vitals_skip_continue": 'Nuk kam matje — Vazhdoni te simptomat →',
        "voice_input_label": '🎤 Hyrje me zë (flisni në vend të shtypjes)',
        "home_explainer_title": 'Nga ku filloj?',
        "home_explainer_body": 'Shtypni <strong>Vlerësim Simptomesh</strong>. Asklepios do të pyesë për profilin tuaj dhe pastaj do të vlerësojë simptomat hap pas hapi.',
        "home_symptoms_btn": 'Vlerësim Simptomesh',
        "home_vitals_btn": 'Shenjat Vitale',
        "home_emergency": 'Për dhimbje gjoksi, vështirësi frymëmarrjeje ose humbje ndërgjegjeje, telefononi menjëherë 166 ose 112.',
        "home_emergency_label": 'Urgjencë',
        "intake_for_whom": 'Ky vlerësim është për kë?',
        "intake_for_me": 'Për mua',
        "intake_for_other": 'Për dikë që kujdesem',
        "intake_tell_us": 'Tregoni për veten tuaj',
        "intake_tell_us_sub": 'Emri, mosha, historia mjekësore',
        "nav_home": 'Kryefaqja',
        "nav_vitals": 'Vitale',
        "nav_symptoms": 'Simptoma',
        "nav_history": 'Historia',
        "hero_h1": "Përshkruani çfarë ndiheni.<br><span style='color:#2D3FE7'>Merrni një vlerësim klinik.</span>",
        "hero_sub": 'Vlerësim i bazuar në dëshmi me referenca PubMed + mendim i dytë GPT-4o. Për <strong>mjekun</strong> tuaj.',
        "hero_f1t": 'Përshkrimi i simptomave',
        "hero_f1s": 'Flisni natyrshëm — AI pyet dhe organizon',
        "hero_f2t": 'Shenjat vitale',
        "hero_f2s": 'HR, BP, SpO₂, temperaturë',
        "hero_f3t": 'Analiza dhe fotografi',
        "hero_f3s": 'Ngarkoni analiza gjaku, PDF ose foto',
        "hero_f4t": 'Raport klinik për mjekun',
        "hero_f4s": 'PubMed + mendim i dytë GPT-4o',
        "hero_p1t": 'Mendim i dytë mjekësor',
        "hero_p1s": 'Claude + GPT-4o në mënyrë të pavarur',
        "hero_p1b": 'Unik',
        "hero_p2t": 'Integrim me gov.gr',
        "hero_p2s": 'Dosja shëndetësore, AMKA, e-Recetë',
        "hero_p2b": 'NHS Grek',
        "hero_p3t": '16 gjuhë',
        "hero_p3s": 'Shqip, Greqisht, Arabisht, Hindishte dhe të tjera',
        "hero_p3b": 'Për të gjithë',
        "hero_disc": 'Nuk zëvendëson mjekun. Urgjencë: <strong>166</strong> ose <strong>112</strong>. Nuk ruajmë të dhëna mjekësore. 🔒 GDPR',
        "hero_how": 'Si funksionon',
        "hero_cta": '✦ Fillo vlerësimin dhe raportin',
        "hero_login_title": 'Fillo — falas, pa fjalëkalim',
        "hero_for_whom": 'Për kë është',
        "hero_aud1t": 'Për të gjithë familjen',
        "hero_aud1d": 'Për ju, fëmijët dhe prindërit. Shkoni te mjeku të përgatitur.',
        "hero_aud1b": 'Të rritur · Fëmijë · Pleq',
        "hero_aud2t": 'Për kujdestarët',
        "hero_aud2d": 'Kujdeseni për dikë tjetër? Funksionon në caregiver mode.',
        "hero_aud2b": 'Caregiver mode',
        "hero_aud3t": 'Për mjekët dhe klinikat',
        "hero_aud3d": 'Pacienti vjen me historinë e organizuar. Thirrje rutinë më pak.',
        "hero_aud3b": 'Kursim kohe',
        "hero_s1l": 'Saktësi triage (Semigran-45)',
        "hero_s2l": 'Unsafe undertriage',
        "hero_s3l": 'Claude + GPT-4o mendim i dytë',
        "hero_gov_title": '🔒 Shërbime dixhitale shëndetësore (gov.gr)',
        "hero_gov_note": 'Hapet në skedë të re në gov.gr — nuk ruajmë të dhëna.',
    },
    "ru": {
        "title": "Asklepios", "subtitle": "Ваша AI медсестра",
        "tagline": "Достоверная медицинская информация · Всегда рядом",
        "start": "Начать оценку",
        "disclaimer_main": "⚠️ Asklepios предоставляет медицинскую информацию исключительно в информационных целях. Не заменяет медицинскую диагностику или лечение. В экстренных случаях звоните **166** (EKAB) или **112**.",
        "emergency": "🚨 ЭКСТРЕННЫЙ СЛУЧАЙ: ЗВОНИТЕ 166 ИЛИ 112",
        "name": "Имя", "age": "Возраст", "sex": "Пол",
        "male": "Мужской", "female": "Женский", "other": "Другое",
        "history": "История болезни (заболевания, операции)", "allergies": "Аллергии",
        "meds": "Текущие лекарства / добавки", "next": "Далее →", "back": "← Назад",
        "vitals_title": "Жизненные показатели", "vitals_sub": "Введите ваши измерения.",
        "hr": "Частота сердечных сокращений (bpm)", "bp_sys": "Артериальное давление — Систолическое (mmHg)",
        "bp_dia": "Артериальное давление — Диастолическое (mmHg)", "br": "Частота дыхания (/min)",
        "spo2": "SpO2 (%)", "temp": "Температура (°C)", "weight": "Вес (kg)", "height": "Рост (cm)",
        "analyse_vitals": "Анализ показателей", "triage_title": "Оценка симптомов",
        "triage_sub": "Опишите ваши симптомы. Asklepios задаст вопросы.",
        "triage_placeholder": "Напр. У меня болит голова три дня с тошнотой...",
        "generate_report": "Создать полный отчёт", "report_title": "Подробная оценка здоровья",
        "second_opinion": "Второе мнение GPT-4o", "pubmed": "Научные ссылки PubMed",
        "skip_vitals": "Пропустить (без измерений)",
        "stepper_profile": '1 Профиль',
        "stepper_vitals": '2 Показатели',
        "stepper_symptoms": '3 Симптомы',
        "stepper_report": '4 Отчёт',
        "please_enter_name": 'Пожалуйста, введите ваше имя.',
        "bp_risk_title": 'Оценка риска артериального давления',
        "articles_label": 'Статьи',
        "read_more": 'Читать далее',
        "triage_explainer": "👇 Шаг 3 — Опишите здесь что беспокоит (напр. 'боль в глазу 2 дня'). Asklepios задаст вопросы и составит отчёт.",
        "triage_quick_select": 'Быстрый выбор',
        "triage_send_selected": 'Отправить выбранные',
        "triage_main_symptoms": 'Основные симптомы: ',
        "vitals_core_title": 'Основные жизненные показатели',
        "vitals_optional": 'Дыхание, вес, рост (по желанию)',
        "vitals_skip_continue": 'Нет измерений — Продолжить к симптомам →',
        "voice_input_label": '🎤 Голосовой ввод (говорите вместо печати)',
        "home_explainer_title": 'С чего начать?',
        "home_explainer_body": 'Нажмите <strong>Оценка симптомов</strong>. Asklepios спросит о вашем профиле и затем пошагово оценит симптомы.',
        "home_symptoms_btn": 'Оценка симптомов',
        "home_vitals_btn": 'Жизненные показатели',
        "home_emergency": 'При боли в груди, затруднённом дыхании или потере сознания немедленно звоните 166 или 112.',
        "home_emergency_label": 'Экстренно',
        "intake_for_whom": 'Для кого эта оценка?',
        "intake_for_me": 'Для меня',
        "intake_for_other": 'Для того, о ком я забочусь',
        "intake_tell_us": 'Расскажите о себе',
        "intake_tell_us_sub": 'Имя, возраст, история болезни',
        "nav_home": 'Главная',
        "nav_vitals": 'Показатели',
        "nav_symptoms": 'Симптомы',
        "nav_history": 'История',
        "hero_h1": "Опишите, что вы чувствуете.<br><span style='color:#2D3FE7'>Получите клиническую оценку.</span>",
        "hero_sub": 'Оценка на основе доказательств с PubMed + второе мнение GPT-4o. Для вашего <strong>врача</strong>.',
        "hero_f1t": 'Описание симптомов',
        "hero_f1s": 'Говорите естественно — AI спрашивает и организует',
        "hero_f2t": 'Жизненные показатели',
        "hero_f2s": 'HR, BP, SpO₂, температура',
        "hero_f3t": 'Анализы и фото',
        "hero_f3s": 'Загрузите анализы крови, PDF или фото',
        "hero_f4t": 'Клинический отчёт для врача',
        "hero_f4s": 'PubMed + второе мнение GPT-4o',
        "hero_p1t": 'Второе медицинское мнение',
        "hero_p1s": 'Claude + GPT-4o независимо',
        "hero_p1b": 'Уникально',
        "hero_p2t": 'Интеграция с gov.gr',
        "hero_p2s": 'Медкарта, AMKA, е-рецепт',
        "hero_p2b": 'Греческий NHS',
        "hero_p3t": '16 языков',
        "hero_p3s": 'Русский, Греческий, Арабский, Хинди и др.',
        "hero_p3b": 'Для всех',
        "hero_disc": 'Не заменяет врача. Экстренно: <strong>166</strong> или <strong>112</strong>. Медданные не хранятся. 🔒 GDPR',
        "hero_how": 'Как это работает',
        "hero_cta": '✦ Начать оценку и отчёт',
        "hero_login_title": 'Начать — бесплатно, без пароля',
        "hero_for_whom": 'Для кого',
        "hero_aud1t": 'Для всей семьи',
        "hero_aud1d": 'Для вас, детей и родителей. Идите к врачу подготовленным.',
        "hero_aud1b": 'Взрослые · Дети · Пожилые',
        "hero_aud2t": 'Для сиделок',
        "hero_aud2d": 'Ухаживаете за кем-то? Работает в caregiver mode.',
        "hero_aud2b": 'Caregiver mode',
        "hero_aud3t": 'Для врачей и клиник',
        "hero_aud3d": 'Пациент приходит с организованной историей. Меньше рутинных звонков.',
        "hero_aud3b": 'Экономия времени',
        "hero_s1l": 'Точность триажа (Semigran-45)',
        "hero_s2l": 'Unsafe undertriage',
        "hero_s3l": 'Claude + GPT-4o второе мнение',
        "hero_gov_title": '🔒 Цифровые медицинские услуги (gov.gr)',
        "hero_gov_note": 'Открывается в новой вкладке на gov.gr — данные не хранятся.',
    },
    "zh": {
        "title": "Asklepios", "subtitle": "您的AI护士",
        "tagline": "可靠的健康信息 · 始终陪伴您",
        "start": "开始评估",
        "disclaimer_main": "⚠️ Asklepios仅出于信息目的提供健康信息。不能替代医疗诊断或治疗。紧急情况请拨打**166**（EKAB）或**112**。",
        "emergency": "🚨 紧急情况：请拨打 166 或 112",
        "name": "姓名", "age": "年龄", "sex": "性别",
        "male": "男", "female": "女", "other": "其他",
        "history": "病史（疾病、手术）", "allergies": "过敏史",
        "meds": "当前药物 / 补充剂", "next": "下一步 →", "back": "← 返回",
        "vitals_title": "生命体征", "vitals_sub": "请输入您的测量值。",
        "hr": "心率 (bpm)", "bp_sys": "血压 — 收缩压 (mmHg)",
        "bp_dia": "血压 — 舒张压 (mmHg)", "br": "呼吸频率 (/min)",
        "spo2": "SpO2 (%)", "temp": "体温 (°C)", "weight": "体重 (kg)", "height": "身高 (cm)",
        "analyse_vitals": "分析生命体征", "triage_title": "症状评估",
        "triage_sub": "描述您的症状。Asklepios将提出有针对性的问题。",
        "triage_placeholder": "例如：我头痛三天并伴有恶心...",
        "generate_report": "生成完整临床报告", "report_title": "详细健康评估",
        "second_opinion": "GPT-4o第二意见", "pubmed": "PubMed参考文献",
        "skip_vitals": "跳过（无测量值）",
        "stepper_profile": '1 档案',
        "stepper_vitals": '2 体征',
        "stepper_symptoms": '3 症状',
        "stepper_report": '4 报告',
        "please_enter_name": '请输入您的姓名。',
        "bp_risk_title": '血压风险评估',
        "articles_label": '文章',
        "read_more": '阅读更多',
        "triage_explainer": "👇 第3步 — 在此描述困扰您的问题（例如'眼睛痛2天'）。Asklepios将提问并生成报告。",
        "triage_quick_select": '快速选择',
        "triage_send_selected": '发送所选',
        "triage_main_symptoms": '主要症状：',
        "vitals_core_title": '核心生命体征',
        "vitals_optional": '呼吸、体重、身高（可选）',
        "vitals_skip_continue": '我不需要生命体征 — 继续症状 →',
        "voice_input_label": '🎤 语音输入（说话代替打字）',
        "home_explainer_title": '我从哪里开始？',
        "home_explainer_body": '点击<strong>症状检查</strong>开始。Asklepios将询问您的基本情况，然后逐步评估您的症状。',
        "home_symptoms_btn": '症状检查',
        "home_vitals_btn": '生命体征',
        "home_emergency": '如有胸痛、呼吸困难或失去意识，请立即拨打 166 或 112。',
        "home_emergency_label": '紧急情况',
        "intake_for_whom": '这次评估是为谁？',
        "intake_for_me": '为我自己',
        "intake_for_other": '为我照顾的人',
        "intake_tell_us": '告诉我们关于您的情况',
        "intake_tell_us_sub": '姓名、年龄、病史',
        "nav_home": '首页',
        "nav_vitals": '体征',
        "nav_symptoms": '症状',
        "nav_history": '历史',
        "hero_h1": "描述您的感受。<br><span style='color:#2D3FE7'>获取临床评估。</span>",
        "hero_sub": '基于证据的评估，含PubMed参考文献 + GPT-4o第二意见。为您的<strong>医生</strong>。',
        "hero_f1t": '症状描述',
        "hero_f1s": '自然说话 — AI询问并整理',
        "hero_f2t": '生命体征',
        "hero_f2s": 'HR, BP, SpO₂, 体温',
        "hero_f3t": '检查和照片',
        "hero_f3s": '上传血液检查、PDF或照片',
        "hero_f4t": '给医生的报告',
        "hero_f4s": 'PubMed + GPT-4o第二意见',
        "hero_p1t": '第二医疗意见',
        "hero_p1s": 'Claude + GPT-4o独立评估',
        "hero_p1b": '独特',
        "hero_p2t": 'gov.gr整合',
        "hero_p2s": '健康档案、AMKA、电子处方',
        "hero_p2b": '希腊NHS',
        "hero_p3t": '16种语言',
        "hero_p3s": '中文、阿拉伯语、印地语等',
        "hero_p3b": '面向所有人',
        "hero_disc": '不替代医生。紧急情况：<strong>166</strong>或<strong>112</strong>。🔒 GDPR',
        "hero_how": '工作原理',
        "hero_cta": '✦ 开始评估',
        "hero_login_title": '开始使用 — 免费，无需密码',
        "hero_for_whom": '适合谁',
        "hero_aud1t": '全家适用',
        "hero_aud1d": '为您、您的孩子和父母。带着整理好的病史去看医生。',
        "hero_aud1b": '成人 · 儿童 · 老人',
        "hero_aud2t": '为照护者',
        "hero_aud2d": '照顾他人？在Caregiver模式下工作。',
        "hero_aud2b": 'Caregiver模式',
        "hero_aud3t": '为医生和诊所',
        "hero_aud3d": '患者带着有序的病史来诊。减少常规电话。',
        "hero_aud3b": '节省时间',
        "hero_s1l": '分诊准确率 (Semigran-45)',
        "hero_s2l": 'Unsafe undertriage',
        "hero_s3l": 'Claude + GPT-4o第二意见',
        "hero_gov_title": '🔒 数字健康服务 (gov.gr)',
        "hero_gov_note": '在gov.gr上新标签页打开。',
    },
    "lb": {
        "title": "Asklepios", "subtitle": "ممرضتك بالذكاء الاصطناعي",
        "tagline": "معلومات صحية موثوقة · دايمًا معك",
        "start": "ابدأ التقييم",
        "disclaimer_main": "⚠️ Asklepios بيقدم معلومات صحية لأغراض إعلامية بس. ما بيحل محل التشخيص أو العلاج الطبي. بحالات الطوارئ اتصل بـ **166** (EKAB) أو **112**.",
        "emergency": "🚨 طوارئ: اتصل بـ 166 أو 112",
        "name": "الاسم", "age": "العمر", "sex": "الجنس",
        "male": "ذكر", "female": "أنثى", "other": "غير ذلك",
        "history": "التاريخ الطبي (أمراض، عمليات)", "allergies": "الحساسية",
        "meds": "الدوا الحالي / مكملات", "next": "التالي →", "back": "← رجوع",
        "vitals_title": "العلامات الحيوية", "vitals_sub": "دخّل قياساتك.",
        "hr": "نبضات القلب (bpm)", "bp_sys": "ضغط الدم — الانقباضي (mmHg)",
        "bp_dia": "ضغط الدم — الانبساطي (mmHg)", "br": "معدل التنفس (/min)",
        "spo2": "SpO2 (%)", "temp": "الحرارة (°C)", "weight": "الوزن (kg)", "height": "الطول (cm)",
        "analyse_vitals": "حلّل العلامات الحيوية", "triage_title": "تقييم الأعراض",
        "triage_sub": "وصف أعراضك. رح يسألك Asklepios أسئلة.",
        "triage_placeholder": "مثلاً: عندي صداع من تلات أيام مع غثيان...",
        "generate_report": "اعمل تقرير طبي كامل", "report_title": "تقييم صحي مفصّل",
        "second_opinion": "رأي ثاني من GPT-4o", "pubmed": "مراجع PubMed العلمية",
        "skip_vitals": "تخطَّ (بدون قياسات)",
        "stepper_profile": '1 الملف',
        "stepper_vitals": '2 الحيوية',
        "stepper_symptoms": '3 الأعراض',
        "stepper_report": '4 التقرير',
        "please_enter_name": 'رجاء أدخل اسمك.',
        "bp_risk_title": 'تقدير خطر ضغط الدم',
        "articles_label": 'مقالات',
        "read_more": 'اقرأ أكثر',
        "triage_explainer": "👇 خطوة 3 — وصف هون شو عم يضايقك (مثلاً 'ألم بالعين يومين'). Asklepios رح يسأل ويعمل تقرير.",
        "triage_quick_select": 'اختيار سريع',
        "triage_send_selected": 'ارسل المختار',
        "triage_main_symptoms": 'الأعراض الرئيسية: ',
        "vitals_core_title": 'العلامات الحيوية الأساسية',
        "vitals_optional": 'التنفس، الوزن، الطول (اختياري)',
        "vitals_skip_continue": 'ما محتاج علامات حيوية — كمّل للأعراض →',
        "voice_input_label": '🎤 إدخال صوتي',
        "home_explainer_title": 'من وين بدأ؟',
        "home_explainer_body": 'اضغط على <strong>فحص الأعراض</strong>. Asklepios رح يسألك عن معلوماتك وبعدين يقيّم أعراضك.',
        "home_symptoms_btn": 'فحص الأعراض',
        "home_vitals_btn": 'العلامات الحيوية',
        "home_emergency": 'لألم الصدر أو صعوبة التنفس أو فقدان الوعي، اتصل فوراً بـ 166 أو 112.',
        "home_emergency_label": 'طوارئ',
        "intake_for_whom": 'التقييم هاد لمين؟',
        "intake_for_me": 'إلي',
        "intake_for_other": 'لحدا عم رعاه',
        "intake_tell_us": 'قولنا عنك',
        "intake_tell_us_sub": 'الاسم، العمر، التاريخ الطبي',
        "nav_home": 'الرئيسية',
        "nav_vitals": 'الحيوية',
        "nav_symptoms": 'الأعراض',
        "nav_history": 'التاريخ',
        "hero_h1": "قول شو حاسس فيه.<br><span style='color:#2D3FE7'>احصل على تقييم طبي.</span>",
        "hero_sub": 'تقييم مبني على الأدلة + رأي ثانٍ من GPT-4o. لـ<strong>دكتورك</strong>.',
        "hero_f1t": 'وصف الأعراض',
        "hero_f1s": 'حكي طبيعي — AI بيسأل وبينظّم',
        "hero_f2t": 'العلامات الحيوية',
        "hero_f2s": 'HR, BP, SpO₂, حرارة',
        "hero_f3t": 'فحوصات وصور',
        "hero_f3s": 'ارفع تحاليل أو صورة',
        "hero_f4t": 'تقرير للدكتور',
        "hero_f4s": 'PubMed + رأي ثانٍ GPT-4o',
        "hero_p1t": 'رأي طبي ثانٍ',
        "hero_p1s": 'Claude + GPT-4o مستقلَّين',
        "hero_p1b": 'فريد',
        "hero_p2t": 'gov.gr',
        "hero_p2s": 'ملف الصحة، AMKA، وصفات',
        "hero_p2b": 'NHS اليوناني',
        "hero_p3t": '16 لغة',
        "hero_p3s": 'عربي، إنجليزي، هندي وغيرو',
        "hero_p3b": 'للكل',
        "hero_disc": 'ما بيحل محل الدكتور. طوارئ: <strong>166</strong> أو <strong>112</strong>. 🔒 GDPR',
        "hero_how": 'كيف بيشتغل',
        "hero_cta": '✦ ابدأ التقييم',
        "hero_login_title": 'ابدأ — مجاناً، بدون باسوورد',
        "hero_for_whom": 'لمين هو',
        "hero_aud1t": 'للعيلة كلها',
        "hero_aud1d": 'إلك ولأولادك ولأهلك. روح عند الدكتور مجهّز.',
        "hero_aud1b": 'كبار · صغار · مسنين',
        "hero_aud2t": 'لمن بيهتم بغيره',
        "hero_aud2d": 'عم تهتم بحدا تاني؟ بيشتغل بـ Caregiver mode.',
        "hero_aud2b": 'Caregiver mode',
        "hero_aud3t": 'للأطباء والعيادات',
        "hero_aud3d": 'المريض بييجي مع تاريخ منظّم. مكالمات أقل.',
        "hero_aud3b": 'توفير الوقت',
        "hero_s1l": 'دقة الفرز (Semigran-45)',
        "hero_s2l": 'Unsafe undertriage',
        "hero_s3l": 'Claude + GPT-4o رأي ثانٍ',
        "hero_gov_title": '🔒 خدمات صحية رقمية (gov.gr)',
        "hero_gov_note": 'بتفتح بتاب جديد على gov.gr.',
    },
    "he": {
        "title": "Asklepios", "subtitle": "האחות ה-AI שלך",
        "tagline": "מידע רפואי אמין · תמיד לצדך",
        "start": "התחל הערכה",
        "disclaimer_main": "⚠️ Asklepios מספק מידע בריאותי למטרות מידע בלבד. אינו מחליף אבחנה רפואית או טיפול. במקרה חירום התקשר **166** (EKAB) או **112**.",
        "emergency": "🚨 חירום: התקשר ל-166 או 112",
        "name": "שם", "age": "גיל", "sex": "מין",
        "male": "זכר", "female": "נקבה", "other": "אחר",
        "history": "היסטוריה רפואית (מחלות, ניתוחים)", "allergies": "אלרגיות",
        "meds": "תרופות נוכחיות / תוספים", "next": "הבא →", "back": "← חזור",
        "vitals_title": "סימנים חיוניים", "vitals_sub": "הזן את המדידות שלך.",
        "hr": "קצב לב (bpm)", "bp_sys": "לחץ דם — סיסטולי (mmHg)",
        "bp_dia": "לחץ דם — דיאסטולי (mmHg)", "br": "קצב נשימה (/min)",
        "spo2": "SpO2 (%)", "temp": "טמפרטורה (°C)", "weight": "משקל (kg)", "height": "גובה (cm)",
        "analyse_vitals": "נתח סימנים חיוניים", "triage_title": "הערכת תסמינים",
        "triage_sub": "תאר את הסימפטומים שלך. Asklepios ישאל שאלות.",
        "triage_placeholder": "למשל: כאב ראש שלושה ימים עם בחילה...",
        "generate_report": "צור דוח קליני מלא", "report_title": "הערכת בריאות מפורטת",
        "second_opinion": "חוות דעת שנייה GPT-4o", "pubmed": "מקורות PubMed",
        "skip_vitals": "דלג (ללא מדידות)",
        "stepper_profile": '1 פרופיל',
        "stepper_vitals": '2 חיוניים',
        "stepper_symptoms": '3 תסמינים',
        "stepper_report": '4 דוח',
        "please_enter_name": 'אנא הזן את שמך.',
        "bp_risk_title": 'הערכת סיכון לחץ דם',
        "articles_label": 'מאמרים',
        "read_more": 'קרא עוד',
        "triage_explainer": "👇 שלב 3 — תאר כאן מה מטריד אותך (למשל 'כאב עין יומיים'). Asklepios ישאל שאלות ואז ייצור דוח.",
        "triage_quick_select": 'בחירה מהירה',
        "triage_send_selected": 'שלח נבחרים',
        "triage_main_symptoms": 'תסמינים עיקריים: ',
        "vitals_core_title": 'סימנים חיוניים בסיסיים',
        "vitals_optional": 'נשימה, משקל, גובה (אופציונלי)',
        "vitals_skip_continue": 'אין לי סימנים חיוניים — המשך לתסמינים →',
        "voice_input_label": '🎤 קלט קולי (דבר במקום להקליד)',
        "home_explainer_title": 'מאיפה מתחילים?',
        "home_explainer_body": 'לחץ על <strong>בדיקת תסמינים</strong>. Asklepios ישאל על הפרופיל שלך ואז יעריך את התסמינים שלך שלב אחר שלב.',
        "home_symptoms_btn": 'בדיקת תסמינים',
        "home_vitals_btn": 'סימנים חיוניים',
        "home_emergency": 'לכאב חזה, קשיי נשימה או אובדן הכרה, התקשר מיד לـ 166 או 112.',
        "home_emergency_label": 'חירום',
        "intake_for_whom": 'עבור מי ההערכה הזו?',
        "intake_for_me": 'בשבילי',
        "intake_for_other": 'עבור מישהו שאני מטפל בו',
        "intake_tell_us": 'ספר לנו על עצמך',
        "intake_tell_us_sub": 'שם, גיל, היסטוריה רפואית',
        "nav_home": 'בית',
        "nav_vitals": 'חיוניים',
        "nav_symptoms": 'תסמינים',
        "nav_history": 'היסטוריה',
        "hero_h1": "תאר מה אתה מרגיש.<br><span style='color:#2D3FE7'>קבל הערכה קלינית.</span>",
        "hero_sub": 'הערכה מבוססת ראיות + חוות דעת שנייה מ-GPT-4o. ל<strong>רופא</strong> שלך.',
        "hero_f1t": 'תיאור תסמינים',
        "hero_f1s": 'דבר בטבעיות — AI שואל ומארגן',
        "hero_f2t": 'סימנים חיוניים',
        "hero_f2s": 'HR, BP, SpO₂, טמפרטורה',
        "hero_f3t": 'בדיקות ותמונות',
        "hero_f3s": 'העלה בדיקות דם, PDF או תמונה',
        "hero_f4t": 'דוח לרופא',
        "hero_f4s": 'PubMed + חוות דעת שנייה GPT-4o',
        "hero_p1t": 'חוות דעת רפואית שנייה',
        "hero_p1s": 'Claude + GPT-4o באופן עצמאי',
        "hero_p1b": 'ייחודי',
        "hero_p2t": 'שילוב עם gov.gr',
        "hero_p2s": 'תיק בריאות, AMKA, מרשם אלקטרוני',
        "hero_p2b": 'NHS יווני',
        "hero_p3t": '16 שפות',
        "hero_p3s": 'עברית, ערבית, הינדי ועוד',
        "hero_p3b": 'לכולם',
        "hero_disc": 'אינו מחליף רופא. חירום: <strong>166</strong> או <strong>112</strong>. 🔒 GDPR',
        "hero_how": 'איך זה עובד',
        "hero_cta": '✦ התחל הערכה',
        "hero_login_title": 'התחל — חינם, ללא סיסמה',
        "hero_for_whom": 'למי זה מיועד',
        "hero_aud1t": 'לכל המשפחה',
        "hero_aud1d": 'לך, לילדיך ולהוריך. לך לרופא מוכן.',
        "hero_aud1b": 'מבוגרים · ילדים · קשישים',
        "hero_aud2t": 'למטפלים',
        "hero_aud2d": 'מטפל במישהו אחר? עובד ב-Caregiver mode.',
        "hero_aud2b": 'Caregiver mode',
        "hero_aud3t": 'לרופאים ומרפאות',
        "hero_aud3d": 'המטופל מגיע עם היסטוריה מסודרת. פחות שיחות שגרתיות.',
        "hero_aud3b": 'חיסכון בזמן',
        "hero_s1l": 'דיוק Triage (Semigran-45)',
        "hero_s2l": 'Unsafe undertriage',
        "hero_s3l": 'Claude + GPT-4o חוות דעת שנייה',
        "hero_gov_title": '🔒 שירותי בריאות דיגיטליים (gov.gr)',
        "hero_gov_note": 'נפתח בכרטיסייה חדשה ב-gov.gr.',
    },
    "pa": {
        "title": "Asklepios", "subtitle": "ਤੁਹਾਡੀ AI ਨਰਸ",
        "tagline": "ਭਰੋਸੇਯੋਗ ਸਿਹਤ ਜਾਣਕਾਰੀ · ਹਮੇਸ਼ਾ ਤੁਹਾਡੇ ਨਾਲ",
        "start": "ਮੁਲਾਂਕਣ ਸ਼ੁਰੂ ਕਰੋ",
        "disclaimer_main": "⚠️ Asklepios ਕੇਵਲ ਜਾਣਕਾਰੀ ਲਈ ਸਿਹਤ ਜਾਣਕਾਰੀ ਪ੍ਰਦਾਨ ਕਰਦਾ ਹੈ। ਐਮਰਜੈਂਸੀ ਵਿੱਚ **166** ਜਾਂ **112** 'ਤੇ ਕਾਲ ਕਰੋ।",
        "emergency": "🚨 ਐਮਰਜੈਂਸੀ: 166 ਜਾਂ 112 'ਤੇ ਕਾਲ ਕਰੋ",
        "name": "ਨਾਮ", "age": "ਉਮਰ", "sex": "ਲਿੰਗ",
        "male": "ਪੁਰਸ਼", "female": "ਔਰਤ", "other": "ਹੋਰ",
        "history": "ਡਾਕਟਰੀ ਇਤਿਹਾਸ", "allergies": "ਐਲਰਜੀ",
        "meds": "ਮੌਜੂਦਾ ਦਵਾਈਆਂ", "next": "ਅਗਲਾ →", "back": "← ਵਾਪਸ",
        "vitals_title": "ਮਹੱਤਵਪੂਰਨ ਸੰਕੇਤ", "vitals_sub": "ਆਪਣੀ ਮਾਪ ਦਰਜ ਕਰੋ।",
        "hr": "ਦਿਲ ਦੀ ਦਰ (bpm)", "bp_sys": "ਬਲੱਡ ਪ੍ਰੈਸ਼ਰ — ਸਿਸਟੋਲਿਕ (mmHg)",
        "bp_dia": "ਬਲੱਡ ਪ੍ਰੈਸ਼ਰ — ਡਾਇਸਟੋਲਿਕ (mmHg)", "br": "ਸਾਹ ਦੀ ਦਰ (/min)",
        "spo2": "SpO2 (%)", "temp": "ਤਾਪਮਾਨ (°C)", "weight": "ਵਜ਼ਨ (kg)", "height": "ਉਚਾਈ (cm)",
        "analyse_vitals": "ਸੰਕੇਤਾਂ ਦਾ ਵਿਸ਼ਲੇਸ਼ਣ", "triage_title": "ਲੱਛਣਾਂ ਦਾ ਮੁਲਾਂਕਣ",
        "triage_sub": "ਆਪਣੇ ਲੱਛਣ ਦੱਸੋ। Asklepios ਸਵਾਲ ਪੁੱਛੇਗਾ।",
        "triage_placeholder": "ਜਿਵੇਂ: ਤਿੰਨ ਦਿਨਾਂ ਤੋਂ ਸਿਰਦਰਦ...",
        "generate_report": "ਪੂਰੀ ਰਿਪੋਰਟ ਬਣਾਓ", "report_title": "ਵਿਸਤ੍ਰਿਤ ਸਿਹਤ ਮੁਲਾਂਕਣ",
        "second_opinion": "GPT-4o ਦੂਜੀ ਰਾਏ", "pubmed": "PubMed ਹਵਾਲੇ",
        "skip_vitals": "ਛੱਡੋ (ਮਾਪ ਤੋਂ ਬਿਨਾਂ)",
        "stepper_profile": '1 ਪ੍ਰੋਫਾਈਲ',
        "stepper_vitals": '2 ਸੰਕੇਤ',
        "stepper_symptoms": '3 ਲੱਛਣ',
        "stepper_report": '4 ਰਿਪੋਰਟ',
        "please_enter_name": 'ਕਿਰਪਾ ਕਰਕੇ ਆਪਣਾ ਨਾਮ ਦਰਜ ਕਰੋ।',
        "bp_risk_title": 'ਬਲੱਡ ਪ੍ਰੈਸ਼ਰ ਜੋਖਮ ਅਨੁਮਾਨ',
        "articles_label": 'ਲੇਖ',
        "read_more": 'ਹੋਰ ਪੜ੍ਹੋ',
        "triage_explainer": "👇 ਕਦਮ 3 — ਇੱਥੇ ਦੱਸੋ ਕੀ ਪਰੇਸ਼ਾਨ ਕਰ ਰਿਹਾ ਹੈ (ਜਿਵੇਂ 'ਅੱਖ ਵਿੱਚ ਦਰਦ 2 ਦਿਨ')। Asklepios ਸਵਾਲ ਪੁੱਛੇਗਾ ਅਤੇ ਰਿਪੋਰਟ ਬਣਾਏਗਾ।",
        "triage_quick_select": 'ਤੇਜ਼ ਚੋਣ',
        "triage_send_selected": 'ਚੁਣੇ ਭੇਜੋ',
        "triage_main_symptoms": 'ਮੁੱਖ ਲੱਛਣ: ',
        "vitals_core_title": 'ਮੁੱਖ ਸੰਕੇਤ',
        "vitals_optional": 'ਸਾਹ, ਭਾਰ, ਉਚਾਈ (ਵਿਕਲਪਿਕ)',
        "vitals_skip_continue": 'ਮੈਨੂੰ ਸੰਕੇਤਾਂ ਦੀ ਲੋੜ ਨਹੀਂ — ਲੱਛਣਾਂ ਤੇ ਜਾਰੀ ਰੱਖੋ →',
        "voice_input_label": '🎤 ਆਵਾਜ਼ ਇਨਪੁੱਟ (ਟਾਈਪ ਕਰਨ ਦੀ ਬਜਾਏ ਬੋਲੋ)',
        "home_explainer_title": 'ਮੈਂ ਕਿੱਥੋਂ ਸ਼ੁਰੂ ਕਰਾਂ?',
        "home_explainer_body": "<strong>ਲੱਛਣ ਜਾਂਚ</strong> 'ਤੇ ਟੈਪ ਕਰੋ। Asklepios ਤੁਹਾਡਾ ਪ੍ਰੋਫਾਈਲ ਪੁੱਛੇਗਾ ਅਤੇ ਫਿਰ ਲੱਛਣਾਂ ਦਾ ਮੁਲਾਂਕਣ ਕਰੇਗਾ।",
        "home_symptoms_btn": 'ਲੱਛਣ ਜਾਂਚ',
        "home_vitals_btn": 'ਮਹੱਤਵਪੂਰਨ ਸੰਕੇਤ',
        "home_emergency": "ਸੀਨੇ ਵਿੱਚ ਦਰਦ, ਸਾਹ ਲੈਣ ਵਿੱਚ ਮੁਸ਼ਕਲ ਜਾਂ ਬੇਹੋਸ਼ੀ ਲਈ ਤੁਰੰਤ 166 ਜਾਂ 112 'ਤੇ ਕਾਲ ਕਰੋ।",
        "home_emergency_label": 'ਐਮਰਜੈਂਸੀ',
        "intake_for_whom": 'ਇਹ ਮੁਲਾਂਕਣ ਕਿਸ ਲਈ ਹੈ?',
        "intake_for_me": 'ਮੇਰੇ ਲਈ',
        "intake_for_other": 'ਜਿਸਦੀ ਮੈਂ ਦੇਖਭਾਲ ਕਰਦਾ ਹਾਂ ਉਸ ਲਈ',
        "intake_tell_us": 'ਸਾਨੂੰ ਆਪਣੇ ਬਾਰੇ ਦੱਸੋ',
        "intake_tell_us_sub": 'ਨਾਮ, ਉਮਰ, ਡਾਕਟਰੀ ਇਤਿਹਾਸ',
        "nav_home": 'ਹੋਮ',
        "nav_vitals": 'ਸੰਕੇਤ',
        "nav_symptoms": 'ਲੱਛਣ',
        "nav_history": 'ਇਤਿਹਾਸ',
        "hero_h1": "ਦੱਸੋ ਤੁਸੀਂ ਕੀ ਮਹਿਸੂਸ ਕਰ ਰਹੇ ਹੋ।<br><span style='color:#2D3FE7'>ਕਲੀਨੀਕਲ ਮੁਲਾਂਕਣ ਪ੍ਰਾਪਤ ਕਰੋ।</span>",
        "hero_sub": "PubMed + GPT-4o ਦੂਜੀ ਰਾਏ ਨਾਲ ਮੁਲਾਂਕਣ। ਤੁਹਾਡੇ <strong>ਡਾਕਟਰ</strong> ਲਈ।",
        "hero_f1t": "ਲੱਛਣਾਂ ਦਾ ਵਰਣਨ", "hero_f1s": "ਕੁਦਰਤੀ ਤੌਰ 'ਤੇ ਬੋਲੋ — AI ਪੁੱਛਦਾ ਹੈ",
        "hero_f2t": "ਮਹੱਤਵਪੂਰਨ ਸੰਕੇਤ", "hero_f2s": "HR, BP, SpO₂, ਤਾਪਮਾਨ",
        "hero_f3t": "ਟੈਸਟ ਅਤੇ ਫੋਟੋਆਂ", "hero_f3s": "ਖੂਨ ਦੇ ਟੈਸਟ, PDF ਜਾਂ ਫੋਟੋ ਅੱਪਲੋਡ ਕਰੋ",
        "hero_f4t": "ਡਾਕਟਰ ਲਈ ਰਿਪੋਰਟ", "hero_f4s": "PubMed + GPT-4o ਦੂਜੀ ਰਾਏ",
        "hero_p1t": "ਦੂਜੀ ਡਾਕਟਰੀ ਰਾਏ", "hero_p1s": "Claude + GPT-4o ਸੁਤੰਤਰ",
        "hero_p1b": "ਵਿਲੱਖਣ", "hero_p2t": "gov.gr ਨਾਲ ਜੁੜਾਅ",
        "hero_p2s": "ਸਿਹਤ ਰਿਕਾਰਡ, AMKA, e-ਨੁਸਖਾ", "hero_p2b": "Greek NHS",
        "hero_p3t": "16 ਭਾਸ਼ਾਵਾਂ", "hero_p3s": "ਪੰਜਾਬੀ, ਹਿੰਦੀ, ਅਰਬੀ ਅਤੇ ਹੋਰ", "hero_p3b": "ਸਭ ਲਈ",
        "hero_disc": "ਡਾਕਟਰ ਦਾ ਬਦਲ ਨਹੀਂ। ਐਮਰਜੈਂਸੀ: <strong>166</strong> ਜਾਂ <strong>112</strong>। 🔒 GDPR",
        "hero_how": "ਇਹ ਕਿਵੇਂ ਕੰਮ ਕਰਦਾ ਹੈ", "hero_cta": "✦ ਮੁਲਾਂਕਣ ਸ਼ੁਰੂ ਕਰੋ",
        "hero_login_title": "ਸ਼ੁਰੂ ਕਰੋ — ਮੁਫ਼ਤ, ਬਿਨਾਂ ਪਾਸਵਰਡ",
        "hero_for_whom": "ਕਿਸ ਲਈ ਹੈ",
        "hero_aud1t": "ਪੂਰੇ ਪਰਿਵਾਰ ਲਈ", "hero_aud1d": "ਤੁਹਾਡੇ, ਬੱਚਿਆਂ ਅਤੇ ਮਾਪਿਆਂ ਲਈ।", "hero_aud1b": "ਬਾਲਗ · ਬੱਚੇ · ਬਜ਼ੁਰਗ",
        "hero_aud2t": "ਦੇਖਭਾਲ ਕਰਨ ਵਾਲਿਆਂ ਲਈ", "hero_aud2d": "Caregiver mode ਵਿੱਚ ਕੰਮ ਕਰਦਾ ਹੈ।", "hero_aud2b": "Caregiver mode",
        "hero_aud3t": "ਡਾਕਟਰਾਂ ਲਈ", "hero_aud3d": "ਮਰੀਜ਼ ਸੰਗਠਿਤ ਇਤਿਹਾਸ ਨਾਲ ਆਉਂਦਾ ਹੈ।", "hero_aud3b": "ਸਮੇਂ ਦੀ ਬੱਚਤ",
        "hero_s1l": "Triage ਸਟੀਕਤਾ (Semigran-45)", "hero_s2l": "Unsafe undertriage", "hero_s3l": "Claude + GPT-4o ਦੂਜੀ ਰਾਏ",
        "hero_gov_title": "🔒 ਡਿਜੀਟਲ ਸਿਹਤ ਸੇਵਾਵਾਂ (gov.gr)", "hero_gov_note": "gov.gr 'ਤੇ ਨਵੀਂ ਟੈਬ ਵਿੱਚ ਖੁੱਲ੍ਹਦਾ ਹੈ।",
    },
}
# RTL languages — these need dir="rtl" in HTML blocks
RTL_LANGS = {"ar", "ur", "lb", "he"}

# For languages without a full UI translation, fall back to English
def t(key):
    lang = st.session_state.get("lang", "el")
    return T.get(lang, T["en"]).get(key, T["en"].get(key, key))

def is_rtl():
    return st.session_state.get("lang", "el") in RTL_LANGS
# UI_LANGUAGES: the languages available for the app interface itself.
# These drive the st.session_state.lang toggle in the topbar.
# Keep to languages where the full UI strings are translated (el/en).
# For report/AI output in other languages, OUTPUT_LANGUAGES handles that.
UI_LANGUAGES = {
    "el": "🇬🇷 ΕΛ",
    "en": "🇬🇧 EN",
}

def render_topbar():
    """Top-right controls visible on every post-login screen.
    - Left: spacer (nav tabs are in render_bottom_nav fixed top)
    - Right: language selector (all OUTPUT_LANGUAGES) + logout
    The language selector here controls BOTH the UI lang (el/en) AND the
    AI output/report language — one unified picker instead of two separate ones.
    """
    lang = st.session_state.lang
    _t1, _t2, _t3 = st.columns([5, 2, 1])
    with _t2:
        # Unified language picker: el/en → UI lang; others → output_lang only
        _all_lang_codes = list(OUTPUT_LANGUAGES.keys())
        _cur = st.session_state.get("output_lang") or lang
        try:    _idx = _all_lang_codes.index(_cur)
        except: _idx = 0
        _chosen = st.selectbox(
            "",
            _all_lang_codes,
            index=_idx,
            format_func=lambda c: OUTPUT_LANGUAGES[c][0],
            key="topbar_lang_select",
            label_visibility="collapsed",
        )
        if _chosen != _cur:
            # Every language changes both the UI lang AND the output lang.
            # t() falls back to English for languages without full UI translation,
            # but the UI lang key drives the Claude system prompt language.
            st.session_state.lang = _chosen
            st.session_state["output_lang"] = _chosen
            st.rerun()
    with _t3:
        if is_logged_in():
            if st.button("🚪", key="topbar_logout", use_container_width=True,
                         help=("Έξοδος" if lang=="el" else "Logout")):
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
    """Top navigation bar — replaces the old fixed-bottom tab bar.
    Sits at the top of every post-login screen via position:fixed top.
    Contains: logo | tabs (Αρχική/Ζωτικά/Συμπτώματα/Ιστορικό) | lang+logout.
    """
    lang = st.session_state.lang
    has_profile = bool(st.session_state.profile.get("name"))
    cur = st.session_state.screen

    tab_for_screen = {
        "home": "home", "intake": "triage", "vitals": "vitals",
        "triage": "triage", "report": "history",
    }
    active_tab = tab_for_screen.get(cur, "home")

    items = [
        ("home",    "🏠", t("nav_home")),
        ("vitals",  "❤️", t("nav_vitals")),
        ("triage",  "💬", t("nav_symptoms")),
        ("history", "📋", t("nav_history")),
    ]

    st.markdown("""
<style>
/* ── TOP NAV spacer — keeps content from hiding under the fixed bar ── */
.bottom-nav-spacer { height: 60px; }

/* Fix the stHorizontalBlock that contains .bn-marker to the TOP */
div[data-testid="stHorizontalBlock"]:has(.bn-marker) {
  position: fixed; left: 0; right: 0; top: 0; bottom: auto; z-index: 999;
  background: white; border-bottom: 1px solid #EEF2FA; border-top: none;
  padding: 0 8px;
  box-shadow: 0 2px 12px rgba(15,42,82,0.07);
  height: 52px;
  display: flex !important; align-items: center !important;
  flex-wrap: nowrap !important;
  justify-content: space-between !important;
  gap: 2px !important;
}
div[data-testid="stHorizontalBlock"]:has(.bn-marker) > div[data-testid="stColumn"] {
  min-width: 0 !important; width: 25% !important; flex: 0 0 25% !important;
  display: flex !important; align-items: center !important;
}
.bn-marker { display: none; }
div[data-testid="stHorizontalBlock"]:has(.bn-marker) button {
  background: transparent !important; border: none !important; box-shadow: none !important;
  color: #B8C2D6 !important; font-weight: 700 !important; font-size: 9.5px !important;
  line-height: 1.3 !important; padding: 2px 2px !important; min-height: 0 !important;
  white-space: nowrap !important; width: 100% !important; height: 48px !important;
  display: flex !important; align-items: center !important; justify-content: center !important;
  text-align: center !important;
}
div[data-testid="stHorizontalBlock"]:has(.bn-marker) button p {
  text-align: center !important; width: 100%;
}
div[data-testid="stHorizontalBlock"]:has(.bn-marker) button[kind="primary"] {
  color: #2D6FE0 !important; border-bottom: 2px solid #2D6FE0 !important;
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
    steps_el = [t("stepper_profile"), t("stepper_vitals"), t("stepper_symptoms"), t("stepper_report")]
    steps_en = [t("stepper_profile"), t("stepper_vitals"), t("stepper_symptoms"), t("stepper_report")]
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
    "hi": ("🇮🇳 हिन्दी",    "Hindi (हिन्दी)"),
    "ur": ("🇵🇰 اردو",      "Urdu (اردو)"),
    "ar": ("🇸🇦 العربية",   "Arabic (العربية)"),
    "bn": ("🇧🇩 বাংলা",     "Bengali (বাংলা)"),
    "bg": ("🇧🇬 Български", "Bulgarian (Български)"),
    "ro": ("🇷🇴 Română",    "Romanian (Română)"),
    "al": ("🇦🇱 Shqip",     "Albanian (Shqip)"),
    "ru": ("🇷🇺 Русский",   "Russian (Русский)"),
    "de": ("🇩🇪 Deutsch",   "German (Deutsch)"),
    "fr": ("🇫🇷 Français",  "French (Français)"),
    "pa": ("🇵🇰 ਪੰਜਾਬੀ",   "Punjabi (ਪੰਜਾਬੀ)"),
    "zh": ("🇨🇳 中文",      "Chinese (中文)"),
    "lb": ("🇱🇧 عربي لبناني", "Lebanese Arabic (عربي لبناني)"),
    "he": ("🇮🇱 עברית",    "Hebrew (עברית)"),
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
            if l.startswith("#"):
                out.append(f"<h3>{_html.escape(l.lstrip('#').strip())}</h3>" if l.startswith("###")
                            else f"<h2>{_html.escape(l.lstrip('#').strip())}</h2>")
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
        def _md_bold(t): return _re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', _html.escape(t or "—"))
        _ex = _md_bold(recs.get("exercise"))
        _nu = _md_bold(recs.get("nutrition"))
        _li = _md_bold(recs.get("lifestyle"))
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
h3{{font-size:12.5px;font-weight:700;color:#2D3FE7;margin:14px 0 6px}}
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
        _exp_title = t("home_explainer_title")
        _exp_body  = t("home_explainer_body")
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
            _lbl1 = t("home_symptoms_btn")
            if st.button(_lbl1, key="home_go_triage", use_container_width=True):
                _go("triage")
    with ac2:
        with st.container(border=True):
            st.markdown(
                '<div class="home-action-marker"></div>'
                '<div style="text-align:center"><div class="home-action-icon warm">❤️</div></div>',
                unsafe_allow_html=True,
            )
            _lbl2 = t("home_vitals_btn")
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
    _em_text = t("home_emergency")
    st.markdown(
        f'<div class="home-emergency"><strong>🚨 {t("home_emergency_label")}:</strong> {_em_text}</div>',
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
        st.markdown("##### 📰 " + t("articles_label"))
        for art in _articles[:10]:
            with st.container(border=True):
                st.markdown(f"**{art.get('title','—')}**")
                _meta = " · ".join(x for x in [art.get("author",""), art.get("source",""),
                                                str(art.get("published_at",""))] if x)
                if _meta:
                    st.caption(_meta)
                st.write(art.get("summary",""))
                if art.get("body"):
                    with st.expander(t("read_more")):
                        st.write(art["body"])
                if art.get("url"):
                    st.markdown(f"[{'Πλήρες άρθρο →' if lang=='el' else 'Full article →'}]({art['url']})")


def render_intake():
    render_stepper("intake")
    lang = st.session_state.lang
    render_doc_header(
        t("intake_tell_us"), t("intake_tell_us"),
        icon="👤",
        sub_el=t("intake_tell_us_sub"),
        sub_en=t("intake_tell_us_sub"),
    )
    # ── Caregiver toggle ───────────────────────────────────────────────────
    # First question: is this assessment for the user themselves or someone
    # they care for (γιαγιά, παιδί, κλπ). Affects copy + Claude system prompt.
    _caregiver_q = t("intake_for_whom")
    _opt_self  = t("intake_for_me")
    _opt_other = t("intake_for_other")
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
                st.warning(t("please_enter_name"))

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
                f'<div class="home-group-title">{t("vitals_core_title")}</div>',
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

        with st.expander("＋ " + t("vitals_optional")):
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
            title   = t("bp_risk_title")
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
        if st.button(t("vitals_skip_continue"), use_container_width=True):
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
        st.info(t("triage_explainer"))
        chips, _chips_label = _symptom_chips(st.session_state.profile, st.session_state.lang)
        _cap = t("triage_quick_select")
        if _chips_label:
            _cap += f" ({_chips_label})"
        st.caption(_cap + ":")
        # Chips in rows of 4 — aligned and wrapped
        _PER_ROW = 4
        for _rs in range(0, len(chips), _PER_ROW):
            _row = chips[_rs:_rs+_PER_ROW]
            _cc = st.columns(_PER_ROW)
            for _j, chip in enumerate(_row):
                _i = _rs + _j
                with _cc[_j]:
                    sel = chip in st.session_state.symptom_chips
                    if st.button(("✓ " if sel else "")+chip, key=f"chip_{_i}", use_container_width=True):
                        if chip in st.session_state.symptom_chips: st.session_state.symptom_chips.remove(chip)
                        else: st.session_state.symptom_chips.append(chip)
                        st.rerun()
        if st.session_state.symptom_chips:
            if st.button("➤ " + t("triage_send_selected"), type="primary"):
                msg = t("triage_main_symptoms") + ", ".join(st.session_state.symptom_chips)
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
    _voice_lbl = t("voice_input_label")
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

    # ── Back / Generate-report buttons ───────────────────────────────────────
    # Rendered BEFORE the "write here" banner + chat_input on purpose: st.chat_input
    # is fixed/sticky to the bottom of the viewport regardless of where it sits in
    # the code, so anything coded *after* it still visually appears ABOVE it —
    # which used to sandwich these buttons between the banner and the actual
    # input box (confusing on mobile, see user report). Moving them here keeps
    # the code order == visual order: chat → these buttons → banner → input.
    col_b,col_r=st.columns([1,2])
    with col_b:
        if st.button(t("back")): st.session_state.screen="vitals"; st.rerun()
    with col_r:
        enabled=triage_ready or len(st.session_state.triage_chat)>=6
        if st.button(t("generate_report"),type="primary",use_container_width=True,disabled=not enabled):
            st.session_state.screen="report"; st.rerun()

    # ── "Write here" indicator — shown when no conversation yet ──────────────
    if not st.session_state.triage_chat:
        _write_lbl = {
            "el": "✍️ Γράψε εδώ τα συμπτώματά σου ↓",
            "en": "✍️ Write your symptoms here ↓",
            "hi": "✍️ यहाँ अपने लक्षण लिखें ↓",
            "ur": "✍️ یہاں اپنی علامات لکھیں ↓",
            "ar": "✍️ اكتب أعراضك هنا ↓",
            "he": "✍️ כתוב כאן את הסימפטומים שלך ↓",
            "zh": "✍️ 在此写下您的症状 ↓",
            "lb": "✍️ اكتب أعراضك هنا ↓",
            "bg": "✍️ Напишете тук симптомите си ↓",
            "ro": "✍️ Scrieți simptomele dvs. aici ↓",
            "al": "✍️ Shkruani këtu simptomat tuaja ↓",
            "ru": "✍️ Напишите здесь свои симптомы ↓",
            "bn": "✍️ এখানে আপনার লক্ষণ লিখুন ↓",
            "pa": "✍️ ਇੱਥੇ ਆਪਣੇ ਲੱਛਣ ਲਿਖੋ ↓",
        }.get(st.session_state.lang, "✍️ Write your symptoms here ↓")
        st.markdown(f"""
<div style="text-align:center;font-size:13px;font-weight:700;color:#2D3FE7;
  background:#EEF2FF;border:1px solid #C7D2FE;border-radius:10px;
  padding:8px 16px;margin:8px 0 4px;font-family:'Inter',system-ui,sans-serif;
  {"direction:rtl;" if is_rtl() else ""}">
  {_write_lbl}
</div>
""", unsafe_allow_html=True)

    # ── Mobile chat-input viewport fix ───────────────────────────────────────
    # st.chat_input renders fixed to the bottom of the browser viewport. Inside
    # mobile webviews (in-app browsers, PWA wrappers) this can misbehave: when
    # the on-screen keyboard opens, some mobile browsers don't update the CSS
    # viewport height, so the fixed input either sits underneath the keyboard
    # or gets pushed out past the visible area, forcing the user to hunt/scroll
    # for it. Using the dynamic-viewport unit (100dvh) instead of the static
    # one keeps the input pinned to the *visually visible* bottom edge, and the
    # safe-area inset avoids it being clipped on notched devices.
    st.markdown("""
<style>
[data-testid="stChatInput"] {
  position: sticky;
  bottom: 0;
  z-index: 999;
  padding-bottom: env(safe-area-inset-bottom, 0px);
  background: var(--background-color, #F4F6FF);
}
@supports (height: 100dvh) {
  [data-testid="stAppViewContainer"] {
    min-height: 100dvh;
  }
}
</style>
""", unsafe_allow_html=True)

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
    def _bold(t): return _re_rec.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', t)
    ex = _bold(_html_r.escape(_flat(recs.get("exercise"))))
    nu = _bold(_html_r.escape(_flat(recs.get("nutrition"))))
    li = _bold(_html_r.escape(_flat(recs.get("lifestyle"))))

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
    import html as _html_mod
    def _maps(q):
        return f"https://www.google.com/maps/search/?api=1&query={_up.quote(q)}"
    # Official Ministry of Health page for Athens hospital duty schedule.
    # vrisko.gr was a 3rd-party aggregator; moh.gov.gr is the authoritative source.
    URL_DOC   = "https://www.vrisko.gr/dir/giatroi-eopyy"
    URL_HOSP  = "https://www.moh.gov.gr/articles/citizen/efhmeries-nosokomeiwn/68-efhmeries-nosokomeiwn-attikhs"
    URL_PHARM = "https://www.vrisko.gr/efimeries-farmakeion"

    # ── Featured / partner provider (B2B2C) ──────────────────────────────────
    # Pulled from the existing `partners` table (already managed via the admin
    # panel — see _admin_partners_tab). Shown ABOVE the general Google Maps /
    # EOPYY links as a supplementary option, never as a replacement: Greece-
    # wide coverage and live geolocation from Google Maps stay available no
    # matter what, so a user outside a partner's city/specialty still gets a
    # genuinely useful "where to go" answer. Any partner shown here carries an
    # explicit "Συνεργαζόμενος Πάροχος" badge — this is a paid/affiliated
    # placement, not a clinical recommendation, and must never be presented
    # as if it were one.
    _partner_html = ""
    try:
        _partners_active = [r for r in _admin_list("partners") if r.get("active", True)]
    except Exception:
        _partners_active = []
    if _partners_active:
        _badge_lbl = "⭐ Συνεργαζόμενος Πάροχος" if lang == "el" else "⭐ Partner Provider"
        _disclosure = ("Συνεργαζόμενος φορέας — εμφανίζεται επιπλέον των γενικών επιλογών παρακάτω, όχι στη θέση τους."
                       if lang == "el" else
                       "Affiliated provider — shown in addition to, not instead of, the general options below.")
        _rows = ""
        for _pr in _partners_active:
            _pname = _html_mod.escape(str(_pr.get("name","")))
            _pspec = _html_mod.escape(str(_pr.get("specialty","") or ""))
            _pcity = _html_mod.escape(str(_pr.get("city","") or ""))
            _pphone = _html_mod.escape(str(_pr.get("phone","") or ""))
            _pweb  = _pr.get("website","") or ""
            if _pweb and not _pweb.startswith(("http://", "https://")):
                _pweb = "https://" + _pweb
            _pmeta = " · ".join(x for x in [_pspec, _pcity] if x)
            _pmaps = _maps(f"{_pr.get('name','')} {_pr.get('city','')}".strip())
            _rows += f"""
    <div class="er-partner-row">
      <div class="er-partner-info">
        <div class="er-partner-name">{_pname}</div>
        {f'<div class="er-partner-meta">{_pmeta}</div>' if _pmeta else ''}
      </div>
      <div class="er-partner-actions">
        {f'<a class="er-partner-btn" href="tel:{_pphone}">📞</a>' if _pphone else ''}
        <a class="er-partner-btn" href="{_pmaps}" target="_blank" rel="noopener">🗺️</a>
        {f'<a class="er-partner-btn" href="{_html_mod.escape(_pweb)}" target="_blank" rel="noopener">🌐</a>' if _pweb else ''}
      </div>
    </div>"""
        _partner_html = f"""
  <div class="er-partner-section">
    <div class="er-partner-title">{_badge_lbl}</div>
    {_rows}
    <div class="er-partner-disclosure">ℹ️ {_disclosure}</div>
  </div>"""

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

.er-partner-section {{
  background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
  border: 1px solid #FCD34D; border-radius: 12px;
  padding: 14px 16px; margin-bottom: 16px;
}}
.er-partner-title {{
  font-size: 10.5px; font-weight: 800; color: #92400E;
  letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 10px;
}}
.er-partner-row {{
  display: flex; justify-content: space-between; align-items: center;
  gap: 12px; padding: 8px 0; border-top: 1px dashed rgba(146,64,14,0.18);
}}
.er-partner-row:first-of-type {{ border-top: none; padding-top: 0; }}
.er-partner-info {{ flex: 1; min-width: 0; }}
.er-partner-name {{ font-size: 13px; font-weight: 700; color: #1A1A2E; }}
.er-partner-meta {{ font-size: 11px; color: #92400E; margin-top: 1px; }}
.er-partner-actions {{ display: flex; gap: 6px; flex-shrink: 0; }}
.er-partner-btn {{
  display: inline-flex; align-items: center; justify-content: center;
  width: 30px; height: 30px; border-radius: 8px;
  background: white; border: 1px solid #FCD34D; text-decoration: none;
  font-size: 13px;
}}
.er-partner-btn:hover {{ background: #FEF3C7; text-decoration: none; }}
.er-partner-disclosure {{
  font-size: 10.5px; color: #92400E; margin-top: 8px; padding-top: 8px;
  border-top: 1px dashed rgba(146,64,14,0.18); line-height: 1.4;
}}

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
  .er-partner-row {{ flex-wrap: wrap; }}
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
{_partner_html}
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
@keyframes ask-msg{{0%{{opacity:0;transform:translateY(6px)}}2%{{opacity:1;transform:translateY(0)}}10%{{opacity:1;transform:translateY(0)}}12%{{opacity:0;transform:translateY(-4px)}}100%{{opacity:0}}}}
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
  animation:ask-msg 80s infinite ease-in-out}}
.ask-msg:nth-child(1){{animation-delay:0s}}
.ask-msg:nth-child(2){{animation-delay:10s}}
.ask-msg:nth-child(3){{animation-delay:20s}}
.ask-msg:nth-child(4){{animation-delay:30s}}
.ask-msg:nth-child(5){{animation-delay:40s}}
.ask-msg:nth-child(6){{animation-delay:50s}}
.ask-msg:nth-child(7){{animation-delay:60s}}
.ask-msg:nth-child(8){{animation-delay:70s}}
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
      <span class="ask-msg">{"📖 Ανάγνωση επιστημονικών άρθρων…" if lang=="el" else "📖 Reading research abstracts…"}</span>
      <span class="ask-msg">{"🩺 Σύνταξη κλινικής εκτίμησης…" if lang=="el" else "🩺 Writing clinical assessment…"}</span>
      <span class="ask-msg">{"💊 Έλεγχος φαρμάκων & αντενδείξεων…" if lang=="el" else "💊 Checking medications…"}</span>
      <span class="ask-msg">{"📚 Αναζήτηση οδηγιών ανά πυλώνα…" if lang=="el" else "📚 Searching guideline-level evidence…"}</span>
      <span class="ask-msg">{"🔁 Βελτίωση θεραπευτικού πλάνου…" if lang=="el" else "🔁 Refining the treatment plan…"}</span>
      <span class="ask-msg">{"📍 Εξατομικευμένες συστάσεις…" if lang=="el" else "📍 Personalised recommendations…"}</span>
      <span class="ask-msg">{"✨ Σχεδόν έτοιμο!" if lang=="el" else "✨ Almost ready!"}</span>
    </div>
  </div>
  <div class="ask-dots">
    <div class="ask-dot"></div><div class="ask-dot"></div><div class="ask-dot"></div>
  </div>
  <div class="ask-foot">{"Μην κλείσεις τη σελίδα — η αναφορά δημιουργείται (60–100″)." if lang=="el" else "Don't close the page — report is being generated (60–100s)."}</div>
</div>
</div>
""", unsafe_allow_html=True)

        with st.spinner("🔬 PubMed..." if lang=="el" else "🔬 Searching PubMed..."):
            refs=pubmed_search(search_query,n=3); st.session_state.report_pubmed=refs
        # Pull abstract text for these refs (best-effort — a PMID with no
        # abstract or a failed fetch just falls back to title-only, same as
        # before). This grounds the report in what the papers actually say
        # instead of letting the model fill in citations from training
        # knowledge with only a title as a label.
        _abstracts = pubmed_fetch_abstracts([a["pmid"] for a in refs]) if refs else {}
        def _ref_block(a):
            head = f"- {a['title']} ({a['journal']}, {a['date']}) {a['url']}"
            abs_txt = _abstracts.get(a["pmid"])
            return f"{head}\n  ABSTRACT: {abs_txt}" if abs_txt else head
        pubmed_ctx="\n".join(_ref_block(a) for a in refs) if refs else "None found."
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
(If ABSTRACT lines are present above, ground the Treatment Plan in what those abstracts actually say, not only the paper titles. Do NOT write a references/bibliography list yourself — a verified, clickable PubMed reference list is generated separately from real PMIDs and appended after your text. A free-text reference list you write would not be independently verifiable and would duplicate that section, so it must be omitted entirely.)
Write these sections IN THIS ORDER, using EXACTLY these headers as written (do not abbreviate or drop letters):
{"1. ΚΥΡΙΟ ΠΑΡΑΠΟΝΟ  2. ΙΣΤΟΡΙΚΟ  3. ΕΚΤΙΜΗΣΗ (Πρωτεύουσα Διάγνωση + Διαφορικές Διαγνώσεις)  4. ΘΕΡΑΠΕΥΤΙΚΟ ΠΛΑΝΟ  5. ΚΟΚΚΙΝΕΣ ΣΗΜΑΙΕΣ" if lang=="el" else "1. CHIEF COMPLAINT  2. HISTORY  3. ASSESSMENT (Primary Diagnosis + Differentials)  4. TREATMENT PLAN  5. RED FLAGS"}
For the differentials use a markdown table with EXACTLY 3 columns and these short headers: {"| Διάγνωση | % | Σχόλιο |" if lang=="el" else "| Diagnosis | % | Comment |"} (keep the probability header as just "%", and put values like "~8%"). Keep cell text short.

After section 5 (Red Flags), append EXACTLY this delimited block — same format, no extra text inside the delimiters:
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
                        # of literature. This refreshes report_pubmed (the single,
                        # PMID-verified reference list rendered separately in the
                        # "🔬 PubMed" expander + PDF export — Claude no longer writes
                        # its own free-text bibliography, see report_prompt above).
                        _futs["bibliography"] = _ex.submit(bibliography_search, _condition, 3)
                        _all_refs = {p: f.result() for p,f in _futs.items()}
                if _all_refs.get("bibliography"):
                    st.session_state.report_pubmed = _all_refs["bibliography"]
                st.session_state.report_recs_refs = {p: _all_refs[p] for p in ("exercise","nutrition","lifestyle")}
                st.session_state.report_physio_refs = _all_refs.get("physio", [])
                st.session_state.report_psych_refs  = _all_refs.get("psychology", [])
                # ── Re-ground Treatment Plan on the FINAL, condition-scoped refs ─
                # The first Claude pass (above) only saw 3 papers from a rough,
                # un-scoped search query (often the raw last chat message). This
                # refetch just got better, condition-scoped papers — re-write the
                # Treatment Plan section so it's grounded in what THESE abstracts
                # say (the same papers the user sees in the expander/PDF), rather
                # than the earlier, looser set. Cheaper and lower-risk than
                # regenerating the whole report, and the prompt explicitly forbids
                # touching the diagnosis/differentials/red flags already shown.
                _final_refs = _all_refs.get("bibliography") or []
                if _final_refs:
                    _final_abstracts = pubmed_fetch_abstracts([a["pmid"] for a in _final_refs])
                    def _final_ref_block(a):
                        head = f"- {a['title']} ({a['journal']}, {a['date']}) {a['url']}"
                        abs_txt = _final_abstracts.get(a["pmid"])
                        return f"{head}\n  ABSTRACT: {abs_txt}" if abs_txt else head
                    _final_pubmed_ctx = "\n".join(_final_ref_block(a) for a in _final_refs)
                    _plan_hdr = "4. ΘΕΡΑΠΕΥΤΙΚΟ ΠΛΑΝΟ" if lang=="el" else "4. TREATMENT PLAN"
                    _regroup_prompt = f"""Here is a clinical assessment already written for this patient:
---
{st.session_state.report}
---
You now have more specific PubMed literature for the diagnosis "{_condition}" than what was available when the assessment above was written:
{_final_pubmed_ctx}

Rewrite ONLY the "{_plan_hdr}" section, grounding it in what these specific abstracts actually say (do not invent citations — just use the findings to make the plan more specific/evidence-aligned). Do NOT change the diagnosis, differentials, or red flags, and do NOT write a references/bibliography list. Keep the header EXACTLY as written above. Output ONLY that one section (header + content), nothing else — no preamble, no other sections. Language: {"Greek" if lang=="el" else "English"}.{output_language_directive()}"""
                    with st.spinner("📚 " + ("Ενημέρωση θεραπευτικού πλάνου..." if lang=="el"
                                              else "Refining treatment plan...")):
                        _regroup = claude([{"role":"user","content":_regroup_prompt}],system=kira_system(),max_tokens=900,timeout=60)
                    if _regroup and not _regroup.startswith("⚠️") and _regroup.strip():
                        import re as _re_regroup
                        # Best-effort swap: replace the existing Treatment Plan section
                        # (from its header up to the next "## "-style header, or end of
                        # text) with the rewritten one. If the header can't be located,
                        # the original report is left untouched — a regex miss must
                        # never corrupt or blank out the report.
                        pat = rf"(#{{1,3}}\s*{_re_regroup.escape(_plan_hdr)}.*?)(?=\n#{{1,3}}\s|\Z)"
                        m = _re_regroup.search(pat, st.session_state.report, _re_regroup.DOTALL)
                        if m:
                            st.session_state.report = (
                                st.session_state.report[:m.start()]
                                + _regroup.strip() + "\n\n"
                                + st.session_state.report[m.end():]
                            )
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
# The hero is shown to EVERY visitor on their first page load of the session.
# It serves as both the marketing landing AND the login form entry point.
# Once the user clicks the CTA (logged in or not), _hero_seen = True and
# they proceed. The LOGIN GATE below then enforces auth if Supabase is active.
if not st.session_state.get("_hero_seen"):
    render_login_screen()
    st.stop()

# ── LOGIN GATE ────────────────────────────────────────────────────────────────
# If Supabase auth is configured, require a verified email session.
# If auth is not configured (local dev / missing secrets), let through.
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
# RTL global override — applied once per render for Arabic/Hebrew/Urdu/Lebanese
if is_rtl():
    st.markdown("""
<style>
.stApp, .stMarkdown, .stTextInput, .stSelectbox, .stTextArea,
div[data-testid="stForm"], div[data-testid="column"] {
  direction: rtl !important; text-align: right !important;
}
input, textarea, select { direction: rtl !important; text-align: right !important; }
</style>
""", unsafe_allow_html=True)
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

# Top nav bar — spacer pushes content below the fixed bar at the top.
st.markdown('<div class="bottom-nav-spacer"></div>', unsafe_allow_html=True)
render_bottom_nav()
