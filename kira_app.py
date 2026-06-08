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
/* Markdown tables — clean column alignment on mobile (override the per-letter break above) */
[data-testid="stMarkdownContainer"] table {
    width: 100%; border-collapse: collapse; table-layout: fixed;
    font-size: 12.5px; margin: 12px 0;
}
[data-testid="stMarkdownContainer"] thead th { background: #F4F6FF; font-weight: 600; }
[data-testid="stMarkdownContainer"] th,
[data-testid="stMarkdownContainer"] td {
    border: 1px solid #E0E5FF; padding: 7px 9px;
    text-align: left; vertical-align: top;
    word-break: normal !important; overflow-wrap: break-word !important; hyphens: none;
}
/* Narrow middle column (probability/%) so Διάγνωση & Σχόλιο get the room */
[data-testid="stMarkdownContainer"] th:nth-child(2),
[data-testid="stMarkdownContainer"] td:nth-child(2) { width: 64px; text-align: center; }
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
def get_ncbi_key():    return _key("NCBI_API_KEY")

# ── AUTH (Supabase email-OTP — gates the premium report) ──────────────────────
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
    # Clear all session state and reset to defaults
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    try:
        if "pe" in st.query_params: del st.query_params["pe"]
    except Exception:
        pass

def render_login_gate():
    """Inline email->OTP login. Returns True once the user is logged in."""
    lang = st.session_state.lang
    if is_logged_in():
        return True
    st.markdown(f'''<div style="background:rgba(45,63,231,0.06);border:1px solid rgba(45,63,231,0.15);border-radius:14px;padding:20px 22px;text-align:center;margin:10px 0">
        <div style="font-size:34px;margin-bottom:6px">🔒</div>
        <div style="font-size:16px;font-weight:700;color:#1A1A2E">{"Σύνδεση" if lang=="el" else "Sign in"}</div>
        <div style="font-size:13px;color:#6B7280;margin-top:4px">{"Βάλε το email σου και τον 6ψήφιο κωδικό που θα λάβεις για να συνεχίσεις." if lang=="el" else "Enter your email and the 6-digit code we send you to continue."}</div>
    </div>''', unsafe_allow_html=True)
    sent_to = st.session_state.get("otp_sent_to")
    if not sent_to:
        # Survive a mobile reload / new session: recover the pending email from the URL
        pe = st.query_params.get("pe")
        if pe:
            st.session_state["otp_sent_to"] = pe
            sent_to = pe
    if not sent_to:
        email = st.text_input("Email", key="otp_email", placeholder="you@example.com")
        if st.button(("Στείλε κωδικό" if lang=="el" else "Send code"), type="primary", use_container_width=True, key="otp_send"):
            if email and "@" in email:
                ok, err = send_otp(email)
                if ok:
                    st.session_state["otp_sent_to"] = email
                    st.query_params["pe"] = email
                    st.rerun()
                else:
                    st.error(("Σφάλμα αποστολής: " if lang=="el" else "Send error: ") + err)
            else:
                st.warning("Έγκυρο email, παρακαλώ." if lang=="el" else "Please enter a valid email.")
    else:
        st.caption((f"Στείλαμε 6ψήφιο κωδικό στο {sent_to}" if lang=="el" else f"We sent a 6-digit code to {sent_to}"))
        code = st.text_input(("Κωδικός" if lang=="el" else "Code"), key="otp_code", placeholder="123456")
        c1, c2 = st.columns([2,1])
        with c1:
            if st.button(("Επιβεβαίωση" if lang=="el" else "Verify"), type="primary", use_container_width=True, key="otp_verify"):
                ok, err = verify_otp(sent_to, code)
                if ok:
                    st.session_state.pop("otp_sent_to", None)
                    if "pe" in st.query_params: del st.query_params["pe"]
                    st.rerun()
                else:
                    st.error(("Δεν έγινε σύνδεση: " if lang=="el" else "Sign-in failed: ") + (err or ("λάθος/ληγμένος κωδικός" if lang=="el" else "wrong/expired code")))
        with c2:
            if st.button(("Άλλο email" if lang=="el" else "Change email"), use_container_width=True, key="otp_reset"):
                st.session_state.pop("otp_sent_to", None)
                if "pe" in st.query_params: del st.query_params["pe"]
                st.rerun()
    return is_logged_in()

def render_login_screen():
    """Full-page login shown at the very start when auth is enabled."""
    lang = st.session_state.lang
    c1, c2 = st.columns([6,1])
    with c2:
        if st.button("🇬🇧 EN" if lang=="el" else "🇬🇷 ΕΛ", key="login_lang"):
            st.session_state.lang = "en" if lang=="el" else "el"; st.rerun()
    st.markdown(f'''<div class="kira-hero"><div style="font-size:64px;margin-bottom:8px">🩺</div><h1>{t("title")}</h1><p>{t("subtitle")}</p><div class="kira-tagline">{t("tagline")}</div></div>''', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        render_login_gate()
    st.markdown(f'<div class="disclaimer">{t("disclaimer_main")}</div>', unsafe_allow_html=True)

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
    "fb_rating": "",
    "fb_sent": False,
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
Φωτογραφία: Αν το σύμπτωμα είναι οπτικό (δέρμα/εξάνθημα, μάτι, τραύμα/πληγή/χτύπημα/εξόγκωμα/μώλωπας, στόμα/λαιμός, νύχια, ορατή αλλοίωση), αφού κάνεις την αρχική σου εκτίμηση πρότεινε στον χρήστη να ανεβάσει φωτογραφία από την επιλογή «📷 Ανάλυση φωτογραφίας» πιο κάτω, για πιο ακριβή εκτίμηση. Για μη-οπτικά συμπτώματα (π.χ. πονοκέφαλος, ζάλη) ΜΗΝ ζητάς φωτογραφία. Η φωτογραφία είναι ΠΡΟΑΙΡΕΤΙΚΗ: αν ο χρήστης δεν ανεβάσει ή δεν θέλει, ΣΥΝΕΧΙΣΕ κανονικά την εκτίμηση χωρίς να σταματάς, να περιμένεις ή να επιμένεις.
Κανόνες: Πάντα συστήνεις επαγγελματία. Κόκκινες σημαίες → 166/112. Όταν έχεις αρκετά: "Έχω αρκετά στοιχεία — μπορούμε να δημιουργήσουμε πλήρη αναφορά." Μία ερώτηση κάθε φορά.
Ζωτικά: Αν τα συμπτώματα είναι καρδιακά/αυτόνομα (αίσθημα παλμών, ταχυπαλμία, πόνος/σφίξιμο στο στήθος, δύσπνοια, ζάλη, λιποθυμία, κρύος ιδρώτας/εφίδρωση) ή αναπνευστικά (βήχας, άσθμα, δύσπνοια), πρότεινε ήπια στον χρήστη να μετρήσει ζωτικά μέσω σάρωσης κάμερας ή χειροκίνητα — ΠΡΟΑΙΡΕΤΙΚΟ, συνέχισε κανονικά αν δεν το κάνει."""

KIRA_SYSTEM_EN = """You are Asklepios — an AI nurse for users in Greece. Clinically accurate, direct, supportive.
Role: Symptom triage (one question at a time), vitals interpretation, medications, Greek health system (EOPYY, EODY, EOF).
Photo: If the symptom is visual (skin/rash, eye, wound/hit/bruise/swelling/lump/hematoma, mouth/throat, nails, any visible lesion), after giving your initial assessment, invite the user to upload a photo via the "📷 Photo analysis" option below for a more accurate assessment. For non-visual symptoms (e.g. headache, dizziness) do NOT ask for a photo. The photo is OPTIONAL: if the user doesn't upload one or declines, CONTINUE the assessment normally — do not stop, wait, or insist.
Rules: Always recommend a professional. Red flags → 166/112. When ready: "I have enough information — we can generate a full clinical report." One question at a time.
Vitals: If the symptoms are cardiac/autonomic (palpitations, racing heart, chest pain/tightness, shortness of breath, dizziness, fainting, cold sweat/sweating) or respiratory (cough, asthma, dyspnoea), gently suggest the user measure vitals via camera scan or manual entry — OPTIONAL, continue normally if they don't."""

def kira_system(): return KIRA_SYSTEM_EL if st.session_state.lang=="el" else KIRA_SYSTEM_EN

# Symptoms where measuring a specific vital genuinely adds value → surface the
# relevant measurement contextually instead of forcing vitals on everyone.
def _strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s.lower())
                   if unicodedata.category(c) != "Mn")
# Each category maps symptom roots → the vital that helps. "scan"=True only where
# the camera face-scan can actually produce the value (heart rate → cardiac only).
_VITAL_CATEGORIES = [
    {"key":"cardio","scan":True,
     "el":"καρδιακός ρυθμός / πίεση","en":"heart rate / blood pressure",
     "roots":["παλμ","ταχυκαρδ","αρρυθμ","στηθ","θωρακ","λιποθυμ","λιγοθυμ",
              "εφιδρ","ιδρωτ","ιδρωσ","ιδρων","ζαλ",
              "palpit","racing heart","irregular heart","tachycard","arrhythm",
              "chest pain","chest tightness","faint","sweat","dizz","lightheaded","light-headed"]},
    {"key":"bp","scan":False,
     "el":"αρτηριακή πίεση","en":"blood pressure",
     "roots":["πιεση","υπερτασ","υποτασ","αρτηριακ",
              "blood pressure","hypertens","hypotens"]},
    {"key":"temp","scan":False,
     "el":"θερμοκρασία","en":"temperature",
     "roots":["πυρετ","θερμοκρασ","δεκατ","εμπυρετ","ριγος","ριγη","κρυαδ",
              "fever","febrile","chills","temperature","high temp"]},
    {"key":"resp","scan":True,
     "el":"οξυγόνο (SpO₂) & αναπνοές","en":"oxygen (SpO₂) & breathing",
     "roots":["δυσπν","βηχ","ασθμ","πνευμον","αναπν","λαχαν","συριγμ","βρογχ","κορον","covid",
              "cough","wheez","asthma","pneumonia","breathless","short of breath",
              "shortness of breath","respiratory","oxygen"]},
]
def _relevant_vitals():
    txt = _strip_accents(" ".join(m["content"] for m in st.session_state.triage_chat if m["role"]=="user"))
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
    "χτυπ","χτυπημ","κτυπ","αιματωμ","χτυπησ","χτυπαω","εκχυμ",
    # English
    "skin","rash","lesion","wound","laceration","abrasion","lump","bump","swelling",
    "swollen","bruise","mole","melanoma","eye","throat","tonsil","tongue","nail",
    "burn","bite","itch","blister","eczema","psoriasis","ulcer","pimple","cyst","wart",
    "hematom","haematom","contusion","bruising","impact injury","ecchymosis",
]
def _visual_relevant():
    txt = _strip_accents(" ".join(m["content"] for m in st.session_state.triage_chat))
    return any(r in txt for r in _VISUAL_ROOTS)

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
.hint{{text-align:center;margin:24px 0 0;font-size:12px;color:#94A3B8;border-top:1px dashed #E0E5FF;padding-top:14px}}
@media print{{body{{padding:16px}}.patient,.emergency{{-webkit-print-color-adjust:exact;print-color-adjust:exact}}@page{{margin:15mm}}}}</style></head><body>
<div class="hdr"><div class="hdr-logo">🩺 Asklepios AI Nurse</div><div class="hdr-date">Κλινική Εκτίμηση<br>{ts}</div></div>
<div class="patient"><div class="patient-name">{name}</div><div class="patient-meta">{age} ετών · {sex}</div>
<div class="patient-detail"><strong>Ιστορικό:</strong> {hx}<br><strong>Αλλεργίες:</strong> {allg}<br><strong>Φάρμακα:</strong> {meds}</div></div>
{vitals_sec}<h2>Κλινική Αξιολόγηση</h2>{md2h(report_text or "")}{refs_html}
<div class="emergency">🚨 ΣΕ ΕΠΕΙΓΟΥΣΑ ΑΝΑΓΚΗ: ΚΑΛΕΣΤΕ 166 (ΕΚΑΒ) ή 112</div>
<div class="disclaimer">⚠️ AI-generated. Δεν αποτελεί ιατρική διάγνωση. Απαιτείται επίσκεψη σε επαγγελματία υγείας.</div>
<div class="hint">💡 Ctrl+P → Save as PDF</div></body></html>"""
    return html_out.encode("utf-8")

def render_ad_banner(lang):
    """Promotional ad banner shown above the disclaimer on home screen."""
    import streamlit.components.v1 as components
    html = """<!DOCTYPE html><html><head>
<meta charset="UTF-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,700;0,800;1,700&display=swap');
@import url('https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@3.19.0/tabler-icons.min.css');
*{box-sizing:border-box;margin:0;padding:0}
body{background:transparent;font-family:'DM Sans',sans-serif}
.ad{width:100%;max-width:480px;margin:0 auto;border-radius:24px;overflow:hidden;background:linear-gradient(160deg,#3D2FE7 0%,#7B2FE0 55%,#9B3FFF 100%);padding:28px 24px 24px;position:relative}
.ad::before{content:'';position:absolute;top:-60px;right:-40px;width:200px;height:200px;border-radius:50%;background:rgba(255,255,255,.06)}
.ad::after{content:'';position:absolute;bottom:-40px;left:-30px;width:160px;height:160px;border-radius:50%;background:rgba(255,255,255,.04)}
.topbar{display:flex;align-items:center;gap:10px;margin-bottom:24px;position:relative;z-index:1}
.logo-box{width:40px;height:40px;border-radius:12px;background:rgba(255,255,255,.18);display:flex;align-items:center;justify-content:center}
.logo-box i{font-size:20px;color:#fff}
.brand-text .name{font-size:15px;font-weight:700;color:#fff;line-height:1}
.brand-text .sub{font-size:11px;color:rgba(255,255,255,.65);letter-spacing:.06em;text-transform:uppercase;margin-top:2px}
.headline{position:relative;z-index:1;margin-bottom:10px}
.headline h1{font-size:36px;font-weight:800;color:#fff;line-height:1.15;letter-spacing:-0.5px}
.headline h1 em{color:#F9C846;font-style:italic}
.subline{font-size:14px;color:rgba(255,255,255,.75);line-height:1.55;margin-bottom:20px;position:relative;z-index:1}
.chat-bubble{background:rgba(255,255,255,.13);border:1px solid rgba(255,255,255,.2);border-radius:14px;padding:12px 14px;margin-bottom:20px;display:flex;align-items:flex-start;gap:10px;position:relative;z-index:1}
.bubble-av{width:28px;height:28px;border-radius:50%;background:rgba(255,255,255,.25);display:flex;align-items:center;justify-content:center;flex-shrink:0}
.bubble-av i{font-size:14px;color:#fff}
.bubble-text{font-size:12.5px;color:#fff;line-height:1.55;font-style:italic}
.bubble-text strong{font-style:normal;font-weight:700}
.arrow-down{text-align:center;color:rgba(255,255,255,.45);font-size:18px;margin:-8px 0 12px;position:relative;z-index:1}
.features{display:flex;flex-direction:column;gap:9px;margin-bottom:22px;position:relative;z-index:1}
.feat{display:flex;align-items:center;gap:11px}
.feat-icon{width:32px;height:32px;border-radius:10px;background:rgba(255,255,255,.14);display:flex;align-items:center;justify-content:center;flex-shrink:0}
.feat-icon i{font-size:16px;color:#fff}
.feat-text{font-size:13px;color:rgba(255,255,255,.9);line-height:1.3}
.cta-btn{width:100%;padding:16px;border-radius:16px;background:linear-gradient(135deg,#3D2FE7,#7B2FE0);border:none;cursor:pointer;font-family:'DM Sans',sans-serif;font-size:15px;font-weight:700;color:#ffffff;letter-spacing:.01em;position:relative;z-index:1;transition:background .25s,color .25s;outline:none}
.cta-btn:hover{background:#ffffff;color:#1a1a1a}
.cta-btn:active{transform:scale(.98)}
.disc{font-size:11px;color:rgba(255,255,255,.45);text-align:center;margin-top:10px;line-height:1.5;position:relative;z-index:1}
</style></head><body>
<div class="ad">
  <div class="topbar">
    <div class="logo-box"><i class="ti ti-stethoscope"></i></div>
    <div class="brand-text"><div class="name">Asklepios</div><div class="sub">AI Nurse</div></div>
  </div>
  <div class="headline"><h1 id="h1"></h1></div>
  <p class="subline" id="sub"></p>
  <div class="chat-bubble">
    <div class="bubble-av"><i class="ti ti-robot"></i></div>
    <div class="bubble-text" id="bubble"></div>
  </div>
  <div class="arrow-down">⌄</div>
  <div class="features">
    <div class="feat"><div class="feat-icon"><i class="ti ti-clock"></i></div><div class="feat-text" id="f1"></div></div>
    <div class="feat"><div class="feat-icon"><i class="ti ti-flag"></i></div><div class="feat-text" id="f2"></div></div>
    <div class="feat"><div class="feat-icon"><i class="ti ti-microscope"></i></div><div class="feat-text" id="f3"></div></div>
  </div>
  <button class="cta-btn" id="cta" onclick="scrollToStart()"></button>
  <p class="disc" id="disc"></p>
</div>
<script>
var lg="'+lang+'";
var T={
  el:{h1:"O νοσηλευτής<br>στην <em>τσέπη</em> σου.",sub:"Γρήγορη εκτίμηση υγείας, στα ελληνικά — όποτε τη χρειαστείς.",bubble:"«Περίγραψέ μου τι νιώθεις και θα σε καθοδηγήσω — <strong>βήμα-βήμα.</strong>»",f1:"Πρώτη εκτίμηση σε ~2 λεπτά",f2:"Στα ελληνικά, διαθέσιμος 24/7",f3:"Με επιστημονική τεκμηρίωση",cta:"Δοκίμασέ το δωρεάν →",disc:"Ενημερωτικό εργαλείο. Δεν αντικαθιστά ιατρική διάγνωση ή θεραπεία."},
  en:{h1:"The nurse<br>in your <em>pocket.</em>",sub:"Quick health assessment, in your language — whenever you need it.",bubble:"«Tell me how you feel and I will guide you — <strong>step by step.</strong>»",f1:"First assessment in ~2 minutes",f2:"In Greek & English, available 24/7",f3:"Evidence-based, powered by PubMed",cta:"Try it for free →",disc:"Informational tool only. Does not replace medical diagnosis or treatment."}
};
function apply(l){
  var d=T[l]||T.el;
  document.getElementById("h1").innerHTML=d.h1;
  document.getElementById("sub").textContent=d.sub;
  document.getElementById("bubble").innerHTML=d.bubble;
  ["f1","f2","f3","disc"].forEach(function(id){document.getElementById(id).textContent=d[id];});
  document.getElementById("cta").textContent=d.cta;
}
function scrollToStart(){
  try{
    var p=window.parent.document;
    var btns=p.querySelectorAll("button");
    var target=null;
    btns.forEach(function(b){if(b.innerText&&(b.innerText.includes("Έναρξη")||b.innerText.includes("Start")||b.innerText.includes("Εναρξη")))target=b;});
    if(target){target.scrollIntoView({behavior:"smooth",block:"center"});setTimeout(function(){target.focus();},600);}
    else window.parent.scrollTo({top:window.parent.document.body.scrollHeight,behavior:"smooth"});
  }catch(e){try{window.parent.scrollTo({top:9999,behavior:"smooth"});}catch(e2){}}
}
apply(lg);
</script></body></html>"""
    html = html.replace('"{LANG}"', '"'+ lang + '"')
    components.html(html, height=580, scrolling=False)

def render_explainer_video(lang):
    """Embedded how-it-works explainer — shown on home screen."""
    import streamlit.components.v1 as components
    html = '''
<style>
@import url(\'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap\');
@import url(\'https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@3.19.0/tabler-icons.min.css\');
*{box-sizing:border-box;margin:0;padding:0}
body{background:transparent;font-family:\'DM Sans\',sans-serif}
.ex{width:100%;border:1px solid rgba(0,0,0,.1);border-radius:14px;overflow:hidden;background:#fff}
.stage{position:relative;width:100%;height:290px;overflow:hidden;background:#F8F9FF}
.slide{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:24px 32px;opacity:0;transform:translateY(14px);transition:opacity .4s,transform .4s;pointer-events:none}
.slide.active{opacity:1;transform:translateY(0);pointer-events:auto}
.slide.exit{opacity:0;transform:translateY(-14px);transition:opacity .25s,transform .25s}
.s-icon{width:52px;height:52px;border-radius:50%;display:flex;align-items:center;justify-content:center;margin-bottom:14px;font-size:22px;flex-shrink:0}
.ic-b{background:#EFF6FF;color:#1D4ED8}.ic-p{background:#EEEDFE;color:#534AB7}
.ic-t{background:#ECFDF5;color:#065F46}.ic-a{background:#FFFBEB;color:#92400E}.ic-c{background:#FEF2F2;color:#991B1B}
.s-step{font-size:10px;font-weight:600;letter-spacing:.09em;color:#9CA3AF;margin-bottom:6px;text-transform:uppercase}
.s-title{font-size:17px;font-weight:600;color:#1A1A2E;text-align:center;margin-bottom:7px;line-height:1.3}
.s-sub{font-size:12px;color:#6B7280;text-align:center;line-height:1.6;max-width:380px}
.chips{display:flex;gap:7px;flex-wrap:wrap;justify-content:center;margin-top:12px}
.chip{font-size:11px;font-weight:500;padding:4px 11px;border-radius:20px;border:1px solid #E5E7EB;color:#374151;background:#fff}
.chip.hi{background:#EFF6FF;color:#1D4ED8;border-color:transparent}
.opt-row{display:flex;gap:9px;margin-top:14px;flex-wrap:wrap;justify-content:center}
.opt{display:flex;align-items:flex-start;gap:7px;padding:9px 12px;border-radius:10px;border:1px solid #E5E7EB;background:#fff;max-width:148px}
.num{width:18px;height:18px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:700;flex-shrink:0;margin-top:1px}
.n1{background:#F3F4F6;color:#6B7280}.n2{background:#EFF6FF;color:#1D4ED8}.n3{background:#EEEDFE;color:#534AB7}
.ot{font-size:11px;font-weight:600;color:#1A1A2E;margin-bottom:2px}.os{font-size:10px;color:#9CA3AF;line-height:1.4}
.ai-box{margin-top:14px;padding:10px 14px;border-radius:10px;background:#fff;border:1px solid #E5E7EB;max-width:360px;width:100%}
.ai-row{display:flex;align-items:flex-start;gap:7px}
.ai-av{width:24px;height:24px;border-radius:50%;background:#EFF6FF;display:flex;align-items:center;justify-content:center;flex-shrink:0}
.ai-bub{font-size:11px;color:#6B7280;line-height:1.55}
.ai-sug{margin-top:7px;padding:5px 9px;background:#F9FAFB;border-radius:8px;font-size:11px;color:#1D4ED8;display:flex;align-items:center;gap:5px;border:1px solid #E5E7EB}
.prog{height:2px;background:#F3F4F6}
.pfill{height:100%;background:#2D3FE7;border-radius:1px;transition:width .3s}
.ctrl{display:flex;align-items:center;justify-content:space-between;padding:9px 13px;border-top:1px solid #F3F4F6;background:#fff}
.dots{display:flex;gap:5px;align-items:center}
.dot{width:5px;height:5px;border-radius:50%;background:#E5E7EB;cursor:pointer;border:none;padding:0;transition:all .25s}
.dot.on{width:14px;border-radius:3px;background:#2D3FE7}
.ibtn{width:28px;height:28px;border-radius:50%;border:1px solid #E5E7EB;background:#fff;cursor:pointer;display:flex;align-items:center;justify-content:center;color:#6B7280;font-size:13px;transition:background .15s}
.ibtn:hover{background:#F9FAFB}.ibtn:disabled{opacity:.3;cursor:default}
.pbtn{width:28px;height:28px;border-radius:50%;border:1px solid #D1D5DB;background:#fff;cursor:pointer;display:flex;align-items:center;justify-content:center;color:#1A1A2E;font-size:13px}
.pbtn:hover{background:#F9FAFB}
.sc{font-size:10px;color:#9CA3AF;min-width:28px;text-align:center}
.lbar{display:flex;justify-content:center;gap:7px;padding:7px 13px 11px;border-top:1px solid #F3F4F6;background:#fff}
.lbtn{font-size:11px;padding:3px 11px;border-radius:20px;border:1px solid #E5E7EB;background:#fff;cursor:pointer;color:#6B7280;transition:.15s}
.lbtn.on{border-color:#2D3FE7;color:#2D3FE7;font-weight:600}
</style>
<div class="ex">
<div class="stage">
<div class="slide active" id="s0">
  <div class="s-icon ic-b"><i class="ti ti-stethoscope"></i></div>
  <div class="s-step" id="p0">asklepios · ai nurse</div>
  <div class="s-title" id="h0">Ο ψηφιακός σου νοσηλευτής</div>
  <div class="s-sub" id="b0">Αξιολόγηση συμπτωμάτων με AI — γρήγορα, στα ελληνικά, πάντα δίπλα σου.</div>
</div>
<div class="slide" id="s1">
  <div class="s-icon ic-p"><i class="ti ti-login"></i></div>
  <div class="s-step" id="p1">βήμα 1</div>
  <div class="s-title" id="h1">Σύνδεση με email</div>
  <div class="s-sub" id="b1">Εισάγεις το email σου, λαμβάνεις κωδικό OTP. Χωρίς password, χωρίς λογαριασμό.</div>
</div>
<div class="slide" id="s2">
  <div class="s-icon ic-t"><i class="ti ti-user-circle"></i></div>
  <div class="s-step" id="p2">βήμα 2</div>
  <div class="s-title" id="h2">Συμπλήρωσε το προφίλ σου</div>
  <div class="s-sub" id="b2">Όνομα, ηλικία, φύλο, ιστορικό, αλλεργίες, φάρμακα. Αποθηκεύεται κρυπτογραφημένο.</div>
  <div class="chips"><span class="chip hi" id="c2a">Ηλικία</span><span class="chip hi" id="c2b">Ιστορικό</span><span class="chip hi" id="c2c">Φάρμακα</span><span class="chip hi" id="c2d">Αλλεργίες</span></div>
</div>
<div class="slide" id="s3">
  <div class="s-icon ic-b"><i class="ti ti-message-chatbot"></i></div>
  <div class="s-step" id="p3">βήμα 3</div>
  <div class="s-title" id="h3">Περίγραψε τα συμπτώματά σου</div>
  <div class="s-sub" id="b3">Ο Asklepios κάνει στοχευμένες ερωτήσεις — μία κάθε φορά. Chips ή ελεύθερο κείμενο.</div>
  <div class="chips"><span class="chip" id="c3a">Πονοκέφαλος</span><span class="chip" id="c3b">Δύσπνοια</span><span class="chip" id="c3c">Τραύμα</span><span class="chip" id="c3d">Άλλο</span></div>
</div>
<div class="slide" id="s4">
  <div class="s-icon ic-a"><i class="ti ti-heart-rate-monitor"></i></div>
  <div class="s-step" id="p4">προαιρετικό · ζωτικά</div>
  <div class="s-title" id="h4">Μέτρηση ζωτικών — 3 επιλογές</div>
  <div class="opt-row">
    <div class="opt"><div class="num n1">1</div><div><div class="ot" id="o1t">Παράλειψη</div><div class="os" id="o1s">Συνέχισε χωρίς μέτρηση</div></div></div>
    <div class="opt"><div class="num n2">2</div><div><div class="ot" id="o2t">Χειροκίνητη</div><div class="os" id="o2s">Καταχώρισε μόνος σου</div></div></div>
    <div class="opt"><div class="num n3">3</div><div><div class="ot" id="o3t">Σάρωση</div><div class="os" id="o3s">Κάμερα · 2 γύροι 60\'\'</div></div></div>
  </div>
</div>
<div class="slide" id="s5">
  <div class="s-icon ic-c"><i class="ti ti-camera"></i></div>
  <div class="s-step" id="p5">ai πρόταση · εφόσον χρειαστεί</div>
  <div class="s-title" id="h5">Φωτό ή σάρωση — μόνο αν χρειαστεί</div>
  <div class="ai-box">
    <div class="ai-row"><div class="ai-av"><i class="ti ti-robot" style="font-size:12px;color:#1D4ED8"></i></div><div class="ai-bub" id="aib">Βλέπω ότι έχεις χτύπημα — μπορείς να ανεβάσεις φωτογραφία για πιο ακριβή εκτίμηση.</div></div>
    <div class="ai-sug"><i class="ti ti-camera" style="font-size:12px"></i><span id="tag1">📷 Ανάλυση φωτογραφίας</span></div>
    <div class="ai-sug" style="margin-top:4px"><i class="ti ti-scan" style="font-size:12px"></i><span id="tag2">📡 Σάρωση για δύσπνοια / καρδιά</span></div>
  </div>
</div>
<div class="slide" id="s6">
  <div class="s-icon ic-t"><i class="ti ti-report-medical"></i></div>
  <div class="s-step" id="p6">βήμα 4</div>
  <div class="s-title" id="h6">Αναλυτική αναφορά υγείας</div>
  <div class="s-sub" id="b6">Κλινική εκτίμηση με PubMed + GPT-4o. Εκτύπωσε ή αποθήκευσε ως PDF για τον γιατρό σου.</div>
  <div class="chips"><span class="chip hi" id="c6a">PubMed</span><span class="chip hi" id="c6b">Dual AI</span><span class="chip hi" id="c6c">PDF</span></div>
</div>
</div>
<div class="prog"><div class="pfill" id="pf" style="width:14.3%"></div></div>
<div class="ctrl">
  <button class="ibtn" id="pb" onclick="nav(-1)" disabled><i class="ti ti-chevron-left"></i></button>
  <div style="display:flex;align-items:center;gap:8px">
    <div class="dots" id="dots"></div>
    <button class="pbtn" id="plb" onclick="togPlay()"><i class="ti ti-player-play" id="pli"></i></button>
  </div>
  <div style="display:flex;align-items:center;gap:5px">
    <span class="sc" id="sc">1 / 7</span>
    <button class="ibtn" id="nb" onclick="nav(1)"><i class="ti ti-chevron-right"></i></button>
  </div>
</div>
<div class="lbar">
  <button class="lbtn on" id="lel" onclick="setL(\'el\')">🇬🇷 Ελληνικά</button>
  <button class="lbtn" id="len" onclick="setL(\'en\')">🇬🇧 English</button>
</div>
</div>
<script>
const T=7;let cur=0,lg=\'el\',play=false,tim=null;
const D={
  el:{p:[\'asklepios · ai nurse\',\'βήμα 1\',\'βήμα 2\',\'βήμα 3\',\'προαιρετικό · ζωτικά\',\'ai πρόταση · εφόσον χρειαστεί\',\'βήμα 4\'],
  h:[\'Ο ψηφιακός σου νοσηλευτής\',\'Σύνδεση με email\',\'Συμπλήρωσε το προφίλ σου\',\'Περίγραψε τα συμπτώματά σου\',\'Μέτρηση ζωτικών — 3 επιλογές\',\'Φωτό ή σάρωση — μόνο αν χρειαστεί\',\'Αναλυτική αναφορά υγείας\'],
  b:[\'Αξιολόγηση συμπτωμάτων με AI — γρήγορα, στα ελληνικά, πάντα δίπλα σου.\',\'Εισάγεις το email σου, λαμβάνεις κωδικό OTP. Χωρίς password, χωρίς λογαριασμό.\',\'Όνομα, ηλικία, φύλο, ιστορικό, αλλεργίες, φάρμακα. Αποθηκεύεται κρυπτογραφημένο.\',\'Ο Asklepios κάνει στοχευμένες ερωτήσεις — μία κάθε φορά. Chips ή ελεύθερο κείμενο.\',\'\',\'\',\'Κλινική εκτίμηση με PubMed + GPT-4o. Εκτύπωσε ή αποθήκευσε ως PDF για τον γιατρό σου.\'],
  c2:[\'Ηλικία\',\'Ιστορικό\',\'Φάρμακα\',\'Αλλεργίες\'],c3:[\'Πονοκέφαλος\',\'Δύσπνοια\',\'Τραύμα\',\'Άλλο\'],c6:[\'PubMed\',\'Dual AI\',\'PDF\'],
  o1t:\'Παράλειψη\',o1s:\'Συνέχισε χωρίς μέτρηση\',o2t:\'Χειροκίνητη\',o2s:\'Καταχώρισε μόνος σου\',o3t:\'Σάρωση\',o3s:"Κάμερα · 2 γύροι 60\'\'",
  aib:"Βλέπω ότι έχεις χτύπημα — μπορείς να ανεβάσεις φωτογραφία για πιο ακριβή εκτίμηση.",
  t1:\'📷 Ανάλυση φωτογραφίας\',t2:\'📡 Σάρωση για δύσπνοια / καρδιά\'},
  en:{p:[\'asklepios · ai nurse\',\'step 1\',\'step 2\',\'step 3\',\'optional · vitals\',\'ai suggestion · when needed\',\'step 4\'],
  h:[\'Your digital nurse\',\'Sign in with email\',\'Fill in your profile\',\'Describe your symptoms\',\'Measure vitals — 3 options\',\'Photo or scan — only when needed\',\'Detailed health report\'],
  b:[\'AI-powered symptom assessment — fast, in your language, always by your side.\',\'Enter your email, receive an OTP code. No password, no account needed.\',\'Name, age, sex, history, allergies, medications. Stored encrypted.\',\'Asklepios asks targeted questions — one at a time. Chips or free text.\',\'\',\'\',\'Clinical assessment with PubMed + GPT-4o. Print or save as PDF for your doctor.\'],
  c2:[\'Age\',\'History\',\'Meds\',\'Allergies\'],c3:[\'Headache\',\'Dyspnoea\',\'Injury\',\'Other\'],c6:[\'PubMed\',\'Dual AI\',\'PDF\'],
  o1t:\'Skip\',o1s:\'Continue without measuring\',o2t:\'Manual\',o2s:\'Enter values yourself\',o3t:\'Scan\',o3s:"Camera · 2 rounds 60\'\'",
  aib:\'I see you have an injury — you can upload a photo for a more accurate assessment.\',
  t1:\'📷 Photo analysis\',t2:\'📡 Scan for dyspnoea / heart\'}
};
function ap(l){
  const d=D[l];
  for(let i=0;i<T;i++){
    const pe=document.getElementById(\'p\'+i),he=document.getElementById(\'h\'+i),be=document.getElementById(\'b\'+i);
    if(pe)pe.textContent=d.p[i];if(he)he.textContent=d.h[i];if(be&&d.b[i])be.textContent=d.b[i];
  }
  [\'a\',\'b\',\'c\',\'d\'].forEach((x,i)=>{const e=document.getElementById(\'c2\'+x);if(e)e.textContent=d.c2[i];});
  [\'a\',\'b\',\'c\',\'d\'].forEach((x,i)=>{const e=document.getElementById(\'c3\'+x);if(e)e.textContent=d.c3[i];});
  [\'a\',\'b\',\'c\'].forEach((x,i)=>{const e=document.getElementById(\'c6\'+x);if(e)e.textContent=d.c6[i];});
  const m={o1t:d.o1t,o1s:d.o1s,o2t:d.o2t,o2s:d.o2s,o3t:d.o3t,o3s:d.o3s,aib:d.aib,tag1:d.t1,tag2:d.t2};
  Object.entries(m).forEach(([k,v])=>{const e=document.getElementById(k);if(e)e.textContent=v;});
  document.getElementById(\'lel\').className=\'lbtn\'+(l===\'el\'?\' on\':\'\');
  document.getElementById(\'len\').className=\'lbtn\'+(l===\'en\'?\' on\':\'\');
}
function setL(l){lg=l;ap(l);}
function mkDots(){
  const d=document.getElementById(\'dots\');d.innerHTML=\'\';
  for(let i=0;i<T;i++){
    const b=document.createElement(\'button\');b.className=\'dot\'+(i===cur?\' on\':\'\');
    b.onclick=(idx=>()=>go(idx))(i);d.appendChild(b);
  }
}
function go(idx){
  const sl=document.querySelectorAll(\'.slide\');
  sl[cur].classList.remove(\'active\');sl[cur].classList.add(\'exit\');
  setTimeout(()=>sl[cur].classList.remove(\'exit\'),280);
  cur=idx;sl[cur].classList.add(\'active\');
  document.getElementById(\'pf\').style.width=((cur+1)/T*100)+\'%\';
  document.getElementById(\'sc\').textContent=(cur+1)+\' / \'+T;
  document.getElementById(\'pb\').disabled=cur===0;
  document.getElementById(\'nb\').disabled=cur===T-1;
  mkDots();
}
function nav(d){
  const n=cur+d;if(n>=0&&n<T)go(n);
  if(play&&n===T-1){clearInterval(tim);play=false;document.getElementById(\'pli\').className=\'ti ti-player-play\';}
}
function togPlay(){
  play=!play;document.getElementById(\'pli\').className=play?\'ti ti-player-pause\':\'ti ti-player-play\';
  if(play){if(cur===T-1)go(0);tim=setInterval(()=>{if(cur<T-1)nav(1);else{clearInterval(tim);play=false;document.getElementById(\'pli\').className=\'ti ti-player-play\';}},3500);}
  else clearInterval(tim);
}
mkDots();ap(\'el\');
</script>
\'\'\'
    components.html(html, height=420, scrolling=False)

def render_home():
    c1,c2,c3=st.columns([5,1,1])
    with c2:
        if st.button("🇬🇧 EN" if st.session_state.lang=="el" else "🇬🇷 ΕΛ"):
            st.session_state.lang="en" if st.session_state.lang=="el" else "el"; st.rerun()
    with c3:
        if is_logged_in():
            if st.button("🚪 " + ("Έξοδος" if st.session_state.lang=="el" else "Logout"), use_container_width=True, key="logout_home"):
                logout(); st.rerun()
    st.markdown(f'''<div class="kira-hero"><div style="font-size:64px;margin-bottom:8px">🩺</div><h1>{t("title")}</h1><p>{t("subtitle")}</p><div class="kira-tagline">{t("tagline")}</div></div>''',unsafe_allow_html=True)
    render_ad_banner(st.session_state.lang)
    st.markdown(f'<div class="disclaimer">{t("disclaimer_main")}</div>',unsafe_allow_html=True)
    # Explainer video: always visible on home, but collapsible for returning users
    _has_history = bool(st.session_state.triage_chat or st.session_state.profile.get("name"))
    if not _has_history:
        render_explainer_video(st.session_state.lang)
    else:
        with st.expander("▶ " + ("Πώς λειτουργεί;" if st.session_state.lang=="el" else "How does it work?"), expanded=False):
            render_explainer_video(st.session_state.lang)
    col1,col2,col3=st.columns([1,2,1])
    with col2:
        if st.button(t("start"),type="primary",use_container_width=True):
            st.session_state.screen="intake"; st.rerun()
        # Show "New Chat" button if a previous conversation exists
        if st.session_state.triage_chat or st.session_state.profile.get("name"):
            if st.button(("🔄 Νέα Συνομιλία" if st.session_state.lang=="el" else "🔄 New Chat"), use_container_width=True, key="new_chat_home"):
                # Clear conversation and profile but keep login
                reset_keys = ["triage_chat","profile","vitals","vitals_analysis","report",
                              "report_pubmed","report_gpt","medications","med_inputs",
                              "symptom_chips","fb_rating","fb_sent","triage_ready",
                              "_draft_loaded","_draft_hash","_from_facescan","_scan_injected",
                              "_scan_reply_pending","_vitals_nudge_off","_fs_banner"]
                for rk in reset_keys:
                    st.session_state.pop(rk, None)
                if is_logged_in():
                    delete_draft(st.session_state.get("auth_user", ""))
                st.rerun()
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
    with c2: age=st.number_input(t("age"),min_value=0,max_value=120,value=st.session_state.profile.get("age",40))
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
    lang=st.session_state.lang
    st.markdown(f"## 📊 {t('vitals_title')} — {p.get('name','')}")

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
                classify_vitals(vd)
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
            st.session_state.vitals=vd; classify_vitals(vd)
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

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(analysis)
                st.markdown('</div>', unsafe_allow_html=True)

                urgent_kw = ["urgent","immediate","επείγον","άμεσα","ιατρό αμέσως","emergency","melanoma","cancer","carcinoma","καρκίν"]
                if any(k.lower() in analysis.lower() for k in urgent_kw):
                    st.error("🚨 " + ("Επείγοντα ευρήματα — επικοινωνήστε με ιατρό ΑΜΕΣΑ" if lang=="el"
                                      else "Urgent findings — contact a doctor IMMEDIATELY"))

                st.session_state["photo_findings"] = {
                    "scan_type": scan_k, "scan_label": sel,
                    "florence_desc": f2_desc, "analysis": analysis,
                }
                if st.button("➤ " + ("Πρόσθεση στην εκτίμηση" if lang=="el" else "Add to assessment"),
                             type="primary", use_container_width=True, key="photo_to_triage_h"):
                    msg = (f"Αποτέλεσμα φωτογραφικής ανάλυσης ({sel}):\n\n{analysis}"
                           if lang=="el" else
                           f"Photo analysis result ({sel}):\n\n{analysis}")
                    st.session_state.triage_chat.append({"role":"user","content":msg})
                    st.session_state["photo_added"] = True
                    st.rerun()
    else:
        st.info("👆 " + ("Ανεβάστε φωτογραφία για να ξεκινήσει η ανάλυση" if lang=="el"
                        else "Upload a photo to begin analysis"))

def render_triage():
    render_stepper("triage")
    p=st.session_state.profile
    st.markdown(f"## 🩺 {t('triage_title')} — {p.get('name','')}")
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
        _PER_ROW = 3
        for _rs in range(0, len(chips), _PER_ROW):
            _row = chips[_rs:_rs+_PER_ROW]
            _cc = st.columns(len(_row))
            for _j, chip in enumerate(_row):
                _i = _rs + _j
                with _cc[_j]:
                    sel = chip in st.session_state.symptom_chips
                    if st.button(("✓ " if sel else "")+chip, key=f"chip_{_i}", use_container_width=True):
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
                st.markdown(f'<a href="{_link}" target="_blank" style="display:block;text-align:center;padding:8px;border-radius:8px;background:#2D3FE7;color:white;text-decoration:none;font-weight:600;font-size:13px">📷 {"Σάρωση" if _lang=="el" else "Scan"}</a>', unsafe_allow_html=True)
            _ci += 1
        with _cols[_ci]:
            if st.button(("Όχι τώρα" if _lang=="el" else "Not now"), key="nudge_off", use_container_width=True):
                st.session_state["_vitals_nudge_off"] = True; st.rerun()
    # Photo analysis appears only after an initial assessment AND only when the
    # complaint is something visible (skin, eye, wound, throat, nails...). For
    # non-visual issues (e.g. chest pain) a photo adds nothing, so it stays hidden.
    if any(m["role"]=="assistant" for m in st.session_state.triage_chat) and _visual_relevant():
        with st.expander("📷 " + ("Ανάλυση φωτογραφίας (προαιρετικό)" if st.session_state.lang=="el" else "Photo analysis (optional)")):
            render_photo_scan()
    # Confirmation after a photo was added — guide the user to keep answering
    if st.session_state.get("photo_added"):
        last_q = next((m["content"] for m in reversed(st.session_state.triage_chat) if m["role"]=="assistant"), "")
        if st.session_state.lang=="el":
            st.success("✅ Η ανάλυση της εικόνας προστέθηκε στην εκτίμηση. Συνέχισε απαντώντας στην τελευταία ερώτηση του Asklepios παρακάτω.")
        else:
            st.success("✅ The image analysis was added to the assessment. Continue by answering Asklepios's last question below.")
        if last_q:
            st.info(("🩺 Τελευταία ερώτηση: " if st.session_state.lang=="el" else "🩺 Last question: ") + last_q)
    ready_phrases=["έχω αρκετά στοιχεία","μπορούμε να δημιουργήσουμε","i have enough information","we can generate","full clinical report","πλήρη αναφορά"]
    last_kira=next((m["content"].lower() for m in reversed(st.session_state.triage_chat) if m["role"]=="assistant"),"")
    triage_ready=any(ph in last_kira for ph in ready_phrases)
    user_input=st.chat_input(t("triage_placeholder"),key="triage_input")
    _auto_reply = st.session_state.pop("_scan_reply_pending", False)
    if user_input or _auto_reply:
        if user_input:
            st.session_state.pop("photo_added", None)
            st.session_state.triage_chat.append({"role":"user","content":user_input})
        with st.spinner("Asklepios..."):
            pp=p.get
            profile_ctx=f"Patient: {pp('name')}, {pp('age')}yo {pp('sex')}, Hx: {pp('history','none')}, Allergies: {pp('allergies','none')}, Meds: {pp('meds_raw','none')}"
            vitals_ctx="Vitals: "+", ".join(f"{k}={val}" for k,val in st.session_state.vitals.items()) if st.session_state.vitals else "Vitals: not provided"
            system_ctx=kira_system()+f"\n\n{profile_ctx}\n{vitals_ctx}"
            reply=claude([{"role":m["role"],"content":m["content"]} for m in st.session_state.triage_chat],system=system_ctx,max_tokens=1500)
            if reply and reply.strip() and reply.strip()[-1] not in ".!?»)": reply=reply.rstrip()+" ..."
        st.session_state.triage_chat.append({"role":"assistant","content":reply}); st.rerun()
    col_b,col_r,col_lo=st.columns([1,2,1])
    with col_b:
        if st.button(t("back")): st.session_state.screen="vitals"; st.rerun()
    with col_r:
        enabled=triage_ready or len(st.session_state.triage_chat)>=6
        if st.button(t("generate_report"),type="primary",use_container_width=True,disabled=not enabled):
            st.session_state.screen="report"; st.rerun()
    with col_lo:
        if is_logged_in():
            if st.button("🚪 " + ("Έξοδος" if st.session_state.lang=="el" else "Logout"), use_container_width=True, key="logout_triage"):
                logout(); st.rerun()
    if not enabled:
        st.caption("Συνεχίστε — ο Asklepios θα σας ειδοποιήσει όταν έχει αρκετά." if st.session_state.lang=="el" else "Continue — Asklepios will let you know when it has enough.")

def render_report():
    render_stepper("report")
    p=st.session_state.profile; lang=st.session_state.lang
    st.markdown(f"## 📋 {t('report_title')}"); st.caption(f"{p.get('name','')}, {p.get('age')}y · {datetime.now().strftime('%d %b %Y %H:%M')}")
    if is_logged_in():
        lo1,lo2=st.columns([5,1])
        with lo2:
            if st.button(("Έξοδος" if lang=="el" else "Logout"), key="logout_btn", use_container_width=True):
                logout(); st.rerun()
    render_vitals_summary()
    if not st.session_state.report:
        conversation="\n".join(f"{'Patient' if m['role']=='user' else 'Asklepios'}: {m['content']}" for m in st.session_state.triage_chat)
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
Write these sections IN THIS ORDER, using EXACTLY these headers as written (do not abbreviate or drop letters):
{"1. ΚΥΡΙΟ ΠΑΡΑΠΟΝΟ  2. ΙΣΤΟΡΙΚΟ  3. ΕΚΤΙΜΗΣΗ (Πρωτεύουσα Διάγνωση + Διαφορικές Διαγνώσεις)  4. ΘΕΡΑΠΕΥΤΙΚΟ ΠΛΑΝΟ  5. ΚΟΚΚΙΝΕΣ ΣΗΜΑΙΕΣ  6. ΒΙΒΛΙΟΓΡΑΦΙΑ" if lang=="el" else "1. CHIEF COMPLAINT  2. HISTORY  3. ASSESSMENT (Primary Diagnosis + Differentials)  4. TREATMENT PLAN  5. RED FLAGS  6. REFERENCES"}
For the differentials use a markdown table with EXACTLY 3 columns and these short headers: {"| Διάγνωση | % | Σχόλιο |" if lang=="el" else "| Diagnosis | % | Comment |"} (keep the probability header as just "%", and put values like "~8%"). Keep cell text short.
Language: {"Greek" if lang=="el" else "English"}. Be direct. End with a one-line AI disclaimer."""
        with st.spinner("Δημιουργία αναφοράς..." if lang=="el" else "Generating report..."):
            result=claude([{"role":"user","content":report_prompt}],system=kira_system(),max_tokens=4000,timeout=120)
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
                        "_draft_hash","_from_facescan","_scan_injected","_vitals_nudge_off"): st.session_state.pop(fbk, None)
            st.rerun()
    with c2:
        st.download_button("📄 TXT",data=st.session_state.report,file_name=fname+".txt",mime="text/plain",use_container_width=True)
    with c3:
        st.download_button("📄 PDF/HTML",data=generate_html_report(st.session_state.profile,st.session_state.vitals,st.session_state.report,st.session_state.report_pubmed,lang=lang),file_name=fname+".html",mime="text/html",use_container_width=True,help="Open in browser → Ctrl+P → Save as PDF")
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

# Restore the in-progress assessment from the ENCRYPTED server-side draft (Supabase,
# keyed by email): profile AND the conversation, so returning from the face scan
# resumes the SAME assessment instead of starting over. Synchronous fetch — once the
# email is known this is deterministic (no cookie race).
if (auth_enabled() and is_logged_in() and not st.session_state.profile.get("name")
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

# ── LOGIN GATE ────────────────────────────────────────────────────────────────
# Login-first: every visitor is identified in Supabase → Authentication → Users.
if auth_enabled() and not is_logged_in():
    render_login_screen()
    st.stop()

# ── PERSIST on a CLEAN render pass ────────────────────────────────────────────
# The login cookie write on the verify *click* + immediate st.rerun() is unreliable
# (the rerun aborts the stx browser write); a normal render that completes lands it.
if CM is not None and is_logged_in() and not st.session_state.get("_cookie_synced"):
    _save_login_cookie(st.session_state.get("auth_user", ""))
    st.session_state["_cookie_synced"] = True
# Save the encrypted draft server-side, only when it actually changed. Includes the
# conversation so the face-scan round-trip (new tab) resumes the same assessment.
if auth_enabled() and is_logged_in() and st.session_state.profile.get("name"):
    _payload = {
        "profile": st.session_state.profile,
        "lang": st.session_state.lang,
        "triage_chat": st.session_state.triage_chat,
        "medications": st.session_state.medications,
        "vitals_analysis": st.session_state.vitals_analysis,
    }
    _ph = json.dumps(_payload, ensure_ascii=False, sort_keys=True)
    if st.session_state.get("_draft_hash") != _ph:
        save_draft(st.session_state.get("auth_user", ""), _payload)
        st.session_state["_draft_hash"] = _ph

screen=st.session_state.screen
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
else: render_home()
