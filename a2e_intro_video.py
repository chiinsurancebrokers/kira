"""
a2e_intro_video.py — Δημιουργία ενός εφάπαξ "intro" avatar video για το
Asklepios (kira_app.py), μέσω του A2E.ai API.

ΣΗΜΑΝΤΙΚΟ ΟΡΙΟ: το A2E TTS endpoint (send_tts) δέχεται ΜΕΧΡΙ 200 χαρακτήρες
ανά κλήση (επιβεβαιωμένο από το ίδιο το A2E UI — "Characters: 120/200").
Το intro script μας είναι ~600 χαρακτήρες, άρα αυτό το script:
  1) Σπάει το κείμενο σε κομμάτια ≤200 χαρακτήρων (σε όρια προτάσεων/λέξεων,
     όχι στη μέση λέξης).
  2) Καλεί το send_tts ξεχωριστά για κάθε κομμάτι (κάθε κλήση επιστρέφει ένα
     URL ηχητικού αρχείου, φιλοξενούμενο από το A2E).
  3) Κατεβάζει τοπικά κάθε κομμάτι ήχου.
  4) Τα ενώνει σε ΕΝΑ ενιαίο αρχείο με ffmpeg (χρειάζεται να υπάρχει ήδη
     εγκατεστημένο· σχεδόν σε κάθε σύστημα υπάρχει).
  5) Ανεβάζει το ενιαίο αρχείο σε δωρεάν, χωρίς-εγγραφή file host
     (tmpfiles.org, με fallback σε 0x0.st) για να πάρει δημόσιο URL.
  6) Δίνει αυτό το ενιαίο URL στο video/generate ως audioSrc — το A2E
     ρητά υποστηρίζει "upload your own audio" (δεν χρειάζεται να είναι
     προϊόν του δικού του TTS).

ΓΙΑΤΙ ΕΙΝΑΙ ΞΕΧΩΡΙΣΤΟ ΑΡΧΕΙΟ, ΟΧΙ ΜΕΣΑ ΣΤΟ kira_app.py:
Το intro video παράγεται ΜΙΑ φορά (όχι ανά χρήστη/ανά request). Τρέχεις αυτό
το script offline, παίρνεις το τελικό mp4 URL, και το βάζεις ως secret
(A2E_INTRO_VIDEO_URL_EL / _EN) στο Railway. Το kira_app.py απλά παίζει το
βίντεο μ' ένα st.video(url) — καμία κλήση API στο runtime.

ΓΙΑΤΙ ΤΡΕΧΕΙ ΤΟΠΙΚΑ (όχι μέσα στο Claude sandbox):
Το δίκτυο του Claude code-execution περιβάλλοντος επιτρέπει πρόσβαση μόνο σε
μια συγκεκριμένη λίστα domains — το video.a2e.ai ΔΕΝ είναι σε αυτή τη λίστα
(επιστρέφει 403). Άρα αυτό χρειάζεται να τρέξει στον υπολογιστή σου ή στο
Railway Console του ίδιου project.

ΑΠΑΙΤΗΣΕΙΣ:
  - Python 3.8+ (καμία εξωτερική pip εξάρτηση — μόνο stdlib)
  - ffmpeg εγκατεστημένο και διαθέσιμο στο PATH
      Mac:    brew install ffmpeg
      Ubuntu: sudo apt install ffmpeg
      Railway: συνήθως ήδη διαθέσιμο στις βασικές images· αν όχι, πρόσθεσε
               "ffmpeg" στο apt.txt ή στο nixpacks config του project.

ΧΡΗΣΗ:
  export A2E_API_KEY="sk_..."
  python a2e_intro_video.py
"""

import os
import sys
import json
import time
import shutil
import subprocess
import tempfile
import urllib.request
import urllib.parse
import urllib.error

A2E_BASE = "https://video.a2e.ai"
MAX_POLL_SECONDS = 360       # ~6 λεπτά — αρκετό για ένα κλιπ 30-60 δευτ.
POLL_INTERVAL = 6
TTS_CHAR_LIMIT = 200          # επιβεβαιωμένο όριο A2E send_tts


class A2EError(Exception):
    pass


# ─────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────────────────

def _request(path, payload=None, method="POST", api_key="", query=None):
    url = f"{A2E_BASE}{path}"
    if query:
        url += "?" + urllib.parse.urlencode(query)
    data = json.dumps(payload).encode("utf-8") if payload is not None else (
        b"{}" if method == "GET" else None
    )
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        raise A2EError(f"A2E HTTP {e.code} on {path}: {detail[:400]}")
    except urllib.error.URLError as e:
        raise A2EError(f"A2E network error on {path}: {e}")


def _download(url, dest_path):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest_path, "wb") as f:
        shutil.copyfileobj(resp, f)


# ─────────────────────────────────────────────────────────────────────────
# Text chunking — splits on sentence/clause boundaries first, falls back to
# word boundaries, never cuts mid-word. Greek and English both use normal
# punctuation (. ! ? ,), so the same logic works for both scripts.
# ─────────────────────────────────────────────────────────────────────────

def chunk_text(text, limit=TTS_CHAR_LIMIT):
    """Σπάει το text σε κομμάτια <= limit χαρακτήρων. Προτεραιότητα στα
    όρια: (1) τέλος πρότασης [. ! ? ;], (2) κόμμα/παύλα [, —], (3) λέξη.
    Δεν κόβει ποτέ στη μέση λέξης."""
    import re
    sentences = re.split(r"(?<=[.!?;])\s+", text.strip())
    chunks, current = [], ""
    for sent in sentences:
        candidate = (current + " " + sent).strip() if current else sent
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(sent) <= limit:
            current = sent
            continue
        # Single sentence longer than the limit — try clause boundaries
        # (comma / em-dash) first, for a more natural pause than a raw
        # mid-sentence word cut.
        clauses = re.split(r"(?<=[,—])\s+", sent)
        sub = ""
        for clause in clauses:
            cand2 = (sub + " " + clause).strip() if sub else clause
            if len(cand2) <= limit:
                sub = cand2
                continue
            if sub:
                chunks.append(sub)
            if len(clause) <= limit:
                sub = clause
                continue
            # Even a single clause is too long — fall back to word boundaries.
            words, sub2 = clause.split(" "), ""
            for w in words:
                cand3 = (sub2 + " " + w).strip() if sub2 else w
                if len(cand3) <= limit:
                    sub2 = cand3
                else:
                    if sub2:
                        chunks.append(sub2)
                    sub2 = w
            sub = sub2
        current = sub
    if current:
        chunks.append(current)
    return chunks


# ─────────────────────────────────────────────────────────────────────────
# A2E API calls
# ─────────────────────────────────────────────────────────────────────────

def list_avatars(api_key):
    """GET /api/v1/anchor/character_list — λίστα διαθέσιμων avatars."""
    return _request("/api/v1/anchor/character_list", method="GET", api_key=api_key)


def text_to_speech(text, api_key, lang="el", region="GR", speech_rate=1.0,
                    user_voice_id=None):
    """POST /api/v1/video/send_tts για ΕΝΑ κομμάτι κειμένου (<=200 chars)."""
    if len(text) > TTS_CHAR_LIMIT:
        raise A2EError(
            f"text_to_speech() called with {len(text)} chars — "
            f"A2E's limit is {TTS_CHAR_LIMIT}. Use chunk_text() first."
        )
    payload = {"msg": text, "speechRate": speech_rate, "country": lang, "region": region}
    if user_voice_id:
        payload["user_voice_id"] = user_voice_id
    result = _request("/api/v1/video/send_tts", payload, api_key=api_key)
    if result.get("code") != 0:
        raise A2EError(f"TTS failed: {result}")
    return result["data"]


def text_to_speech_long(text, api_key, lang="el", region="GR", speech_rate=1.0,
                         tmp_dir=None, on_chunk=None):
    """Καλύπτει το όριο των 200 χαρακτήρων: σπάει το κείμενο, κάνει TTS σε
    κάθε κομμάτι, κατεβάζει κάθε κομμάτι τοπικά, και τα ενώνει σε ΕΝΑ wav με
    ffmpeg. Επιστρέφει το local path του ενιαίου αρχείου."""
    if shutil.which("ffmpeg") is None:
        raise A2EError(
            "ffmpeg δεν βρέθηκε στο PATH. Εγκατάστησέ το πρώτα "
            "(brew install ffmpeg / apt install ffmpeg) και ξανατρέξε."
        )
    chunks = chunk_text(text, TTS_CHAR_LIMIT)
    tmp_dir = tmp_dir or tempfile.mkdtemp(prefix="a2e_tts_")
    os.makedirs(tmp_dir, exist_ok=True)
    local_paths = []
    for i, chunk in enumerate(chunks, 1):
        if on_chunk:
            on_chunk(i, len(chunks), chunk)
        audio_url = text_to_speech(chunk, api_key, lang=lang, region=region,
                                    speech_rate=speech_rate)
        ext = ".wav" if audio_url.endswith(".wav") else ".mp3"
        local_path = os.path.join(tmp_dir, f"chunk_{i:02d}{ext}")
        _download(audio_url, local_path)
        local_paths.append(local_path)

    if len(local_paths) == 1:
        return local_paths[0]

    concat_list_path = os.path.join(tmp_dir, "concat_list.txt")
    normalized = []
    for i, p in enumerate(local_paths):
        norm_path = os.path.join(tmp_dir, f"norm_{i:02d}.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", p, "-ar", "44100", "-ac", "1", norm_path],
            check=True, capture_output=True,
        )
        normalized.append(norm_path)
    with open(concat_list_path, "w") as f:
        for p in normalized:
            f.write(f"file '{p}'\n")
    merged_path = os.path.join(tmp_dir, "merged.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list_path,
         "-c", "copy", merged_path],
        check=True, capture_output=True,
    )
    return merged_path


def upload_audio_public(local_path):
    """Ανεβάζει το τοπικό αρχείο σε δωρεάν, χωρίς-εγγραφή file host ώστε να
    έχει δημόσιο URL. Δοκιμάζει tmpfiles.org πρώτα, μετά 0x0.st ως fallback."""
    try:
        boundary = "----A2EBoundary"
        with open(local_path, "rb") as f:
            file_bytes = f.read()
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
            f"Content-Type: audio/wav\r\n\r\n"
        ).encode() + file_bytes + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(
            "https://tmpfiles.org/api/v1/upload", data=body, method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        if result.get("status") == "success":
            raw_url = result["data"]["url"]
            direct_url = raw_url.replace("tmpfiles.org/", "tmpfiles.org/dl/", 1)
            return direct_url
    except Exception as e:
        print(f"    ⚠️ tmpfiles.org upload απέτυχε ({e}), δοκιμάζω 0x0.st...")

    try:
        boundary = "----A2EBoundary2"
        with open(local_path, "rb") as f:
            file_bytes = f.read()
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
            f"Content-Type: audio/wav\r\n\r\n"
        ).encode() + file_bytes + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(
            "https://0x0.st", data=body, method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                     "User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            url = resp.read().decode("utf-8").strip()
        if url.startswith("http"):
            return url
    except Exception as e:
        raise A2EError(f"Both tmpfiles.org and 0x0.st uploads failed: {e}")

    raise A2EError("Audio upload failed on all hosts.")


def generate_avatar_video(audio_url, anchor_id, api_key, title="Asklepios Intro",
                           resolution=720, caption_lang="el-GR",
                           is_caption_enabled=True):
    """POST /api/v1/video/generate — ξεκινά τη δημιουργία του avatar video."""
    payload = {
        "title": title,
        "anchor_id": anchor_id,
        "anchor_type": 1,
        "audioSrc": audio_url,
        "resolution": resolution,
        "isSkipRs": False,
        "isCaptionEnabled": is_caption_enabled,
    }
    if is_caption_enabled:
        payload["captionAlign"] = {
            "language": caption_lang,
            "PrimaryColour": "rgba(255,255,255,1)",
            "OutlineColour": "rgba(0,0,0,1)",
            "BorderStyle": 4,
            "BackColour": "rgba(45,63,231,0.85)",
            "FontName": "Arial",
            "Fontsize": 42,
            "subtitle_position": 0.85,
        }
    result = _request("/api/v1/video/generate", payload, api_key=api_key)
    if result.get("code") != 0:
        raise A2EError(f"Video generation failed to start: {result}")
    return result["data"]["_id"]


def check_video_status(task_id, api_key):
    """GET /api/v1/video/awsList — ψάχνει το task_id στη λίστα πρόσφατων
    video tasks."""
    result = _request("/api/v1/video/awsList", method="GET", api_key=api_key,
                       query={"current": 1, "pageSize": 20})
    if result.get("code") != 0:
        raise A2EError(f"Status check failed: {result}")
    items = (result.get("data") or {}).get("data") or []
    for item in items:
        if item.get("_id") == task_id:
            return item
    return None


def wait_for_video(task_id, api_key, on_progress=None, max_seconds=MAX_POLL_SECONDS):
    start = time.time()
    while time.time() - start < max_seconds:
        item = check_video_status(task_id, api_key)
        if item:
            status = item.get("status", "?")
            percent = item.get("process", 0)
            if on_progress:
                on_progress(int(time.time() - start), status, percent)
            if status == "success":
                return item.get("result")
            if status in ("error", "failed"):
                raise A2EError(f"Video generation failed: {item}")
        time.sleep(POLL_INTERVAL)
    raise A2EError(f"Timeout after {max_seconds}s waiting for video {task_id}")


# ─────────────────────────────────────────────────────────────────────────
# Intro script — φυσικά-ομιλούμενο κείμενο. Edit ελεύθερα, είναι strings.
# Το μήκος δεν περιορίζεται πλέον από το A2E 200-char όριο — το chunking
# το χειρίζεται αυτόματα το text_to_speech_long.
# ─────────────────────────────────────────────────────────────────────────
INTRO_SCRIPT_EL = (
    "Γεια σου! Είμαι ο Ασκληπιός, ο AI νοσηλευτής σου. "
    "Περίγραψέ μου τι νιώθεις — έναν πονοκέφαλο, μια ενόχληση, οτιδήποτε σε απασχολεί — "
    "και θα σου κάνω μερικές ερωτήσεις, ακριβώς όπως θα έκανε ένας νοσηλευτής στο ιατρείο. "
    "Μπορώ επίσης να καταγράψω τα ζωτικά σου σημεία, να διαβάσω εργαστηριακές εξετάσεις, "
    "και στο τέλος θα ετοιμάσω μια πλήρη αναφορά με επιστημονική τεκμηρίωση από το PubMed, "
    "έτοιμη να την πάρεις μαζί σου στον γιατρό σου. "
    "Θυμήσου πάντα: αυτό είναι ένα εργαλείο ενημέρωσης, όχι αντικαθιστά τον γιατρό σου. "
    "Σε επείγουσα κατάσταση, κάλεσε πάντα το 166 ή το 112. "
    "Έτοιμος να ξεκινήσουμε;"
)

INTRO_SCRIPT_EN = (
    "Hi! I'm Asklepios, your AI nurse. "
    "Tell me what you're feeling — a headache, some discomfort, anything on your mind — "
    "and I'll ask you a few questions, just like a nurse would at the clinic. "
    "I can also record your vital signs, read your lab results, "
    "and at the end I'll prepare a full report backed by scientific references from PubMed, "
    "ready for you to bring to your doctor. "
    "Always remember: this is an informational tool, not a replacement for your doctor. "
    "In an emergency, always call 166 or 112. "
    "Ready to get started?"
)


def build_intro_video(lang, api_key, anchor_id, tmp_dir):
    script = INTRO_SCRIPT_EL if lang == "el" else INTRO_SCRIPT_EN
    country, region = ("el", "GR") if lang == "el" else ("en", "US")
    caption_lang = "el-GR" if lang == "el" else "en-US"

    n_chunks = len(chunk_text(script, TTS_CHAR_LIMIT))
    print(f"\n[{lang.upper()}] Κείμενο {len(script)} χαρακτήρων → {n_chunks} "
          f"κομμάτι(α) TTS (όριο A2E: {TTS_CHAR_LIMIT}/κλήση)")

    def _on_chunk(i, n, text_chunk):
        preview = text_chunk[:60] + "..." if len(text_chunk) > 60 else text_chunk
        print(f"    [{i}/{n}] TTS για: \"{preview}\"")

    merged_audio_path = text_to_speech_long(
        script, api_key, lang=country, region=region,
        tmp_dir=os.path.join(tmp_dir, lang), on_chunk=_on_chunk,
    )
    print(f"    ✅ Ενιαίο audio αρχείο: {merged_audio_path}")

    print(f"[{lang.upper()}] Ανέβασμα ενιαίου audio σε δημόσιο URL...")
    public_audio_url = upload_audio_public(merged_audio_path)
    print(f"    ✅ Public audio URL: {public_audio_url}")

    print(f"[{lang.upper()}] Εκκίνηση δημιουργίας avatar video...")
    task_id = generate_avatar_video(
        public_audio_url, anchor_id, api_key,
        title=f"Asklepios Intro ({lang})",
        caption_lang=caption_lang,
    )
    print(f"    ✅ Task started, id: {task_id}")
    print("    Polling (μπορεί να πάρει 2-6 λεπτά)...")

    def progress(elapsed, status, percent):
        print(f"      ...{elapsed}s — status={status} ({percent}%)")

    video_url = wait_for_video(task_id, api_key, on_progress=progress)
    print(f"\n    🎬 ΕΤΟΙΜΟ [{lang}]: {video_url}")
    return video_url


if __name__ == "__main__":
    API_KEY = os.environ.get("A2E_API_KEY", "").strip()
    if not API_KEY:
        print("❌ Δεν βρέθηκε A2E_API_KEY. Τρέξε πρώτα:")
        print('   export A2E_API_KEY="sk_..."')
        sys.exit(1)

    if shutil.which("ffmpeg") is None:
        print("❌ Δεν βρέθηκε ffmpeg στο PATH.")
        print("   Mac:    brew install ffmpeg")
        print("   Ubuntu: sudo apt install ffmpeg")
        sys.exit(1)

    print("=" * 60)
    print("Asklepios × A2E.ai — Δημιουργία Intro Video")
    print("=" * 60)

    print("\n[1] Λίστα διαθέσιμων avatars...")
    try:
        avatars = list_avatars(API_KEY)
        items = (avatars.get("data") or {}).get("data") or avatars.get("data") or []
        if isinstance(items, list) and items:
            print(f"    Βρέθηκαν {len(items)} avatars. Πρώτα 5:")
            for a in items[:5]:
                print(f"      _id={a.get('_id')}  name={a.get('name', '—')}  gender={a.get('gender','—')}")
        else:
            print("    ⚠️ Δεν βρέθηκαν avatars στη λίστα — έλεγξε το raw response:")
            print("   ", json.dumps(avatars)[:500])
    except A2EError as e:
        print("    ❌", e)
        sys.exit(1)

    chosen = os.environ.get("A2E_ANCHOR_ID", "").strip()
    if not chosen:
        print("\n❌ Διάλεξε ένα anchor_id από τη λίστα πάνω και τρέξε ξανά με:")
        print('   export A2E_ANCHOR_ID="<το _id που διάλεξες>"')
        print("   python a2e_intro_video.py")
        sys.exit(0)

    print(f"\nΧρησιμοποιείται avatar: {chosen}")

    work_dir = tempfile.mkdtemp(prefix="asklepios_intro_")
    try:
        url_el = build_intro_video("el", API_KEY, chosen, work_dir)
        url_en = build_intro_video("en", API_KEY, chosen, work_dir)
    except A2EError as e:
        print("\n❌", e)
        sys.exit(1)

    print("\n" + "=" * 60)
    print("ΕΤΟΙΜΟ — πρόσθεσε αυτά ως secrets στο Railway:")
    print("=" * 60)
    print(f'A2E_INTRO_VIDEO_URL_EL = "{url_el}"')
    print(f'A2E_INTRO_VIDEO_URL_EN = "{url_en}"')
