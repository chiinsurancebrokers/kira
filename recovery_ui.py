"""
recovery_ui.py — Streamlit screen for "Αποκατάσταση στο σπίτι" (home monitoring after discharge).

kira_app.py is a script (run as __main__), so this module cannot import it.  Everything it
needs from the app is handed in through `deps` (a SimpleNamespace):

    deps.lang            "el" | "en"
    deps.email           signed-in user's e-mail ("" when auth is off)
    deps.supabase()      -> client | None
    deps.fernet()        -> Fernet
    deps.api_key()       -> Claude key
    deps.log(...)        -> log_event
    deps.rate_gate(name) -> bool  (shows its own warning when blocked)
    deps.heic / deps.downscale   image helpers for the letter extraction
    deps.header(...)     -> render_doc_header
Storage: tables recovery_plans / recovery_entries (see supabase/recovery_schema.sql), Fernet-encrypted
like `drafts`.  If the tables are missing the screen still works for the session and says so.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, date, timedelta

import streamlit as st

import recovery as R

KEY = "_rec"


# ── tiny helpers ──────────────────────────────────────────────────────────────
def _t(d, el, en):
    return el if d.lang == "el" else en


def _state():
    if KEY not in st.session_state:
        st.session_state[KEY] = {"plans": [], "entries": {}, "active": None, "loaded": False,
                                 "persist_ok": True, "just_saved": None, "flash": None}
    return st.session_state[KEY]


# ── storage ───────────────────────────────────────────────────────────────────
def _enc(d, obj):
    return d.fernet().encrypt(json.dumps(obj, ensure_ascii=False).encode()).decode()


def _dec(d, blob):
    return json.loads(d.fernet().decrypt(blob.encode()).decode())


def _db_load(d, S):
    S["loaded"] = True
    sb = d.supabase()
    if not sb or not d.email:
        S["persist_ok"] = False
        return
    t0 = time.time()
    try:
        rows = sb.table("recovery_plans").select("plan_id,data").eq("user_email", d.email).execute().data or []
        plans = []
        for r in rows:
            try: plans.append(_dec(d, r["data"]))
            except Exception: continue
        S["plans"] = plans
        for p in plans:
            er = sb.table("recovery_entries").select("entry_id,data").eq("user_email", d.email) \
                   .eq("plan_id", p["id"]).execute().data or []
            es = []
            for r in er:
                try: es.append(_dec(d, r["data"]))
                except Exception: continue
            S["entries"][p["id"]] = R.sort_entries(es)
        if plans and not S["active"]:
            S["active"] = plans[0]["id"]
        S["persist_ok"] = True
        d.log("supabase_read", ok=True, ms=(time.time() - t0) * 1000, table="recovery")
    except Exception as e:  # tables not created yet, offline, …
        S["persist_ok"] = False
        d.log("supabase_read", ok=False, ms=(time.time() - t0) * 1000, error=str(e)[:200], table="recovery")


def _db_save_plan(d, S, plan):
    sb = d.supabase()
    if not sb or not d.email or not S["persist_ok"]:
        return
    try:
        sb.table("recovery_plans").upsert({"user_email": d.email, "plan_id": plan["id"], "data": _enc(d, plan),
                                           "updated_at": datetime.utcnow().isoformat()},
                                          on_conflict="user_email,plan_id").execute()
    except Exception as e:
        S["persist_ok"] = False
        d.log("supabase_write", ok=False, error=str(e)[:200], table="recovery_plans")


def _db_save_entry(d, S, plan_id, entry):
    sb = d.supabase()
    if not sb or not d.email or not S["persist_ok"]:
        return
    try:
        sb.table("recovery_entries").upsert({"user_email": d.email, "plan_id": plan_id, "entry_id": entry["id"],
                                             "ts": entry["ts"], "data": _enc(d, entry)},
                                            on_conflict="user_email,plan_id,entry_id").execute()
    except Exception as e:
        S["persist_ok"] = False
        d.log("supabase_write", ok=False, error=str(e)[:200], table="recovery_entries")


def _db_delete(d, S, plan_id=None, entry_id=None):
    sb = d.supabase()
    if not sb or not d.email or not S["persist_ok"]:
        return
    try:
        if entry_id:
            sb.table("recovery_entries").delete().eq("user_email", d.email).eq("plan_id", plan_id).eq("entry_id", entry_id).execute()
        else:
            sb.table("recovery_entries").delete().eq("user_email", d.email).eq("plan_id", plan_id).execute()
            sb.table("recovery_plans").delete().eq("user_email", d.email).eq("plan_id", plan_id).execute()
    except Exception as e:
        d.log("supabase_delete", ok=False, error=str(e)[:200], table="recovery")


def delete_all_for_user(sb, email):
    """Called from the GDPR erasure page."""
    for tb in ("recovery_entries", "recovery_plans"):
        try: sb.table(tb).delete().eq("user_email", email).execute()
        except Exception: pass


def _mail_cfg(d):
    sec = getattr(d, "secret", None) or (lambda k, dflt="": "")
    return {"api_key": sec("RESEND_API_KEY", ""), "from": sec("RESEND_FROM", ""),
            "app_url": sec("APP_URL", ""), "owner": d.email or ""}


# ── presentation helpers ──────────────────────────────────────────────────────
CSS = """
<style>
.rc-banner{border-radius:18px;padding:18px 20px;margin:6px 0 14px;color:#fff}
.rc-banner h2{margin:0 0 4px;font-size:21px;color:#fff;letter-spacing:-.01em}
.rc-banner p{margin:4px 0 0;font-size:14px;line-height:1.5;color:#fff}
.rc-pill{display:inline-block;background:rgba(255,255,255,.22);border-radius:99px;padding:2px 10px;font-size:12px;font-weight:700;margin-right:6px}
.rc-alert{border-left:6px solid;border-radius:10px;padding:10px 14px;margin:8px 0;background:#fff;font-size:14px;line-height:1.5}
.rc-alert b{display:block;font-size:15px}
.rc-tel a{display:inline-block;background:#fff;color:#B91C1C !important;font-weight:800;border-radius:12px;padding:12px 22px;margin:10px 10px 0 0;text-decoration:none;font-size:18px}
.rc-chip{display:inline-block;background:#EEF1FB;border-radius:99px;padding:3px 11px;font-size:12.5px;font-weight:600;margin:2px 4px 2px 0;color:#0A1030}
.rc-vit{display:inline-block;min-width:92px;background:#fff;border:1px solid #E1E5F4;border-radius:12px;padding:8px 12px;margin:0 8px 8px 0;text-align:center}
.rc-vit .v{font-size:20px;font-weight:800;color:#0A1030}.rc-vit .l{font-size:11px;color:#5A6388;font-weight:700}
</style>
"""


def _banner(d, S, plan, entries):
    res = R.evaluate(plan, entries)
    lvl, lang = res["level"], d.lang
    col = R.LEVEL_COLOR[lvl]
    title = R.LEVEL_LABEL[lang][lvl]
    day = res["day"]
    intensive = day <= plan.get("intensive_days", 3)
    comp = R.compliance(plan, entries)
    sub = (f"{_t(d,'Ημέρα','Day')} {day} {_t(d,'από την έξοδο','since discharge')} · "
           f"{_t(d,'Καταγραφές σήμερα','Check-ins today')}: {comp['done']}/{comp['planned']}"
           + (f" · {_t(d,'εντατική φάση','intensive phase')}" if intensive else ""))
    body = ""
    note = ""
    if lvl >= R.CALL and res["alerts"]:
        body = res["alerts"][0].action[lang]
        if res["overdue"] and res["overdue"]["kind"] == "recheck":
            note = _t(d, " · Ξαναμετρήστε μέσα σε 1 ώρα.", " · Measure again within 1 hour.")
    elif not entries:
        body = _t(d, "Δεν υπάρχει ακόμη καταγραφή. Κάντε την πρώτη μέτρηση από την καρτέλα «Καταγραφή».",
                  "No readings yet. Take the first one in the “Check-in” tab.")
    elif res["overdue"]:
        k = res["overdue"]["kind"]
        body = {"recheck": _t(d, "Η τελευταία καταγραφή είχε ειδοποίηση και πέρασε πάνω από 1 ώρα — ξαναμετρήστε τώρα.",
                              "The last reading raised an alert and over an hour has passed — measure again now."),
                "missed": _t(d, f"Δεν υπάρχει καταγραφή εδώ και {res['overdue']['gap_h']} ώρες.",
                             f"No reading for {res['overdue']['gap_h']} hours."),
                "first": _t(d, "Δεν έχει γίνει ακόμη καμία καταγραφή σήμερα.", "No reading logged yet today.")}[k]
    elif lvl == R.OK:
        body = _t(d, "Η τελευταία καταγραφή είναι εντός των ορίων που έχουν οριστεί. Συνεχίστε τις προγραμματισμένες μετρήσεις.",
                  "The last reading is within the limits that were set. Keep up the scheduled checks.")
    st.markdown(CSS, unsafe_allow_html=True)
    tel = ('<div class="rc-tel"><a href="tel:166">📞 166</a><a href="tel:112">📞 112</a></div>' if lvl == R.EMERGENCY else "")
    st.markdown(f'<div class="rc-banner" style="background:{col}"><span class="rc-pill">{sub}</span>'
                f'<h2>{title}</h2><p>{body}{note}</p>{tel}</div>', unsafe_allow_html=True)
    for a in res["alerts"]:
        c = R.LEVEL_COLOR[a.level]
        det = f"<br><span style='color:#5A6388'>{a.detail[lang]}</span>" if a.detail[lang] else ""
        st.markdown(f'<div class="rc-alert" style="border-color:{c}"><b>{a.title[lang]}</b>{R.LEVEL_LABEL[lang][a.level]}{det}</div>',
                    unsafe_allow_html=True)
    return res


def _last_vitals(d, entries):
    if not entries:
        return
    e = R.sort_entries(entries)[-1]
    cells = []
    if e.get("hr"): cells.append((e["hr"], _t(d, "Σφυγμοί", "Pulse")))
    if e.get("sbp"): cells.append((f'{e["sbp"]}/{e.get("dbp") or "?"}', _t(d, "Πίεση", "BP")))
    if e.get("spo2"): cells.append((f'{e["spo2"]}%', "SpO₂"))
    if e.get("temp"): cells.append((f'{e["temp"]}°', _t(d, "Θερμ.", "Temp")))
    if e.get("weight"): cells.append((f'{e["weight"]} kg', _t(d, "Βάρος", "Weight")))
    when = R._dt(e["ts"]).astimezone(R.TZ).strftime("%d/%m %H:%M")
    st.caption(f'{_t(d,"Τελευταία καταγραφή","Last reading")}: {when}' + (f' · {e["by"]}' if e.get("by") else ""))
    st.markdown("".join(f'<div class="rc-vit"><div class="v">{v}</div><div class="l">{l}</div></div>' for v, l in cells),
                unsafe_allow_html=True)
    if e.get("symptoms"):
        st.markdown("".join(f'<span class="rc-chip">{R.SYMPTOMS[k][2 if d.lang=="el" else 3]}</span>' for k in e["symptoms"]),
                    unsafe_allow_html=True)


# ── tabs ──────────────────────────────────────────────────────────────────────
def _tab_checkin(d, S, plan, entries):
    lang = d.lang
    st.markdown(f"**{_t(d,'Νέα καταγραφή','New check-in')}** — "
                + _t(d, "όλα τα πεδία είναι προαιρετικά· γράψτε ό,τι μετρήσατε.", "every field is optional — enter what you measured."))
    with st.form("rc_form", clear_on_submit=True):
        c = st.columns(3)
        hr = c[0].number_input(_t(d, "Σφυγμοί (/λεπτό)", "Pulse (/min)"), 0, 250, value=None, step=1)
        sbp = c[1].number_input(_t(d, "Συστολική (πάνω)", "Systolic (upper)"), 0, 300, value=None, step=1)
        dbp = c[2].number_input(_t(d, "Διαστολική (κάτω)", "Diastolic (lower)"), 0, 200, value=None, step=1)
        c = st.columns(3)
        spo2 = c[0].number_input("SpO₂ (%)", 0, 100, value=None, step=1)
        temp = c[1].number_input(_t(d, "Θερμοκρασία (°C)", "Temperature (°C)"), 0.0, 45.0, value=None, step=0.1, format="%.1f")
        wt = c[2].number_input(_t(d, "Βάρος πρωί (kg)", "Morning weight (kg)"), 0.0, 300.0, value=None, step=0.1, format="%.1f")
        with st.expander(_t(d, "Πίεση όρθια (αν σηκώθηκε)", "Standing BP (if she stood up)")):
            c = st.columns(2)
            ss = c[0].number_input(_t(d, "Συστολική όρθια", "Standing systolic"), 0, 300, value=None, step=1)
            sd = c[1].number_input(_t(d, "Διαστολική όρθια", "Standing diastolic"), 0, 200, value=None, step=1)

        st.markdown(f"**{_t(d,'Συμπτώματα τώρα','Symptoms right now')}**")
        picked = []
        by_group = {}
        for k, v in R.SYMPTOMS.items():
            by_group.setdefault(v[1], []).append(k)
        for gkey, gel, gen in R.SYMPTOM_GROUPS:
            keys = by_group.get(gkey, [])
            if not keys: continue
            has_em = any(R.SYMPTOMS[k][0] == R.EMERGENCY for k in keys)
            with st.expander(("🚨 " if has_em else "") + _t(d, gel, gen), expanded=has_em and gkey in ("heart", "breath")):
                for k in keys:
                    if st.checkbox(R.SYMPTOMS[k][2 if lang == "el" else 3], key=f"rc_s_{k}"):
                        picked.append(k)
        names = [m["name"] for m in plan.get("meds", []) if m.get("name")]
        missed = st.multiselect(_t(d, "Δόσεις που ΔΕΝ ελήφθησαν από την προηγούμενη καταγραφή", "Doses NOT taken since the last check-in"),
                                names) if names else []
        notes = st.text_input(_t(d, "Σημείωση (προαιρετικά)", "Note (optional)"))
        by = st.text_input(_t(d, "Ποιος κατέγραψε;", "Who is recording?"), value=S.get("by", ""))
        ok = st.form_submit_button(_t(d, "💾 Αποθήκευση & έλεγχος", "💾 Save & check"), type="primary", use_container_width=True)
    if ok:
        e = R.new_entry(hr=hr, sbp=sbp, dbp=dbp, spo2=spo2, temp=temp, weight=wt, sbp_stand=ss, dbp_stand=sd,
                        symptoms=picked, meds_missed=missed, notes=notes.strip(), by=by.strip())
        has_data = any(e.get(k) is not None for k in ("hr", "sbp", "spo2", "temp", "weight")) or e["symptoms"] or e["meds_missed"]
        bad = R.validate_entry(e)
        if not has_data:
            st.warning(_t(d, "Δεν δόθηκε καμία τιμή ή σύμπτωμα.", "Nothing was entered."))
        elif bad:
            for _f, en, el in bad:
                st.error(_t(d, el, en))
        else:
            S["by"] = by.strip()
            S["entries"].setdefault(plan["id"], []).append(e)
            S["entries"][plan["id"]] = R.sort_entries(S["entries"][plan["id"]])
            _db_save_entry(d, S, plan["id"], e)
            d.log("recovery_checkin", ok=True, level=R.evaluate(plan, S["entries"][plan["id"]])["level"])
            S["flash"] = True
            nres = R.notify_if_needed(plan, S["entries"][plan["id"]], _mail_cfg(d), log=d.log)
            if nres["reason"] in ("sent", "error", "not_configured"):
                S["flash_notify"] = nres
                if nres["sent"]:
                    _db_save_plan(d, S, plan)
            for k in R.SYMPTOMS:
                st.session_state.pop(f"rc_s_{k}", None)
            st.rerun()
    # remove a mistaken last entry
    if entries:
        with st.expander(_t(d, "Διόρθωση λάθους καταγραφής", "Fix a wrong entry")):
            opts = {f'{R._fmt_ts(e["ts"])} · {e.get("hr") or "-"} / {e.get("sbp") or "-"}/{e.get("dbp") or "-"}': e["id"]
                    for e in reversed(entries[-10:])}
            sel = st.selectbox(_t(d, "Καταγραφή", "Entry"), list(opts.keys()), key="rc_del_sel")
            if st.button(_t(d, "🗑 Διαγραφή αυτής της καταγραφής", "🗑 Delete this entry"), key="rc_del_btn"):
                eid = opts[sel]
                S["entries"][plan["id"]] = [e for e in entries if e["id"] != eid]
                _db_delete(d, S, plan["id"], eid)
                st.rerun()


def _tab_trends(d, S, plan, entries):
    import pandas as pd
    if len(entries) < 2:
        st.info(_t(d, "Χρειάζονται τουλάχιστον 2 καταγραφές για γραφήματα.", "At least 2 readings are needed for charts."))
    df = pd.DataFrame(R.series(entries))
    if df.empty:
        return
    df = df.set_index("ts")
    th = plan["thresholds"]
    blocks = [(_t(d, "Σφυγμοί", "Pulse"), ["hr"]), (_t(d, "Πίεση", "Blood pressure"), ["sbp", "dbp"]),
              ("SpO₂", ["spo2"]), (_t(d, "Βάρος (kg)", "Weight (kg)"), ["weight"]), (_t(d, "Θερμοκρασία", "Temperature"), ["temp"])]
    import altair as alt
    rules = {"hr": [(th["hr_low_call"], "#EA580C"), (th["hr_low_emergency"], "#DC2626"), (th["hr_high_call"], "#EA580C")],
             "sbp": [(th["sbp_low_call"], "#EA580C"), (th["sbp_low_emergency"], "#DC2626"), (th["sbp_high_call"], "#EA580C")],
             "spo2": [(th["spo2_call"], "#EA580C"), (th["spo2_emergency"], "#DC2626")],
             "temp": [(th["temp_call"], "#EA580C")], "weight": []}
    for title, cols in blocks:
        sub = df[cols].dropna(how="all")
        if sub.empty: continue
        st.markdown(f"**{title}**")
        long = sub.reset_index().melt("ts", var_name="series", value_name="v").dropna()
        lo, hi = float(long["v"].min()), float(long["v"].max())
        rl = [(v, c) for k in cols for (v, c) in rules.get(k, [])]
        vals = [lo, hi] + [v for v, _ in rl if lo - 25 <= v <= hi + 25]
        dom = [min(vals) - (max(vals) - min(vals)) * 0.08 - 1, max(vals) + (max(vals) - min(vals)) * 0.08 + 1]
        base = alt.Chart(long).encode(x=alt.X("ts:T", title=None),
                                      y=alt.Y("v:Q", title=None, scale=alt.Scale(domain=dom, zero=False)),
                                      color=alt.Color("series:N", legend=alt.Legend(orient="bottom", title=None)))
        layers = [base.mark_line(point=True)]
        for v, c in rl:
            if dom[0] <= v <= dom[1]:
                layers.append(alt.Chart(alt.Data(values=[{"y": v}])).mark_rule(strokeDash=[5, 4], color=c, opacity=.8).encode(y="y:Q"))
        st.altair_chart(alt.layer(*layers).properties(height=190), use_container_width=True)
    st.caption(_t(d, "Οι διακεκομμένες γραμμές είναι τα όρια ειδοποίησης (πορτοκαλί = επικοινωνία, κόκκινο = επείγον).",
                  "Dashed lines are the alert limits (orange = contact team, red = emergency)."))
    st.markdown(f"**{_t(d,'Πίνακας καταγραφών','Readings table')}**")
    rows = R.summary_rows(plan, entries, d.lang)
    head = [_t(d, "Ημ/νία", "When"), _t(d, "Σφυγμοί", "Pulse"), _t(d, "ΑΠ", "BP"), "SpO₂", _t(d, "Θερμ.", "Temp"),
            _t(d, "Βάρος", "Weight"), _t(d, "Συμπτώματα", "Symptoms"), _t(d, "Σημείωση", "Note")]
    st.dataframe(pd.DataFrame(rows[::-1], columns=head), use_container_width=True, hide_index=True)


def _tab_plan(d, S, plan, new=False):
    lang = d.lang
    pid = plan["id"]
    st.caption(_t(d, "Συμπληρώστε από το εξιτήριο. Μπορείτε να ανεβάσετε φωτογραφία/PDF του εξιτηρίου και ο Asklepios θα το αντιγράψει — εσείς ελέγχετε.",
                  "Fill in from the discharge papers. You can upload a photo/PDF of the discharge letter and Asklepios will copy it — you review."))
    with st.container(border=True):
        st.markdown(f"**📄 {_t(d,'Εξιτήριο (προαιρετικά)','Discharge letter (optional)')}**")
        fs = st.file_uploader(_t(d, "Φωτογραφία ή PDF του εξιτηρίου", "Photo or PDF of the discharge letter"),
                              type=["pdf", "jpg", "jpeg", "png", "heic", "webp"], accept_multiple_files=True, key=f"rc_up_{pid}")
        if fs and st.button(_t(d, "Διάβασε το εξιτήριο", "Read the letter"), key=f"rc_read_{pid}"):
            if d.rate_gate("recovery_extract"):
                with st.spinner(_t(d, "Διαβάζω το έγγραφο…", "Reading the document…")):
                    n_ok = 0
                    for f in fs:
                        r = R.extract_discharge(d.api_key(), f.getvalue(), f.type, f.name, d.heic, d.downscale, d.log)
                        if r["ok"]:
                            R.merge_letter(plan, r["data"]); n_ok += 1
                        else:
                            st.error(f"{f.name}: {r['error']}")
                if n_ok:
                    _db_save_plan(d, S, plan)
                    st.session_state.pop(f"rc_meds_{pid}", None)
                    S["flash_msg"] = _t(d, "Το εξιτήριο διαβάστηκε. Ελέγξτε προσεκτικά όλα τα πεδία παρακάτω.",
                                        "Letter read. Please review every field below carefully.")
                    st.rerun()
    if S.get("flash_msg"):
        st.success(S.pop("flash_msg"))

    with st.form(f"rc_plan_{pid}"):
        c = st.columns([2, 1, 1])
        name = c[0].text_input(_t(d, "Ονοματεπώνυμο ασθενούς", "Patient name"), plan["patient"].get("name", ""))
        age = c[1].number_input(_t(d, "Ηλικία", "Age"), 0, 120, value=plan["patient"].get("age") or None, step=1)
        sex = c[2].text_input(_t(d, "Φύλο", "Sex"), plan["patient"].get("sex", ""))
        c = st.columns([2, 1])
        proc_keys = list(R.PROCEDURES)
        proc = c[0].selectbox(_t(d, "Τύπος", "Type"), proc_keys, index=proc_keys.index(plan.get("procedure", "tavi")),
                              format_func=lambda k: R.PROCEDURES[k][lang])
        dd = c[1].date_input(_t(d, "Ημερομηνία εξόδου", "Discharge date"), date.fromisoformat(plan["discharge_date"]))
        course = st.text_area(_t(d, "Πορεία νοσηλείας (π.χ. ΜΕΘ, αναπνευστήρας, ασταθής πίεση)", "Hospital course (e.g. ICU, ventilator, unstable BP)"),
                              plan.get("hospital_course", ""), height=80)
        b = plan.get("baseline", {})
        st.markdown(f"**{_t(d,'Γνωστές βασικές τιμές (αν υπάρχουν)','Known baseline values (if any)')}**")
        c = st.columns(4)
        bh = c[0].number_input(_t(d, "Συνήθης σφυγμός", "Usual pulse"), 0, 200, value=b.get("hr_usual"), step=1)
        bm = c[1].number_input(_t(d, "Ελάχιστος γνωστός σφυγμός (Holter)", "Lowest known pulse (Holter)"), 0, 200, value=b.get("hr_min_known"), step=1)
        bs = c[2].number_input(_t(d, "Συνήθης συστολική", "Usual systolic"), 0, 250, value=b.get("sbp_usual"), step=1)
        bo = c[3].number_input(_t(d, "Συνήθης SpO₂", "Usual SpO₂"), 0, 100, value=b.get("spo2_usual"), step=1)
        cpd = st.slider(_t(d, "Καταγραφές ανά ημέρα στην εντατική φάση", "Check-ins per day in the intensive phase"), 1, 6, plan.get("checkins_per_day", 3))
        idays = st.slider(_t(d, "Διάρκεια εντατικής φάσης (ημέρες)", "Intensive phase length (days)"), 1, 14, plan.get("intensive_days", 3))
        restr = st.text_area(_t(d, "Περιορισμοί / οδηγίες (μία ανά γραμμή)", "Restrictions / instructions (one per line)"),
                             "\n".join(plan.get("restrictions", [])), height=90)
        flags = st.text_area(_t(d, "Προειδοποιητικά σημεία που γράφει το εξιτήριο (μία ανά γραμμή)", "Warning signs listed in the discharge letter (one per line)"),
                             "\n".join(plan.get("letter_red_flags", [])), height=80)
        saved = st.form_submit_button(_t(d, "💾 Αποθήκευση σχεδίου", "💾 Save plan"), type="primary", use_container_width=True)

    st.markdown(f"**💊 {_t(d,'Φάρμακα','Medication')}**")
    import pandas as pd
    mdf = pd.DataFrame([{"name": m.get("name", ""), "dose": m.get("dose", ""), "times": ", ".join(m.get("times", [])),
                         "critical": bool(m.get("critical")), "note": m.get("note", "")} for m in plan.get("meds", [])]
                       or [{"name": "", "dose": "", "times": "", "critical": False, "note": ""}])
    med_ed = st.data_editor(mdf, num_rows="dynamic", use_container_width=True, hide_index=True, key=f"rc_meds_{pid}",
                            column_config={"name": _t(d, "Φάρμακο", "Medication"), "dose": _t(d, "Δόση", "Dose"),
                                           "times": _t(d, "Ώρες", "Times"), "critical": st.column_config.CheckboxColumn(_t(d, "Κρίσιμο", "Critical")),
                                           "note": _t(d, "Σημείωση", "Note")})
    if d.dossier_meds and st.button(_t(d, "↧ Εισαγωγή φαρμάκων από τον Φάκελο εξετάσεων", "↧ Import medication from the exam dossier"), key=f"rc_imp_{pid}"):
        have = {m["name"].lower() for m in plan["meds"]}
        for m in d.dossier_meds():
            if m.get("drug") and m["drug"].lower() not in have:
                plan["meds"].append({"name": m["drug"], "dose": m.get("dose", ""), "times": [m["time"]] if m.get("time") else [],
                                     "critical": R.is_critical_med(m["drug"]), "note": ""})
        _db_save_plan(d, S, plan); st.session_state.pop(f"rc_meds_{pid}", None); st.rerun()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**📞 {_t(d,'Επαφές ομάδας','Care-team contacts')}**")
        cdf = pd.DataFrame(plan.get("contacts") or [{"label": "", "phone": ""}])
        ced = st.data_editor(cdf, num_rows="dynamic", hide_index=True, use_container_width=True, key=f"rc_ct_{pid}",
                             column_config={"label": _t(d, "Ποιος", "Who"), "phone": _t(d, "Τηλέφωνο", "Phone")})
    with c2:
        st.markdown(f"**📅 {_t(d,'Επόμενα ραντεβού','Follow-up appointments')}**")
        fdf = pd.DataFrame(plan.get("followups") or [{"label": "", "date": "", "place": ""}])
        fed = st.data_editor(fdf, num_rows="dynamic", hide_index=True, use_container_width=True, key=f"rc_fu_{pid}",
                             column_config={"label": _t(d, "Τι", "What"), "date": _t(d, "Ημ/νία (ΕΕΕΕ-ΜΜ-ΗΗ)", "Date (YYYY-MM-DD)"), "place": _t(d, "Πού", "Where")})

    if saved:
        plan["patient"] = {"name": name.strip(), "age": int(age) if age else None, "sex": sex.strip()}
        plan["procedure"] = proc
        plan["procedure_label"] = R.PROCEDURES[proc]["en"] if proc != "other" or not plan.get("procedure_label") else plan["procedure_label"]
        plan["discharge_date"] = dd.isoformat()
        plan["hospital_course"] = course.strip()
        plan["baseline"] = {"hr_usual": bh, "hr_min_known": bm, "sbp_usual": bs, "spo2_usual": bo}
        plan["checkins_per_day"], plan["intensive_days"] = cpd, idays
        plan["restrictions"] = [x.strip() for x in restr.splitlines() if x.strip()]
        plan["letter_red_flags"] = [x.strip() for x in flags.splitlines() if x.strip()]
        plan["meds"] = [{"name": str(r["name"]).strip(), "dose": str(r["dose"] or "").strip(),
                         "times": [x.strip() for x in str(r["times"] or "").split(",") if x.strip()],
                         "critical": bool(r["critical"]) or R.is_critical_med(str(r["name"])), "note": str(r["note"] or "")}
                        for _, r in med_ed.iterrows() if str(r["name"]).strip() not in ("", "nan", "None")]
        plan["contacts"] = [{"label": str(r["label"] or ""), "phone": str(r["phone"] or "")}
                            for _, r in ced.iterrows() if str(r["phone"]).strip() not in ("", "nan", "None")]
        plan["followups"] = [{"label": str(r["label"] or ""), "date": str(r["date"] or ""), "place": str(r["place"] or "")}
                             for _, r in fed.iterrows() if str(r["label"]).strip() not in ("", "nan", "None")]
        _db_save_plan(d, S, plan)
        S["flash_msg"] = _t(d, "Το σχέδιο αποθηκεύτηκε.", "Plan saved.")
        st.rerun()

    # family e-mail alerts
    _notify_section(d, S, plan)

    # thresholds
    st.markdown(f"**🎚 {_t(d,'Όρια ειδοποίησης','Alert limits')}**")
    if plan.get("thresholds_source") != "clinician":
        st.warning(_t(d, "Τα όρια είναι γενικά προεπιλεγμένα και ΔΕΝ έχουν επιβεβαιωθεί από τον γιατρό. Ζητήστε από τον καρδιολόγο τα δικά της όρια και καταχωρήστε τα εδώ.",
                      "These limits are generic defaults and have NOT been confirmed by the doctor. Ask the cardiologist for her own limits and enter them here."))
    for h in R.baseline_hints(plan, lang):
        st.error(h)
    with st.expander(_t(d, "Προβολή / αλλαγή ορίων", "View / change limits")):
        with st.form(f"rc_th_{pid}"):
            vals = {}
            cols = st.columns(2)
            for i, k in enumerate(R.DEFAULT_THRESHOLDS):
                lo, hi = R.THRESHOLD_BOUNDS[k]
                lab = R.THRESHOLD_LABELS[k]
                step = 0.1 if isinstance(R.DEFAULT_THRESHOLDS[k], float) else 1
                vals[k] = cols[i % 2].number_input(f"{lab[0] if lang=='el' else lab[1]} ({lab[2]})", float(lo), float(hi),
                                                  float(plan["thresholds"][k]), step=float(step), key=f"rc_thv_{pid}_{k}")
            conf = st.checkbox(_t(d, "Αυτά τα όρια μου τα έδωσε ο θεράπων γιατρός", "These limits were given to me by the treating doctor"),
                               value=plan.get("thresholds_source") == "clinician")
            note = st.text_input(_t(d, "Από ποιον / πότε", "From whom / when"), plan.get("thresholds_note", ""))
            go = st.form_submit_button(_t(d, "Αποθήκευση ορίων", "Save limits"))
        if go:
            bad = R.thresholds_conflicts(vals)
            if bad:
                st.error(_t(d, "Λογικό σφάλμα: ", "Logic error: ") + ", ".join(bad))
            else:
                plan["thresholds"] = {k: (int(v) if isinstance(R.DEFAULT_THRESHOLDS[k], int) else round(v, 1)) for k, v in vals.items()}
                plan["thresholds_source"] = "clinician" if conf else "default"
                plan["thresholds_note"] = note
                _db_save_plan(d, S, plan); st.rerun()
        if st.button(_t(d, "↺ Επαναφορά προεπιλογών", "↺ Reset to defaults"), key=f"rc_thr_reset_{pid}"):
            plan["thresholds"] = dict(R.DEFAULT_THRESHOLDS); plan["thresholds_source"] = "default"
            for k in R.DEFAULT_THRESHOLDS: st.session_state.pop(f"rc_thv_{pid}_{k}", None)
            _db_save_plan(d, S, plan); st.rerun()

    if st.button("➕ " + _t(d, "Προσθήκη άλλου ασθενούς", "Add another patient"), key="rc_add_plan"):
        p = R.new_plan("", None, "", "tavi"); S["plans"].append(p); S["entries"][p["id"]] = []; S["active"] = p["id"]
        _db_save_plan(d, S, p); st.rerun()
    with st.expander("🗑 " + _t(d, "Διαγραφή αυτού του σχεδίου και όλων των καταγραφών", "Delete this plan and all its entries")):
        if st.checkbox(_t(d, "Ναι, να διαγραφούν οριστικά", "Yes, delete permanently"), key=f"rc_delok_{pid}") and \
           st.button(_t(d, "Διαγραφή", "Delete"), key=f"rc_delbtn_{pid}"):
            _db_delete(d, S, pid)
            S["plans"] = [p for p in S["plans"] if p["id"] != pid]; S["entries"].pop(pid, None)
            S["active"] = S["plans"][0]["id"] if S["plans"] else None
            st.rerun()


def _notify_section(d, S, plan):
    pid, lang = plan["id"], d.lang
    n = plan.setdefault("notify", dict(R.NOTIFY_DEFAULT, emails=[]))
    cfg = _mail_cfg(d)
    st.markdown(f"**📧 {_t(d,'Ειδοποίηση οικογένειας με email','Family e-mail alerts')}**")
    if not (cfg["api_key"] and cfg["from"]):
        st.caption(_t(d, "Η υπηρεσία email δεν έχει ρυθμιστεί στον διακομιστή (RESEND_API_KEY, RESEND_FROM).",
                      "E-mail service is not configured on the server (RESEND_API_KEY, RESEND_FROM)."))
    if not d.email:
        st.caption(_t(d, "Συνδεθείτε για να στέλνονται ειδοποιήσεις.", "Sign in to enable alert e-mails."))
    with st.form(f"rc_nf_{pid}"):
        txt = st.text_area(_t(d, f"Email φροντιστών (έως {R.MAX_NOTIFY_EMAILS}, ένα ανά γραμμή)", f"Carer e-mails (up to {R.MAX_NOTIFY_EMAILS}, one per line)"),
                           "\n".join(n.get("emails", [])), height=80)
        c = st.columns(2)
        lvl = c[0].selectbox(_t(d, "Πότε να στέλνεται", "Send when"), [R.CALL, R.EMERGENCY], index=0 if n.get("min_level", R.CALL) == R.CALL else 1,
                             format_func=lambda v: _t(d, "Επικοινωνία με ομάδα ή επείγον", "Contact team or emergency") if v == R.CALL
                             else _t(d, "Μόνο επείγον", "Emergency only"))
        lg = c[1].selectbox(_t(d, "Γλώσσα email", "E-mail language"), ["el", "en"], index=0 if n.get("lang", "el") == "el" else 1,
                            format_func=lambda v: "Ελληνικά" if v == "el" else "English")
        det = st.checkbox(_t(d, "Να περιλαμβάνονται οι τιμές και το όνομα (το email δεν είναι ασφαλές κανάλι· αν το αποεπιλέξετε, στέλνεται μόνο «υπάρχει ειδοποίηση»)",
                             "Include name and readings (e-mail is not a secure channel; if unticked only “there is an alert” is sent)"),
                          value=n.get("detail", True))
        ok = st.form_submit_button(_t(d, "Αποθήκευση παραληπτών", "Save recipients"))
    if ok:
        good, bad = R.parse_emails(txt)
        if bad:
            st.error(_t(d, "Μη έγκυρα ή επιπλέον email: ", "Invalid or extra e-mails: ") + ", ".join(bad))
        else:
            n.update({"emails": good, "min_level": lvl, "lang": lg, "detail": det})
            _db_save_plan(d, S, plan)
            S["flash_msg"] = _t(d, "Οι παραλήπτες αποθηκεύτηκαν.", "Recipients saved."); st.rerun()
    if n.get("emails") and cfg["api_key"] and d.email:
        import time as _time
        if st.button(_t(d, "✉️ Αποστολή δοκιμαστικού email", "✉️ Send a test e-mail"), key=f"rc_nt_{pid}"):
            if _time.time() - S.get("_last_test", 0) < 600:
                st.warning(_t(d, "Περιμένετε λίγα λεπτά πριν από νέα δοκιμή.", "Wait a few minutes before another test."))
            else:
                S["_last_test"] = _time.time()
                e = R.new_entry()
                subj, text, html = R.build_alert_email(plan, e, [], n.get("lang", "el"), cfg["app_url"], cfg["owner"], test=True)
                r = R.send_resend(cfg["api_key"], cfg["from"], n["emails"], subj, text, html)
                d.log("recovery_notify_test", ok=r["ok"], error=r["error"])
                (st.success(_t(d, "Στάλθηκε.", "Sent.")) if r["ok"] else st.error(_t(d, "Αποτυχία αποστολής.", "Sending failed.")))


def _tab_export(d, S, plan, entries):
    if not entries:
        st.info(_t(d, "Δεν υπάρχουν ακόμη καταγραφές.", "No readings yet.")); return
    st.caption(_t(d, "Ένα καθαρό έγγραφο με όλες τις μετρήσεις, τις ειδοποιήσεις και τα όρια — για το επόμενο ραντεβού ή για την ομάδα. Δεν περιέχει ερμηνεία.",
                  "One clean document with all readings, alerts and limits — for the next appointment or the care team. No interpretation."))
    nm = (plan["patient"].get("name") or "patient").replace(" ", "_")
    c = st.columns(2)
    c[0].download_button("⬇ Word (.docx)", R.build_docx(plan, entries, d.lang),
                         file_name=f"home_monitoring_{nm}_{date.today().isoformat()}.docx",
                         mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)
    c[1].download_button("⬇ " + _t(d, "Εκτυπώσιμο HTML (→ PDF)", "Printable HTML (→ PDF)"), R.build_html(plan, entries, d.lang).encode("utf-8"),
                         file_name=f"home_monitoring_{nm}_{date.today().isoformat()}.html", mime="text/html", use_container_width=True)


# ── entry point ───────────────────────────────────────────────────────────────
def render(deps):
    d = deps
    S = _state()
    if not S["loaded"]:
        _db_load(d, S)
    d.header("Αποκατάσταση στο σπίτι", "Recovery at home", icon="🏡",
             sub_el="Παρακολούθηση μετά την έξοδο από το νοσοκομείο — καταγραφή, ειδοποιήσεις, αναφορά για τον γιατρό",
             sub_en="Monitoring after hospital discharge — log, alerts, report for the doctor")

    plans = S["plans"]
    if not plans:
        st.error(_t(d, "Εργαλείο οργάνωσης και υπενθύμισης — δεν αντικαθιστά τον γιατρό και δεν κάνει διάγνωση. "
                       "Σε επείγον σύμπτωμα καλέστε 166 ή 112 χωρίς να περιμένετε την εφαρμογή.",
                    "An organising and reminder tool — it does not replace the doctor and does not diagnose. "
                    "In an emergency call 166 or 112 without waiting for the app."), icon="🚨")
        st.markdown(f"### {_t(d,'Ξεκινήστε ένα σχέδιο παρακολούθησης','Start a monitoring plan')}")
        for key in ("tavi", "icu_general", "other"):
            if st.button(R.PROCEDURES[key][d.lang], key=f"rc_new_{key}", use_container_width=True):
                prof = d.profile() or {}
                p = R.new_plan(prof.get("name", ""), prof.get("age"), prof.get("sex", ""), key)
                S["plans"].append(p); S["entries"][p["id"]] = []; S["active"] = p["id"]
                _db_save_plan(d, S, p); st.rerun()
        return

    ids = [p["id"] for p in plans]
    if S["active"] not in ids:
        S["active"] = ids[0]
    plan = next(p for p in plans if p["id"] == S["active"])
    entries = S["entries"].setdefault(plan["id"], [])

    if S.pop("flash", None):
        st.success(_t(d, "Καταγράφηκε. Δείτε το αποτέλεσμα παρακάτω.", "Saved. See the result below."))
    fn = S.pop("flash_notify", None)
    if fn:
        if fn["sent"]:
            st.info(_t(d, f"📧 Στάλθηκε ειδοποίηση σε: {', '.join(fn['to'])}", f"📧 Alert e-mailed to: {', '.join(fn['to'])}"))
        elif fn["reason"] == "not_configured":
            st.warning(_t(d, "📧 Έχουν οριστεί παραλήπτες αλλά η υπηρεσία email δεν έχει ρυθμιστεί στον διακομιστή — δεν στάλθηκε τίποτα. Καλέστε εσείς.",
                          "📧 Recipients are set but the e-mail service is not configured on the server — nothing was sent. Please phone them yourself."))
        else:
            st.error(_t(d, "📧 Η αποστολή email απέτυχε — ενημερώστε τηλεφωνικά τους φροντιστές.",
                        "📧 Sending the e-mail failed — please phone the carers."))
    _banner(d, S, plan, entries)
    _last_vitals(d, entries)
    if len(plans) > 1:
        S["active"] = st.selectbox(_t(d, "Ασθενής / σχέδιο", "Patient / plan"), ids, index=ids.index(S["active"]),
                                   format_func=lambda i: next((p["patient"].get("name") or _t(d, "(χωρίς όνομα)", "(unnamed)")) + " · " + p["discharge_date"] for p in plans if p["id"] == i))
    st.caption("🚨 " + _t(d, "Εργαλείο οργάνωσης και υπενθύμισης — δεν αντικαθιστά τον γιατρό και δεν κάνει διάγνωση. Σε επείγον σύμπτωμα καλέστε 166 ή 112 χωρίς να περιμένετε την εφαρμογή.",
                          "An organising and reminder tool — it does not replace the doctor and does not diagnose. In an emergency call 166 or 112 without waiting for the app."))
    if not S["persist_ok"]:
        st.warning(_t(d, "⚠️ Η αποθήκευση στον διακομιστή δεν είναι διαθέσιμη (πίνακες Supabase ή σύνδεση). Οι καταγραφές κρατιούνται μόνο όσο είναι ανοιχτή αυτή η σελίδα — κατεβάστε την αναφορά συχνά.",
                      "⚠️ Server storage is unavailable (Supabase tables or login). Entries are kept only while this page stays open — download the report often."))

    tabs = st.tabs([_t(d, "📝 Καταγραφή", "📝 Check-in"), _t(d, "📈 Τάσεις", "📈 Trends"),
                    _t(d, "⚙️ Σχέδιο & φάρμακα", "⚙️ Plan & meds"), _t(d, "📄 Αναφορά", "📄 Report")])
    with tabs[0]: _tab_checkin(d, S, plan, entries)
    with tabs[1]: _tab_trends(d, S, plan, entries)
    with tabs[2]: _tab_plan(d, S, plan)
    with tabs[3]: _tab_export(d, S, plan, entries)
