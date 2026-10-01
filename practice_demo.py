"""
practice_demo.py — made-up patients for the "Asklepios για ιατρεία" demo.

Every name, number and result here is FICTIONAL, written for the demo only.
No real patient data belongs in this file.
"""
from __future__ import annotations

import copy

DEMO_LETTERHEAD = {
    "practice": "Ιατρείο Επίδειξης",
    "doctor": "Δρ. Όνομα Επώνυμο",
    "specialty": "Παθολόγος",
    "address": "Οδός Παραδείγματος 1, Αθήνα",
    "phone": "210 000 0000",
}

_LAB = "Μικροβιολογικό Εργαστήριο (παράδειγμα)"
_COLS = ["Παράμετρος", "Τιμή", "Τιμές αναφοράς"]


def _exam(eid, title, etype, iso, facility, rows, conclusion=None, clinician=None, section="Αποτελέσματα"):
    y, m, d = iso.split("-")
    return {
        "id": eid, "exam_title": title, "exam_type": etype,
        "date": f"{d}/{m}/{y}", "date_iso": iso,
        "facility": facility, "clinician": clinician, "source_note": None,
        "sections": [{"title": section, "description": None, "columns": list(_COLS), "rows": rows}],
        "conclusion": [{"item": "", "detail": c} for c in (conclusion or [])],
    }


def _dossier(name, age, sex, amka, history, allergies, dx, meds, exams):
    return {
        "purpose": "",
        "patient": {"name": name, "amka": amka, "age": str(age), "sex": sex,
                    "history": history, "allergies": allergies, "dx": dx},
        "meds": [{"time": t, "drug": d, "dose": s} for t, d, s in meds],
        "exams": exams, "points": [], "symptoms": "",
        "done_files": [], "doc_names": {}, "errors": [],
        "docx": None, "html": None, "built_at": None,
        "demo": True,
    }


_PATIENTS = [
    {
        "pid": "demo-1",
        "dossier": _dossier(
            "Μαρία Δοκιμή", 68, "Γυναίκα", "ΔΟΚ-0001",
            "Υπέρταση, Σακχαρώδης διαβήτης τύπου 2", "Πενικιλλίνη", "Τακτικός έλεγχος",
            [("Πρωί", "Μετφορμίνη 850 mg", "1 δισκίο"), ("Πρωί", "Βαλσαρτάνη 80 mg", "1 δισκίο")],
            [
                _exam("d1a", "Γενική αίματος", "blood", "2026-09-12", _LAB, [
                    ["Αιμοσφαιρίνη (Hb)", "11,8 g/dL", "12,0–16,0"],
                    ["Αιματοκρίτης (Hct)", "36,1 %", "36–46"],
                    ["Λευκά αιμοσφαίρια", "6,9 ×10³/μL", "4,0–10,5"],
                    ["Αιμοπετάλια", "245 ×10³/μL", "150–400"],
                ]),
                _exam("d1b", "Βιοχημικές εξετάσεις", "blood", "2026-09-12", _LAB, [
                    ["Γλυκόζη νηστείας", "131 mg/dL", "70–100"],
                    ["HbA1c", "7,1 %", "< 5,7"],
                    ["Κρεατινίνη", "1,02 mg/dL", "0,5–1,1"],
                    ["Κάλιο (K)", "4,6 mmol/L", "3,5–5,1"],
                ]),
                _exam("d1c", "Υπερηχογράφημα καρδιάς", "imaging", "2026-06-02",
                      "Καρδιολογικό Ιατρείο (παράδειγμα)", [
                          ["Κλάσμα εξώθησης (EF)", "58 %", "≥ 55"],
                          ["Αριστερός κόλπος", "41 mm", "< 40"],
                      ], conclusion=["Ήπια διάταση αριστερού κόλπου. Φυσιολογική συστολική λειτουργία."],
                      section="Μετρήσεις"),
                _exam("d1d", "Γενική αίματος", "blood", "2026-03-10", _LAB, [
                    ["Αιμοσφαιρίνη (Hb)", "12,4 g/dL", "12,0–16,0"],
                    ["Αιματοκρίτης (Hct)", "37,9 %", "36–46"],
                    ["Λευκά αιμοσφαίρια", "7,2 ×10³/μL", "4,0–10,5"],
                    ["Αιμοπετάλια", "251 ×10³/μL", "150–400"],
                ]),
                _exam("d1e", "Βιοχημικές εξετάσεις", "blood", "2026-03-10", _LAB, [
                    ["Γλυκόζη νηστείας", "142 mg/dL", "70–100"],
                    ["HbA1c", "7,6 %", "< 5,7"],
                    ["Κρεατινίνη", "0,96 mg/dL", "0,5–1,1"],
                    ["Κάλιο (K)", "4,4 mmol/L", "3,5–5,1"],
                ]),
            ],
        ),
    },
    {
        "pid": "demo-2",
        "dossier": _dossier(
            "Γιώργος Παράδειγμα", 54, "Άνδρας", "ΔΟΚ-0002",
            "Δυσλιπιδαιμία", "—", "Έλεγχος λιπιδίων",
            [("Βράδυ", "Ατορβαστατίνη 20 mg", "1 δισκίο")],
            [
                _exam("d2a", "Λιπιδαιμικό προφίλ", "blood", "2026-08-28", _LAB, [
                    ["Ολική χοληστερόλη", "196 mg/dL", "< 200"],
                    ["LDL", "118 mg/dL", "< 130"],
                    ["HDL", "44 mg/dL", "> 40"],
                    ["Τριγλυκερίδια", "170 mg/dL", "< 150"],
                ]),
                _exam("d2b", "Ηλεκτροκαρδιογράφημα", "ecg", "2026-08-28",
                      "Καρδιολογικό Ιατρείο (παράδειγμα)", [
                          ["Ρυθμός", "Φλεβοκομβικός", ""],
                          ["Καρδιακή συχνότητα", "68 /λεπτό", ""],
                      ], conclusion=["Χωρίς ευρήματα οξείας ισχαιμίας."], section="Ευρήματα"),
                _exam("d2c", "Λιπιδαιμικό προφίλ", "blood", "2026-02-15", _LAB, [
                    ["Ολική χοληστερόλη", "241 mg/dL", "< 200"],
                    ["LDL", "162 mg/dL", "< 130"],
                    ["HDL", "41 mg/dL", "> 40"],
                    ["Τριγλυκερίδια", "190 mg/dL", "< 150"],
                ]),
            ],
        ),
    },
    {
        "pid": "demo-3",
        "dossier": _dossier(
            "Κατερίνα Υπόδειγμα", 37, "Γυναίκα", "ΔΟΚ-0003",
            "Θυρεοειδίτιδα Hashimoto", "—", "Παρακολούθηση θυρεοειδούς",
            [("Πρωί, νηστεία", "Λεβοθυροξίνη 75 μg", "1 δισκίο")],
            [
                _exam("d3a", "Θυρεοειδικές ορμόνες", "blood", "2026-09-05", _LAB, [
                    ["TSH", "2,1 μIU/mL", "0,4–4,0"],
                    ["FT4", "1,3 ng/dL", "0,9–1,7"],
                ]),
                _exam("d3b", "Θυρεοειδικές ορμόνες", "blood", "2026-05-20", _LAB, [
                    ["TSH", "5,8 μIU/mL", "0,4–4,0"],
                    ["FT4", "0,9 ng/dL", "0,9–1,7"],
                ]),
                _exam("d3c", "Υπερηχογράφημα θυρεοειδούς", "imaging", "2026-05-20",
                      "Ακτινολογικό Κέντρο (παράδειγμα)", [
                          ["Δεξιός λοβός", "15 × 17 × 44 mm", ""],
                          ["Αριστερός λοβός", "14 × 16 × 42 mm", ""],
                      ], conclusion=["Ανομοιογενής ηχοδομή. Δεν απεικονίζονται όζοι."], section="Μετρήσεις"),
            ],
        ),
    },
]


def demo_patients():
    """A fresh copy every call, so the demo can be reset."""
    return copy.deepcopy(_PATIENTS)


def demo_letterhead():
    return dict(DEMO_LETTERHEAD)
