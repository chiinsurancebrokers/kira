# Source register — Asklepios (humans)

Rule for every rule and every lifestyle text: **a source URL (PubMed id/DOI or guideline page) and a clinician sign-off.**
Paraphrase and cite; never paste long passages. Check the licence before reuse, especially if you charge.

| Source | Use | Licence / note |
|---|---|---|
| MedlinePlus (US NLM), CDC | patient-level wording, symptoms | US public domain |
| NHS website content | patient-level wording | Open Government Licence (attribute) |
| WHO guidelines | rules, public health | CC BY-NC-SA IGO — non-commercial: **matters if you charge** |
| StatPearls (NCBI Bookshelf) | background | CC BY-NC — check before commercial use |
| PMC open-access, Cochrane abstracts | evidence for rules | free to read; cite, do not copy |
| AHRQ Emergency Severity Index handbook | triage logic | free |
| NICE | guidelines | **commercial reuse needs permission** — check first |
| Manchester Triage, NHS Pathways | — | licensed: **do not use** |
| Bone health: IOF/ESCEO, NOGG (UK FRAX), Endocrine Society | `lifestyle_info.py` | see PubMed ids in that file |

PubMed rule: whenever a PubMed result is used, name PubMed and keep the DOI link.

## Verified in PubMed (abstracts read)
- NOGG 2024 — PMID 40921943, doi 10.1007/s11657-025-01588-3 (scope: postmenopausal women, men 50+)
- BMJ 2026 calcium/vitamin D meta-analysis — PMID 42161415, doi 10.1136/bmj-2025-088050 (little to no fracture benefit in adults not at high risk)
- Exercise network meta-analysis 2026 — PMID 41617082, doi 10.1016/j.jshs.2026.101127 (postmenopausal women)
- ISCD 2019 monitoring executive summary — PMID 31400968, doi 10.1016/j.jocd.2019.07.001

## Still to source (hidden in `lifestyle_info.ITEMS` until a verified source is attached)
smoking and alcohol, falls prevention, dietary protein, and the clinician note that for premenopausal women
(this patient is 47) the T-score "osteopenia" label is not meant to be applied and the Z-score is used
(ISCD Official Positions: could not be matched in PubMed from here; verify on iscd.org).

## Switches
`LIFESTYLE_INFO=on` shows the section in the exam summary (default off). `REVIEWED_BY` in `lifestyle_info.py` is empty until a clinician signs the texts off; the report says so.
