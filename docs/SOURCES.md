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

## What was actually read (2026-10-25) — licence terms and findings
| Source | Read | Result that changes how we use it |
|---|---|---|
| MedlinePlus | terms page | Public domain EXCEPT A.D.A.M. encyclopedia and ASHP drug monographs (copyrighted). Credit "MedlinePlus, National Library of Medicine". No framing under another domain. |
| NHS website | terms page | OGL v3.0, commercial use allowed. If we ADAPT or translate: use OGL attribution and do NOT credit the NHS as source; we cannot charge users specifically for access to NHS content; logos, third-party images excluded. Linking to a page as a source is fine. |
| CDC | "Use of agency materials" | Mostly public domain; attribute to CDC, add a no-endorsement disclaimer, do not change the substance, no logos. Contractor/third-party material may be restricted. |
| WHO | copyright page | CC BY-NC-SA 3.0 IGO: non-commercial only; commercial use needs WHO permission; adaptations need a disclaimer and the same licence. |
| StatPearls | NCBI Bookshelf | CC BY-NC-ND 4.0: no commercial use AND no adaptation. Cite and link only; never paraphrase into the product. (Earlier note said NC only; corrected.) |
| NICE | UK Open Content Licence page | Commercial reuse allowed, UK use; self-assessed with a use form; outside the UK needs the international form; CKS, BNF, BNFc not covered. We are in Greece: use the international form before reusing. |
| Cochrane Library | permissions page | Abstracts/summaries reuse is for Cochrane groups; others go through RightsLink/Wiley. Cite with DOI; do not copy. |
| PMC open access | tools page | Licence varies per article: check each one before reuse. |
| AHRQ ESI handbook v4 (2012) | govinfo PDF | The algorithm is copyrighted by the ESI Triage Research Team LLC: usable unchanged in clinical practice, not for adapting. It is a hospital-nurse tool. **Do not embed or adapt.** |
| Endocrine Society | guidelines page | All rights reserved; licensing page exists. Not read further. |
| IOF website | page failed to load | Not read. |
| Kanis 2019 European guidance (ESCEO/IOF) | PDF on esceo.org | Calcium preferably from diet; vitamin D for those at increased risk; protein; weight-bearing exercise; falls prevention; smoking and 3+ alcohol units/day are risk factors; "osteopenia" is for epidemiological description, not a disease. PMID 30324412. Postmenopausal women. Now used in `lifestyle_info.py`. |
| NHS pages (pregnancy bleeding, suicidal thoughts, poisoning, eye injuries) | read | Now the source URL of rules PREGNANCY-DANGER, SUICIDE, OVERDOSE, EYE-CHEMICAL. |

### Differences between a rule and its source (for the clinician)
- PREGNANCY-DANGER: any "bleeding" in pregnancy gives EMERGENCY. The NHS page treats light spotting with little pain as "urgent help, contact the maternity unit" and only heavy bleeding, severe pain, dizziness or fainting as 999. The rule over-triages light spotting.
- EYE-CHEMICAL: now cites MedlinePlus "Eye emergencies" (flush at least 15 minutes, get help right away); the NHS eye page gives no number. Source changed from NHS to MedlinePlus (public domain).
- TESTICLE-PAIN: now cites MedlinePlus "Testicular torsion" (emergency room, most of the testicle saved if operated within 6 hours). An NHS Essex ICB page also read: torsion needs A&E/999, good salvage within 6 h, risk rises after 12 h, removal usually after 24 h.
- SUICIDE / OVERDOSE / EYE-CHEMICAL / PREGNANCY-DANGER cite UK pages that say 999; our messages say 112/166. Logic matches; numbers are Greek.
- Not read: MedlinePlus and CDC condition pages for the other rules (they already cite ESC/AHA/ERC/AAP papers).

### Retry (same day): what loaded this time
- IOF (osteoporosis.foundation): page loaded but gives only generic advice (bone-healthy diet, exercise, avoid negative lifestyle factors) and no reuse terms (© IOF; contact info@osteoporosis.foundation). Not used as a source; Kanis 2019 stays.
- CDC stroke and heart attack pages: read. Stroke: face, arm, speech, time; call 911 immediately. Heart attack: chest pain, arm/jaw pain, breathlessness, sweating, call 911. They agree with CHEST-ACS and STROKE-FAST (our numbers are 166/112). CDC is mostly public domain; attribute to CDC and add a no-endorsement line if text is reused.
- Europe PMC: blocked by robots.txt. PMC article pages: show a reCAPTCHA, so not read.
- AHRQ ESI page: still did not load (the handbook PDF on govinfo was read earlier).
- NHS testicular-torsion and testicular-pain pages: did not load.
