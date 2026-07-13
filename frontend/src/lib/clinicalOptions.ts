// Standard ophthalmic presentation options for the Diagnosis / Medical History /
// Current Issue fields. These are *suggestions* only — every field stays free text,
// this just gives staff a fast pick-list of common TTSH Eye Centre presentations.
//
// The Current Issue options are deliberately phrased to match the red-flag and
// archetype keyword rules in backend/app/services/triage.py and pathway.py, so
// picking one from the list exercises the same triage/pathway logic a real
// free-text entry would.

export const DIAGNOSIS_OPTIONS = [
  "Primary open-angle glaucoma (POAG), both eyes",
  "Angle-closure glaucoma",
  "Age-related cataract",
  "Diabetic retinopathy",
  "Age-related macular degeneration (AMD)",
  "Retinal detachment",
  "Dry eye syndrome",
  "Bacterial conjunctivitis",
  "Corneal abrasion",
  "Uveitis",
  "Blepharitis",
  "Refractive error — myopia",
  "Refractive error — hyperopia / presbyopia",
  "Ocular hypertension",
  "Vitreous haemorrhage",
];

export const HISTORY_OPTIONS = [
  "Diabetic, on insulin",
  "Diabetic, on oral medication",
  "Hypertension, on medication",
  "Previous cataract surgery (right eye)",
  "Previous cataract surgery (left eye)",
  "Previous trabeculectomy (glaucoma surgery)",
  "Previous retinal laser treatment",
  "On anticoagulant / blood-thinner therapy",
  "Known drug allergy",
  "Family history of glaucoma",
  "Family history of macular degeneration",
  "Smoker",
  "No significant past ocular history",
];

export const CURRENT_ISSUE_OPTIONS = [
  // P1 — immediate, sight-threatening (see triage.py _RULES)
  "Sudden vision loss in one eye",
  "Chemical splash to the eye",
  "Eye trauma with foreign body sensation",
  "New flashes of light and floaters, curtain over vision",
  "Severe eye pain with nausea and halos",
  "Increasing pain and vision loss after recent eye surgery",
  // P2 — urgent, same-session escalation
  "New floaters since yesterday",
  "Double vision (diplopia)",
  "Red eye with discharge",
  "Eye pressure and halos around lights",
  "Vision has been getting worse over the past week",
  // Routine — maps to a specific visit pathway (see pathway.py _ARCHETYPE_RULES)
  "Blurred vision, needs new spectacle prescription",
  "Routine glaucoma review, raised IOP on last visit",
  "Diabetic retinopathy follow-up, blurred central vision",
  "Due for anti-VEGF injection",
  "Scheduled for YAG laser capsulotomy",
  "Cataract surgery workup, biometry needed",
  "Follow-up imaging and photos",
  "Routine annual eye check",
];
