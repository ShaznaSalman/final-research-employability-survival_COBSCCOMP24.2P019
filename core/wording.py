"""ALL user-facing wording lives here so it can be edited in one place.

Rules for every template:
  - ASSOCIATIONAL, never causal: "graduates with X tended to ...", never "doing X will ...".
  - No technical jargon (hazard, censoring, C-index, coefficient) - those belong in the About tab only.
Placeholders in {braces} are filled by core/explain.py, core/predict.py and core/whatif.py.
"""

# ---------------------------------------------------------------------------
# Plain-language feature descriptions
# ---------------------------------------------------------------------------
FEATURES = {
    "gpa": {"name": "GPA", "higher": "a higher GPA", "lower": "a lower GPA", "unit": "{v:.2f}",
            "step": 0.5, "step_text": "0.5 GPA points higher"},
    "internship_months": {"name": "internship experience", "higher": "more internship experience",
                          "lower": "less internship experience", "unit": "{v:g} months",
                          "step": 3, "step_text": "3 more months of internship"},
    "certifications_count": {"name": "professional certifications", "higher": "more certifications",
                             "lower": "fewer certifications", "unit": "{v:g}", "step": 1,
                             "step_text": "one more certification"},
    "project_count": {"name": "personal / academic projects", "higher": "more projects",
                      "lower": "fewer projects", "unit": "{v:g}", "step": 1, "step_text": "one more project"},
    "technical_mean": {"name": "technical skill ratings", "higher": "higher technical skill ratings",
                       "lower": "lower technical skill ratings", "unit": "{v:.1f} / 5", "step": 1,
                       "step_text": "technical skills rated 1 point higher (on the 1-5 scale)"},
    "soft_mean": {"name": "soft skill ratings", "higher": "higher soft skill ratings",
                  "lower": "lower soft skill ratings", "unit": "{v:.1f} / 5", "step": 1,
                  "step_text": "soft skills rated 1 point higher (on the 1-5 scale)"},
    # Transformed feature set: values are shown in their original units through "display".
    "gpa_3plus": {"name": "GPA of 3.0 or above", "higher": "a GPA of 3.0 or above", "lower": "a GPA below 3.0",
                  "unit": "binary", "display": "gpa_3plus", "step": 1,
                  "step_text": "a GPA of 3.0 or above (beyond what the GPA itself suggests)"},
    "log_internship_months": {"name": "internship experience", "higher": "more internship experience",
                              "lower": "less internship experience", "unit": "{v:g} months",
                              "display": "internship_months", "step": 0.6931,
                              "step_text": "roughly twice as much internship experience"},
    "log_certifications": {"name": "professional certifications", "higher": "more certifications",
                           "lower": "fewer certifications", "unit": "{v:g}", "display": "certifications_count",
                           "step": 0.6931, "step_text": "roughly twice as many certifications"},
    "log_projects": {"name": "personal / academic projects", "higher": "more projects", "lower": "fewer projects",
                     "unit": "{v:g}", "display": "project_count", "step": 0.6931,
                     "step_text": "roughly twice as many projects"},
    # Extended feature set: preparation before completing the degree (Q14, Q20, Q21, Q23, Q24, Q29).
    "has_portfolio": {"name": "a portfolio (GitHub / website)", "higher": "a portfolio", "lower": "no portfolio",
                      "unit": "binary", "step": 1, "step_text": "a portfolio (GitHub / website)"},
    "prep_years": {"name": "early preparation for IT jobs", "higher": "an earlier start preparing",
                   "lower": "a later start preparing", "unit": "{v:g} years", "step": 1,
                   "step_text": "starting to prepare one year earlier"},
    "self_learning_freq": {"name": "self-learning during the degree", "higher": "more frequent self-learning",
                           "lower": "less frequent self-learning", "unit": "{v:g} / 5", "step": 1,
                           "step_text": "self-learning rated 1 point higher (on the 1-5 scale)"},
    "hackathon_count": {"name": "hackathons and competitions", "higher": "more hackathons / competitions",
                        "lower": "fewer hackathons / competitions", "unit": "{v:g}", "step": 1,
                        "step_text": "one more hackathon or competition"},
    "open_source": {"name": "open-source contribution", "higher": "open-source contributions",
                    "lower": "no open-source contributions", "unit": "binary", "step": 1,
                    "step_text": "open-source contributions"},
    "sdlc_familiarity": {"name": "familiarity with software development practices",
                         "higher": "more familiarity with SDLC practices", "lower": "less familiarity with SDLC practices",
                         "unit": "{v:g} / 5", "step": 1, "step_text": "SDLC familiarity rated 1 point higher"},
    "degree_group": {"name": "degree programme"},
    "university_type": {"name": "type of institution"},
}
PREP_START_LABELS = {
    "before_university": "Before starting university", "first_year": "During first year",
    "second_third_year": "During second / third year", "final_year": "During final year",
    "after_graduation": "After graduation", "not_started": "I have not started preparing",
}
LEVELS = {
    "degree_group": {"computing": "Computer Science / Software Engineering", "it_is": "IT / Information Systems",
                     "specialised": "specialised (Cyber Security, Data Science, AI/ML or other)"},
    "university_type": {"state": "state universities", "non_state": "non-state institutes",
                        "foreign_affiliated": "foreign-affiliated institutes"},
}

# ---------------------------------------------------------------------------
# Global explanations (whole model)
# ---------------------------------------------------------------------------
GLOBAL_HR = {   # Cox-type models: ratio of the monthly chance of finding work
    "sooner": "Graduates with {step_text} tended to find work sooner - about {ratio:.2f} times the chance "
              "of finding a job in any given month (95% interval {lo:.2f}-{hi:.2f}).",
    "later": "Graduates with {step_text} tended to take longer to find work - about {ratio:.2f} times the chance "
             "of finding a job in any given month (95% interval {lo:.2f}-{hi:.2f}).",
    "level_sooner": "Graduates from {level} tended to find work sooner than those from {reference} "
                    "(about {ratio:.2f} times the monthly chance; 95% interval {lo:.2f}-{hi:.2f}).",
    "level_later": "Graduates from {level} tended to take longer to find work than those from {reference} "
                   "(about {ratio:.2f} times the monthly chance; 95% interval {lo:.2f}-{hi:.2f}).",
}
GLOBAL_TR = {   # AFT models: time ratios (< 1 = shorter search)
    "sooner": "Graduates with {step_text} tended to have a search about {pct:.0f}% shorter "
              "(time ratio {ratio:.2f}; 95% interval {lo:.2f}-{hi:.2f}).",
    "later": "Graduates with {step_text} tended to have a search about {pct:.0f}% longer "
             "(time ratio {ratio:.2f}; 95% interval {lo:.2f}-{hi:.2f}).",
    "level_sooner": "Graduates from {level} tended to have a search about {pct:.0f}% shorter than those from "
                    "{reference} (time ratio {ratio:.2f}; 95% interval {lo:.2f}-{hi:.2f}).",
    "level_later": "Graduates from {level} tended to have a search about {pct:.0f}% longer than those from "
                   "{reference} (time ratio {ratio:.2f}; 95% interval {lo:.2f}-{hi:.2f}).",
}
GLOBAL_NO_INTERVAL = "Graduates with {step_text} tended to find work {direction} (no interval available for this model)."
UNCERTAIN_SUFFIX = " The data are also consistent with no real difference."

# ---------------------------------------------------------------------------
# Local explanations (this profile vs the typical profile)
# ---------------------------------------------------------------------------
# |log-scale contribution| cut-offs -> how strongly the pattern is described
STRENGTH = [(0.30, "noticeably"), (0.10, "somewhat"), (0.0, "a little")]
LOCAL = {
    "helping_higher": "{Name}: {value}, above the typical graduate in our data. Graduates with {higher} "
                      "tended to find work {strength} sooner.",
    "helping_lower": "{Name}: {value}, below the typical graduate in our data. In our data, graduates with {lower} "
                     "tended to find work {strength} sooner.",
    "hurting_higher": "{Name}: {value}, above the typical graduate in our data. In our data, graduates with "
                      "{higher} tended to take {strength} longer to find work.",
    "hurting_lower": "{Name}: {value}, below the typical graduate in our data. Graduates with {lower} "
                     "tended to take {strength} longer to find work.",
    "helping_level": "Graduates from {level} tended to find work {strength} sooner than average in our data.",
    "hurting_level": "Graduates from {level} tended to take {strength} longer than average to find work in our data.",
}
LOCAL_NOTE = ("These are patterns in past graduates' data, not guarantees. They describe what graduates with "
              "profiles like yours experienced; they do not show that changing one thing on its own changes the "
              "outcome.")

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
HEADLINE = {   # {low}/{high}/{median} are whole months; final-year students get forward-looking wording
    "graduated": "Graduates with profiles like yours typically found work {low}-{high} months after graduating "
                 "(most likely around {median}).",
    "final_year": "Based on graduates with profiles like yours, you might find work {low}-{high} months after you "
                  "graduate (most likely around {median}).",
}
HEADLINE_OPEN = {
    "graduated": "Graduates with profiles like yours typically found work {low} to more than {horizon} months after "
                 "graduating (most likely around {median}).",
    "final_year": "Based on graduates with profiles like yours, you might find work {low} to more than {horizon} "
                  "months after you graduate (most likely around {median}).",
}
HEADLINE_NO_MEDIAN = {
    "graduated": "Many graduates with profiles like yours were still looking {horizon} months after graduating; "
                 "the quickest quarter found work within {low} months.",
    "final_year": "Many graduates with profiles like yours were still looking {horizon} months after graduating; "
                  "the quickest quarter found work within {low} months.",
}
# Graduates who are still looking: the prediction is conditional on not having found work after {m} months.
HEADLINE_SEARCHING = ("Graduates with profiles like yours who were still looking {m} months after graduating typically "
                      "found work {low}-{high} months after graduating - about {rlow}-{rhigh} more months (most "
                      "likely around {median}).")
HEADLINE_SEARCHING_OPEN = ("Graduates with profiles like yours who were still looking {m} months after graduating "
                           "typically found work {low} to more than {horizon} months after graduating (most likely "
                           "around {median}, about {rmedian} more months).")
HEADLINE_SEARCHING_NO_MEDIAN = ("Many graduates with profiles like yours who were still looking {m} months after "
                                "graduating were still looking at {horizon} months; the quickest quarter found work "
                                "within {low} months of graduating.")
CAUTION_FEW_STILL_SEARCHING = ("Only {n} graduates in our data were still looking {m} months after graduating, so this "
                               "estimate is based on very few people - treat it with caution.")
SEARCH_BEYOND_FOLLOW_UP = ("Our data follow graduates for {horizon} months, so we cannot estimate times for someone who "
                           "has already been looking for {m} months.")
SEARCH_ONLY_GRADUATES = "'Months since graduating' only applies to graduates who are still looking for work."

TIME_REFERENCE = {"graduated": "months after graduating", "final_year": "months after you graduate"}
CONFIDENCE_EXPLANATION = {
    "High": "Re-estimating the model on resampled data moves the typical time by about {w} months or less.",
    "Medium": "Re-estimating the model on resampled data moves the typical time by up to about {w} months.",
    "Low": "The estimate is unstable for profiles like this - re-estimating the model on resampled data moves "
           "it by more than {w} months, or few similar graduates were in the data.",
}

# ---------------------------------------------------------------------------
# What-if
# ---------------------------------------------------------------------------
WHATIF = {
    "sooner": "Profiles like the changed one tended to find work about {months} month{s} sooner "
              "(typical time {new} vs {old} months).",
    "later": "Profiles like the changed one tended to take about {months} month{s} longer "
             "(typical time {new} vs {old} months).",
    "same": "Profiles like the changed one had about the same typical time ({new} months).",
    "open_ended": "Typical time: {new} months for the changed profile vs {old} months now.",
}
CAUTION_RARE = ("Few people in our data had this profile ({features}) - treat this estimate with caution.")
CAUTION_OUTSIDE = ("Nobody in our data had a value this {direction} for {feature} - this estimate goes beyond "
                   "what the data can support.")

# ---------------------------------------------------------------------------
# Market alignment
# ---------------------------------------------------------------------------
MARKET_GAP = ("{label}: employers rated it {importance} out of 5 for graduates and it appeared in {ads_pct}% of "
              "the job ads we collected; you rated yourself {rating} out of 5.")
MARKET_GAP_NO_EMPLOYER = ("{label}: it appeared in {ads_pct}% of the job ads we collected; you rated yourself "
                          "{rating} out of 5.")
MARKET_NO_GAPS = "Your ratings are at or above what employers and job ads emphasise for every skill we track."
MARKET_NOTE = ("Based on {n_employers} employer responses and {n_ads} job ads. Demand describes what employers "
               "asked for, not what guarantees a job.")

# ---------------------------------------------------------------------------
# Peer benchmarks
# ---------------------------------------------------------------------------
PEER_GROUPS = {
    "overall": "All graduates",
    "degree_group": "Degree programme",
    "university_type": "Type of institution",
    "internship": "Internship experience",
    "gpa_band": "GPA",
}
PEER_LEVELS = {
    "overall": {"all": "All graduates"},
    "degree_group": LEVELS["degree_group"],
    "university_type": LEVELS["university_type"],
    "internship": {"yes": "Had an internship", "no": "No internship"},
    "gpa_band": {"below_3.0": "Below 3.00", "3.0_to_3.49": "3.00-3.49", "3.5_plus": "3.50 and above"},
}
PEER_MEDIAN = ("Half had found work by {median} months after graduating (given the group size, this could "
               "plausibly be anywhere from {lo} to {hi} months).")
PEER_MEDIAN_OPEN = "Fewer than half had found work within {horizon} months of graduating."
PEER_SUPPRESSED = "Not enough data (fewer than {min_n} graduates in this group)."
PEER_NOTE = ("Medians use all graduates in each group, including those still looking when surveyed, so they are "
             "not pulled down by leaving out people who had not found work yet.")

# ---------------------------------------------------------------------------
# About / disclaimer (shown by the frontend; served by GET /model-info)
# ---------------------------------------------------------------------------
DISCLAIMER = ("Decision support only - not a hiring tool. Based on survey data from {n} Sri Lankan IT graduates.")
SYNTHETIC_BANNER = "DEMO - SYNTHETIC DATA. These results come from computer-generated data, not real graduates."
ASSOCIATIONAL_NOTE = ("All explanations describe patterns in past graduates' data. They show which profiles tended to "
                      "find work sooner; they cannot tell you the effect of changing any single thing about yourself. "
                      "Many things that affect a job search (the economy, networks, interviews, location) are not in "
                      "the data.")
PRIVACY = ("Nothing you enter is stored. Your profile is used only to calculate the results on this page and is "
           "discarded as soon as the page is closed. There are no accounts and no tracking.")
LIMITATIONS = [
    "The model learned from {n_train} of the {n_modelling} graduates who answered the survey (the rest were kept "
    "aside to test it), so estimates carry real uncertainty - see the confidence label and ranges.",
    "Skill levels are self-rated, and graduates rated them looking back to when they completed their degree.",
    "Respondents chose to take part, so they may not represent every Sri Lankan IT graduate.",
    "Follow-up ends at 24 months; times beyond that are reported as 'more than 24 months', not estimated.",
    "Patterns are associations only: the model cannot say what effect changing one thing would have.",
    "The model was checked for similar accuracy across gender and university type, but small groups limit "
    "what that check can detect.",
]



def limitations(info: dict) -> list[str]:
    """LIMITATIONS with the sample sizes of the data actually used."""
    n = {"n_train": info.get("training_n", "the"), "n_modelling": info.get("modelling_n", info.get("training_n", ""))}
    return [s.format(**n) for s in LIMITATIONS]


# ---------------------------------------------------------------------------
# Frontend input labels
# ---------------------------------------------------------------------------
STATUS_LABELS = {"graduated": "I have graduated", "final_year": "I'm in my final year"}
DEGREE_TYPE_LABELS = {
    "computer_science": "Computer Science", "information_technology": "Information Technology",
    "software_engineering": "Software Engineering", "information_systems": "Information Systems",
    "cyber_security": "Cyber Security", "data_science": "Data Science",
    "ai_ml": "Artificial Intelligence / Machine Learning", "other": "Other",
}
INSTITUTION_LABELS = {"state": "State university", "non_state": "Non-state (private) institute",
                      "foreign_affiliated": "Foreign-affiliated institute"}
# Real Sri Lankan institutions, for the name picker only. The name is NOT sent to the model (it uses the type)
# and no statistics are shown per institution. State: the 17 UGC universities plus state universities under
# other ministries. Non-state / foreign-affiliated: widely known IT-degree providers, not exhaustive.
# Check against the current UGC lists (ugc.ac.lk) before data collection.
INSTITUTION_OTHER = "Other (not listed)"
INSTITUTIONS_BY_TYPE = {
    "state": [
        "University of Colombo (incl. UCSC)", "University of Peradeniya", "University of Sri Jayewardenepura",
        "University of Kelaniya", "University of Moratuwa", "University of Jaffna", "University of Ruhuna",
        "The Open University of Sri Lanka", "Eastern University, Sri Lanka", "South Eastern University of Sri Lanka",
        "Rajarata University of Sri Lanka", "Sabaragamuwa University of Sri Lanka", "Wayamba University of Sri Lanka",
        "Uva Wellassa University", "University of the Visual and Performing Arts",
        "Gampaha Wickramarachchi University of Indigenous Medicine", "University of Vavuniya",
        "General Sir John Kotelawala Defence University (KDU)", "University of Vocational Technology (UNIVOTEC)",
        "Ocean University of Sri Lanka",
    ],
    "non_state": [
        "Sri Lanka Institute of Information Technology (SLIIT)", "NSBM Green University",
        "Horizon Campus", "CINEC Campus", "Sri Lanka Technological Campus (SLTC)", "KIU", "Saegis Campus",
        "Aquinas College of Higher Studies",
    ],
    "foreign_affiliated": [
        "Informatics Institute of Technology (IIT)", "APIIT Sri Lanka", "ICBT Campus",
        "National Institute of Business Management (NIBM)", "ESOFT Metro Campus", "Java Institute",
        "BCAS Campus",
    ],
}
DEGREE_CLASS_LABELS = {"first": "First class", "second_upper": "Second class upper",
                       "second_lower": "Second class lower", "third_or_pass": "Third class / general pass"}
CONFIDENCE_LABEL = {"High": "High confidence", "Medium": "Medium confidence", "Low": "Low confidence"}
PEERS_UNADJUSTED_NOTE = ("Each group is compared on its own, without accounting for other differences (GPA, skills, "
                         "experience). That is why a group can look faster or slower here than the model's "
                         "explanations suggest.")
WHATIF_PROMPT = "Move a slider to see how graduates with the changed profile compared."

# ---------------------------------------------------------------------------
# Model performance in plain terms (About tab, model report)
# ---------------------------------------------------------------------------
PLAIN_METRICS = {
    "ranking": ("Ranking: out of every 100 pairs of graduates, the model correctly picks which one found work "
                "first about {pairs} times (50 would be pure guessing, 100 would be perfect)."),
    "calibration": ("Accuracy of the predicted shares: the share of graduates it predicted would be working within "
                    "{h} months was, on average, within {pts} percentage points of what actually happened."),
    "why_not_accuracy": ("There is no single 'accuracy %': many graduates were still looking when surveyed, so "
                         "their final outcome is unknown. The two measures above are the standard ones for this "
                         "kind of data."),
}

# ---------------------------------------------------------------------------
# "Chance of having found work by ..." milestones
# ---------------------------------------------------------------------------
MILESTONE = ("By {t} months after graduating, about {pct}% of graduates with profiles like yours had found an IT job "
             "(likely between {lo}% and {hi}%).")
MILESTONE_SEARCHING = ("Of graduates like you who were still looking after {m} months, about {pct}% had found an IT "
                       "job by {t} months after graduating (likely between {lo}% and {hi}%).")

# ---------------------------------------------------------------------------
# Suggested combinations to explore (associational, never promises)
# ---------------------------------------------------------------------------
SUGGESTION = ("Graduates with profiles like yours but with {changes} tended to find work about {n} month{s} sooner "
              "(typical time {new} vs {old} months).")
SUGGESTION_OPEN = ("Graduates with profiles like yours but with {changes} more often found work within the "
                   "{horizon} months we can observe (typical time {new} months).")
SUGGESTION_CHANGE = {
    "internship_months": "{d:g} more months of internship",
    "certifications_count": "{d:g} more certification{s}",
    "project_count": "{d:g} more project{s}",
    "gpa": "a GPA of {new:.2f}",
    "area": "{label} rated {new:g} instead of {old:g}",
}
SUGGESTIONS_NOTE = ("These combinations are patterns in past graduates' data, chosen as small, realistic changes. "
                    "They are ideas to explore, not guarantees.")
NO_SUGGESTIONS = ("No small, realistic change in the things you can control is associated with a noticeably "
                  "shorter search for profiles like yours.")
