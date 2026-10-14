# Testing

Run everything from the project root (venv active):

```
pytest
```

216 automated tests, about 2 minutes. No test needs `artifacts/` or `data/` to exist - each builds its
own small synthetic dataset and models, and anything written goes to a temporary folder.

## Requirement -> test map

| Requirement | Test(s) |
|---|---|
| **Cleaning:** training set is graduates only | `test_clean.py::test_training_set_is_graduates_only`, `::test_exclusion_rules` |
| Correct duration / event: employed, still searching, job before graduation | `test_clean.py::test_duration_and_event_for_each_case` |
| Audit counts add up (raw N -> exclusions -> final N) | `test_clean.py::test_audit_counts_add_up` |
| Every injected data defect is handled | `test_clean.py::test_every_injected_defect_is_handled`, `::test_out_of_range_values_become_missing` |
| Real (Google Forms) export cleans identically to the canonical format | `test_clean.py::test_google_forms_export_cleans_identically` |
| Real data file takes priority over synthetic | `test_clean.py::test_real_file_takes_priority_over_synthetic` |
| **No leakage** of outcome / post-employment / fairness fields | `test_clean.py::test_no_leakage_into_model_features`, `test_preprocess.py::test_output_has_exactly_the_model_columns` |
| Preprocessing fitted on training folds only; missing categories imputed (never silently the reference level) | `test_preprocess.py::test_fitted_on_training_fold_only`, `::test_missing_category_is_imputed_not_treated_as_reference` |
| Every graduate has exactly one role (train / validation / test), fixed and stratified | `test_pipeline.py::test_split_gives_every_graduate_exactly_one_role` |
| Tuning and model comparison never fit on validation or test rows | `test_pipeline.py::test_tuning_and_comparison_never_fit_on_validation_or_test` |
| Final model fitted on train only; test set used once for evaluation | `test_pipeline.py::test_final_model_and_test_evaluation_use_each_set_once` |
| **Cox recovers known effect directions** on synthetic data | `test_synthetic.py::test_cox_recovers_true_effect_directions` (generator truth), `test_pipeline.py::test_cox_through_full_pipeline_recovers_known_directions` (full pipeline) |
| Model interface: S(0)=1, curves non-increasing, pickling | `test_models.py::test_model_interface[*]` (all 6 models) |
| No forest / boosting / ensemble methods | `test_models.py::test_no_forest_or_boosting_models` |
| Model-selection rule | `test_models.py::test_selection_*` |
| Extended predictor set: candidates preprocessed, no leakage; pre-registered adoption rule needs size AND consistency; plain-language metrics | `test_feature_sets.py` |
| Bootstrap interval never NaN when some bootstrap medians are never reached; fast AFT bootstrap = lifelines | `test_predict.py::test_bootstrap_interval_handles_never_reached_medians`, `::test_fast_aft_bootstrap_matches_lifelines[*]` |
| **Predict:** low <= median <= high | `test_predict.py::test_values_are_ordered_for_many_profiles` |
| Predict: "more than X" instead of extrapolating | `test_predict.py::test_more_than_x_when_curve_never_reaches_level`, `::test_crossings_beyond_horizon_are_not_reported` |
| Fast bootstrap path = exact slow path | `test_predict.py::test_fast_bootstrap_path_matches_slow_path` |
| **What-if** rejects non-actionable features | `test_explain_whatif.py::test_whatif_rejects_non_actionable[*]` |
| What-if rejects out-of-range values; GPA rule for final-year | `test_explain_whatif.py::test_whatif_rejects_out_of_range[*]`, `::test_gpa_change_only_for_final_year_within_limit` |
| Adding internship experience does not lengthen predicted time | `test_explain_whatif.py::test_adding_internship_does_not_lengthen_predicted_time` |
| What-if flags profiles outside the training range | `test_explain_whatif.py::test_whatif_flags_profiles_outside_training_range` |
| All wording associational, no jargon | `test_explain_whatif.py::test_all_wording_is_associational_and_jargon_free`, `::test_generated_explanations_are_associational` |
| Explanations for every model family (Cox, Elastic Net, AFT, DeepSurv/SHAP) | `test_explain_whatif.py::test_*_uses_*` |
| Graduates still looking: conditional prediction, never before "now", Low confidence when few comparable graduates | `test_conditional_actionable.py::test_*searching*`, `::test_remaining_months_are_relative_to_now`, `::test_invalid_months_since_graduation` |
| "What's helping / holding back" shows only actionable features (GPA for final-year only) | `test_conditional_actionable.py::test_local_explanations_show_only_actionable_features`, `::test_gpa_is_actionable_only_for_final_year` |
| **Peer benchmarks** suppress n < 10 | `test_market_peers.py::test_small_groups_are_suppressed` |
| Peer medians are Kaplan-Meier, not raw | `test_market_peers.py::test_peer_medians_are_kaplan_meier_not_raw` |
| reference.db holds no respondent-level data; no scraping code | `test_market_peers.py::test_reference_db_contents_and_privacy`, `::test_no_web_scraping_code` |
| Fairness: suppression n < 20, gender never a feature | `test_fairness.py` |
| **Every API endpoint returns its declared schema** | `test_api.py::test_health`, `::test_predict`, `::test_explain`, `::test_whatif`, `::test_suggestions`, `::test_market_alignment`, `::test_peer_benchmark`, `::test_model_info` |
| Milestones rise over time with 80% ranges and skip months already passed; suggestions use only actionable features, different features each, effort <= 3, at least 1 month sooner, associational wording; nothing suggested at the top of every range; every skill has a free resource and top gaps carry up to 2 | `test_guidance.py` |
| **Chat assistant**: recognises each question type, turns what-if questions into allowed changes (clipped to valid ranges), repeats the model's own numbers and wording, refuses GPA changes for graduates and off-topic questions, stores nothing and calls no external service; `/chat` returns its schema and rejects empty, over-long or extra fields | `test_chat.py`, `test_api.py::test_chat*` |
| API rejects invalid / extra fields (e.g. gender) and never stores profiles | `test_api.py::test_invalid_*`, `::test_api_never_stores_profiles` |
| **Benchmark scorer handles unparseable responses** | `test_benchmark.py::test_parse_months_unparseable[*]`, `::test_score_with_responses_including_unparseable` |
| LLM API mode: fills template rows, resumes, appends a second LLM, saves on failure, records refusals, no network in tests | `test_run_llm.py` |
| Secondary analyses: Q16a leaves the main data unchanged and old exports without it are accepted; competing-outcome rules; cumulative incidences <= 1; careless / heaped detection; Rubin's rules; multiple imputation; earlier-vs-later split; conservative clustering CI never narrower, institutions never named | `test_robustness.py` |
| Prompt variants: standard prompt unchanged, Sri Lanka absent only in `no_country`, range parsing and coverage, variant rows kept separate in API mode, old responses without a variant column still scored | `test_benchmark.py::test_*variant*`, `::test_parse_range*`, `test_run_llm.py::test_variant_rows_are_kept_separate`, `::test_range_prompts_are_sent_for_range_variant` |
| Prompts contain no outcome; filled responses are never overwritten; stale responses refused | `test_benchmark.py::test_prompts_contain_no_outcome`, `::test_existing_responses_are_never_overwritten`, `::test_score_refuses_stale_responses` |

## Tests per file

| File | Tests |
|---|---|
| test_api.py | 21 |
| test_benchmark.py | 27 |
| test_clean.py | 12 |
| test_conditional_actionable.py | 9 |
| test_explain_whatif.py | 21 |
| test_fairness.py | 5 |
| test_guidance.py | 7 |
| test_feature_sets.py | 6 |
| test_market_peers.py | 9 |
| test_models.py | 14 |
| test_pipeline.py | 2 |
| test_predict.py | 14 |
| test_preprocess.py | 7 |
| test_robustness.py | 11 |
| test_run_llm.py | 13 |
| test_synthetic.py | 8 |

## Manual checks (not automated)

- Frontend: run API + Streamlit, check the synthetic banner, Quick vs Detailed mode, a what-if change, a "Try this" button
  and all three tabs. (A headless Streamlit `AppTest` smoke run was used during development.)
- `python -m core.run_all` twice and compare `artifacts/model_comparison.csv` - results are identical.
