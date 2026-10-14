"""Research lab: the evidence behind the model (GET /research and GET /model-info)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend import charts_study as cs
from frontend import ui
from frontend.api_client import APIError

try:
    info = ui.model_info()
except APIError as e:
    st.error(str(e))
    st.stop()
r = ui.fetch("/research")
if not r:
    st.stop()
data_tag = "synthetic" if r["is_synthetic"] else "real"
sel = next((m for m in r["models"] if m["model"] == r["selected_model"]), r["models"][0] if r["models"] else None)

ui.hero("Research lab", "How good is the model - and can it be trusted?",
        "Seven survival models fitted on a training set and compared on a validation set; the chosen model is tested "
        "once on a locked test set, then checked for calibration, fairness and robustness"
        + (" - and, because the generator's true effects are known, whether the analysis recovers them."
           if r["is_synthetic"] else "."), compact=True)
if r["is_synthetic"]:
    st.markdown("<div class='notice'>⚠️ <div>These results come from the <b>synthetic graduate survey</b>. They show that "
                "the methods work as designed; they are not findings about real graduates.</div></div>",
                unsafe_allow_html=True)

test = info.get("test_metrics") or {}
if sel and test:
    ui.kpis([
        (f"{test['uno_c']:.2f}", "Uno's C-index (locked test set)",
         f"95% CI {test['uno_c_lo']:.2f}-{test['uno_c_hi']:.2f} · {info['selected_model_label']}", data_tag),
        (f"{test['ibs']:.3f}", "Integrated Brier score (test)", "lower is better (0.25 = uninformative)"),
        (f"{100 * test['cal_err_6m']:.1f} pts", "Calibration error at 6 months (test)", "predicted vs observed share employed"),
        (info.get("modelling_n", info["training_n"]), "Graduates",
         f"train {info['training_n']} · validation {info.get('validation_n', 0)} · test {info.get('test_n', 0)}"),
    ])

tabs = st.tabs([":material/leaderboard: Models", ":material/target: Calibration", ":material/balance: What matters",
                ":material/timeline: Time to job", ":material/diversity_3: Fairness", ":material/shield: Robustness",
                ":material/casino: Simulation study", ":material/quiz: Questionnaire", ":material/description: About"])

with tabs[0]:
    ui.section("Model comparison", "Discrimination of each candidate on the validation set (fitted on the training "
                                   "set; no graduate is reused). Bars show a bootstrap interval over the validation "
                                   f"graduates. Selected: <b>{info['selected_model_label']}</b> - "
                                   f"{info['selection_reason']}.", data_tag)
    c1, c2 = st.columns([1.4, 1], gap="large")
    with c1:
        st.plotly_chart(cs.model_cindex(r["models"], r["selected_model"]), config=ui.PLOT, key="models")
    with c2:
        st.dataframe(pd.DataFrame(r["models"])[["label", "uno_c_mean", "harrell_c_mean", "ibs_mean", "cal_err_mean"]]
                     .rename(columns={"label": "Model", "uno_c_mean": "Uno's C", "harrell_c_mean": "Harrell's C",
                                      "ibs_mean": "IBS", "cal_err_mean": "Cal. error"}).round(3), hide_index=True)
        st.caption("Uno's C: chance a graduate who found work sooner was ranked as more likely to - 0.5 is guessing. "
                   "IBS: average squared error of predicted employment curves over time.")
    acc = r.get("accuracy")
    if acc:
        ui.section("Can a more flexible method do better?",
                   "Exploratory and report-only: non-linear terms, gradient boosting, a random survival forest and an "
                   "ensemble, fitted on the training set and scored on the validation set."
                   + (" The green line is the ceiling - the C-index a model would reach if it knew every graduate's "
                      "true risk." if acc.get("ceiling") else ""), data_tag)
        c1, c2 = st.columns([1.4, 1], gap="large")
        with c1:
            st.plotly_chart(cs.accuracy_ladder(acc["summary"], acc.get("ceiling")), config=ui.PLOT, key="ladder")
        with c2:
            best = max(acc["summary"], key=lambda s: s["uno_c_mean"])
            base = next(s for s in acc["summary"] if s["model"] == "cox_core")
            ui.finding(f"Best flexible method: +{best['uno_c_mean'] - base['uno_c_mean']:.3f}",
                       f"{best['label']} vs plain Cox PH - within the uncertainty of a small validation set "
                       f"(± {base['uno_c_sd']:.2f}).")
            if acc.get("ceiling"):
                c = acc["ceiling"]["mean"]
                ui.finding(f"Ceiling ≈ {c:.2f}",
                           f"The model captures {100 * (base['uno_c_mean'] - 0.5) / (c - 0.5):.0f}% of the discrimination "
                           "that is achievable at all; the rest is luck, networks and the gap between self-rated and true "
                           "skill, which no survey item measures.")
    if r["feature_sets"]:
        st.markdown("**Do other feature sets help? (extended = + portfolio, hackathons, open source, preparation; transformed = log counts and GPA cut-off)**")
        fs = pd.DataFrame(r["feature_sets"])
        st.dataframe(fs[["model", "challenger", "c_base", "c_other", "delta_c", "win_share"]]
                     .rename(columns={"model": "Model", "challenger": "Compared set", "c_base": "C (core)",
                                      "c_other": "C (compared set)",
                                      "delta_c": "Change", "win_share": "Better in (share of resamples)"}).round(3), hide_index=True)

with tabs[1]:
    ui.section("Calibration", "Graduates are split into thirds by predicted chance of employment; each point compares "
                              "the predicted share with the share that actually found work. On the dotted line = "
                              "perfectly calibrated.", data_tag)
    c1, c2 = st.columns([1.3, 1], gap="large")
    with c1:
        if r["calibration"]:
            st.plotly_chart(cs.calibration(r["calibration"]), config=ui.PLOT, key="cal")
    with c2:
        st.markdown("**In plain words**")
        st.markdown("\n".join(f"- {x}" for x in info["plain_metrics"]))

with tabs[2]:
    ui.section("What is associated with finding work sooner", "Hazard ratios from the Cox model with 95% intervals. "
                                                              "Above 1 = found an IT job sooner; grey = interval includes 1.",
               data_tag)
    if r["hazard_ratios"]:
        st.plotly_chart(cs.forest(r["hazard_ratios"]), config=ui.PLOT, key="forest")
    st.caption(info["associational_note"])

with tabs[3]:
    ui.section("Time to first IT job", "Kaplan-Meier estimate: the share of graduates who had found an IT job by each "
                                       "month, accounting for those still looking. Shaded = 95% interval.", data_tag)
    if r["km"]:
        st.plotly_chart(cs.km_curve(r["km"], info["horizon_months"]), config=ui.PLOT, key="km")

with tabs[4]:
    ui.section("Fairness by subgroup", "Does the model rank graduates equally well across gender and institution type? "
                                       "Overlapping intervals = no evidence of a difference. Gender is never a model input.",
               data_tag)
    if r["fairness"]:
        c1, c2 = st.columns([1.3, 1], gap="large")
        with c1:
            st.plotly_chart(cs.subgroup_c(r["fairness"]), config=ui.PLOT, key="fair")
        with c2:
            f = pd.DataFrame(r["fairness"])
            st.dataframe(f[~f["suppressed"]][["group", "level", "n", "uno_c", "pred_employed_6m", "obs_employed_6m"]]
                         .rename(columns={"group": "Group", "level": "Level", "uno_c": "Uno's C",
                                          "pred_employed_6m": "Predicted 6m", "obs_employed_6m": "Observed 6m"}).round(3),
                         hide_index=True)

with tabs[5]:
    ui.section("Robustness", "Do the results survive different data decisions, and a model trained on earlier cohorts "
                             "tested on later ones?", data_tag)
    c1, c2 = st.columns([0.45, 0.55], gap="large")
    with c1:
        st.markdown("**Sensitivity to cleaning decisions**")
        if r["sensitivity"]:
            sens = r["sensitivity"]
            main = {s["model"]: s for s in sens if s["variant"] == "main"}
            same = {s["variant"] for s in sens if s["variant"] != "main" and all(
                    o["n"] == main[o["model"]]["n"] and o["uno_c"] == main[o["model"]]["uno_c"]
                    for o in sens if o["variant"] == s["variant"] and o["model"] in main)}
            sens = [dict(s, label=s["label"] + " *") if s["variant"] in same else s for s in sens]
            labels = {m["model"]: m["label"] for m in r["models"]}
            labels[r["selected_model"]] = info["selected_model_label"]
            st.plotly_chart(cs.variant_c(sens, labels, r["selected_model"]), config=ui.PLOT, key="sens")
            if same:
                st.caption("\\* Same graduates as the main analysis (e.g. no model input was missing, so complete cases "
                           "= everyone), so the result is identical by construction.")
    with c2:
        st.markdown("**Temporal validation (train early cohorts, test later)**")
        if r["temporal"]:
            t = pd.DataFrame(r["temporal"])
            st.dataframe(t[["model", "uno_c", "uno_c_cv", "ibs", "max_abs_cal_error"]]
                         .rename(columns={"model": "Model", "uno_c": "C later", "uno_c_cv": "C all valid.",
                                          "ibs": "IBS", "max_abs_cal_error": "Max cal. err"}).round(3),
                         hide_index=True, use_container_width=True)
            st.caption("C later: trained on earlier cohorts, tested on the latest one. C all valid.: the whole "
                       "validation set. Max cal. err: largest gap between predicted and observed share employed.")

with tabs[6]:
    sim = r.get("simulation")
    ui.section("Simulation study", "Because the synthetic generator's true effects are known, we can test the method "
                                   "itself: draw hundreds of fresh surveys, refit, and count how often each true effect is "
                                   "found. The curves show how many respondents a real survey needs.", "synthetic")
    if not sim:
        st.info("Run `python -m core.simulation_study` to produce the simulation results.")
    else:
        sizes = sim["sample_sizes"]
        st.plotly_chart(cs.power_curves(sim), config=ui.PLOT, key="power")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("**False alarms for effects that are truly zero** (expected ≈ 5%)")
            nulls = [k for k, e in sim["by_size"][str(sizes[0])]["effects"].items() if e["true"] == 0]
            st.dataframe(pd.DataFrame([{"Predictor": sim["by_size"][str(sizes[0])]["effects"][k]["label"],
                                        **{f"n={n}": f"{100 * sim['by_size'][str(n)]['effects'][k]['false_positive_rate']:.0f}%"
                                           for n in sizes}} for k in nulls]), hide_index=True)
        with c2:
            st.markdown("**Bias and 95% interval coverage** (predictors measured exactly)")
            exact = [k for k, e in sim["by_size"][str(sizes[0])]["effects"].items() if "coverage" in e]
            st.dataframe(pd.DataFrame([{"Predictor": sim["by_size"][str(sizes[0])]["effects"][k]["label"],
                                        "True": sim["by_size"][str(sizes[0])]["effects"][k]["true"],
                                        **{f"n={n}": f"{sim['by_size'][str(n)]['effects'][k]['mean_estimate']:+.3f} "
                                                     f"({100 * sim['by_size'][str(n)]['effects'][k]['coverage']:.0f}%)"
                                           for n in sizes}} for k in exact]), hide_index=True)
        st.caption(f"{sim['reps_per_size']} simulated surveys per sample size. Unobserved factors (frailty) shrink "
                   "estimated effects towards zero, so coverage below 95% is expected and is itself a finding about "
                   "survey-based studies like this one.")

with tabs[7]:
    qs = ui.fetch("/questionnaire")
    if qs:
        q_tag = "synthetic" if qs["is_synthetic"] else "real"
        ui.section("Repeated questions",
                   "Questions that ask about a model question again in other words, tested against it. A strong "
                   "correlation means the two measure the same thing, so one of them can be dropped.", q_tag)
        c1, c2 = st.columns([1.5, 1], gap="large")
        with c1:
            st.plotly_chart(cs.repeat_rho(qs["repeated"]), config=ui.PLOT, key="q_repeat")
        with c2:
            st.dataframe(pd.DataFrame(qs["repeated"])[["repeat", "model_question", "rho", "n", "verdict"]]
                         .rename(columns={"repeat": "Repeat item", "model_question": "Model question",
                                          "verdict": "Verdict"}), hide_index=True)
            st.dataframe(pd.DataFrame(qs["logic"])[["check", "contradictions", "n", "pct"]]
                         .rename(columns={"check": "Logic check", "contradictions": "Contradictions", "pct": "%"}),
                         hide_index=True)
        gve = qs["graduates_vs_employers"]
        ui.section("Graduates vs employers", "The same question asked of both groups. ★ = significant after "
                                             "Benjamini-Hochberg correction.", q_tag)
        st.dataframe(pd.DataFrame(gve["likert"])[["question", "mean_a", "mean_b", "rank_biserial", "p"]]
                     .rename(columns={"question": "Question", "mean_a": "Graduates (mean 1-5)",
                                      "mean_b": "Employers (mean 1-5)", "rank_biserial": "Effect size"}).round(3),
                     hide_index=True)
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("**Skill gaps** - graduates who say they need to improve vs employers who see it as a "
                        "common weakness")
            st.plotly_chart(cs.group_dumbbell(gve["skill_gaps"]), config=ui.PLOT, key="q_gaps")
        with c2:
            st.markdown("**What matters most** - graduates who chose it vs employers who ranked it in their top 3")
            st.plotly_chart(cs.group_dumbbell(gve["what_matters"]), config=ui.PLOT, key="q_matters")
        sp, ar = qs["self_perception"], qs["artifact"]
        ui.section("Self-perception and the artifact", "", q_tag)
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown(f"Self-rated employability (Q40) alone ranks graduates by time to first job with "
                        f"**C = {sp.get('harrell_c', '-')}**. It is asked after the outcome, so it partly reflects "
                        "whether the graduate already has a job - it is reported, never used as a predictor.")
            st.dataframe(pd.DataFrame([{"Group": k, "n": v.get("n"), "Mean (1-5)": v.get("mean"),
                                        "% rating 4-5": v.get("pct_agree")} for k, v in sp["by_group"].items()]),
                         hide_index=True)
        with c2:
            st.markdown(f"- **{ar['explanation_importance'].get('pct_agree')}%** of graduates say understanding the "
                        f"reasons behind a prediction is important (Q44).\n"
                        f"- **{ar['would_use_system'].get('pct_agree')}%** would use the system before applying (Q46); "
                        f"**{ar['employer_usefulness'].get('pct_agree')}%** of employers find it useful (E16).")
            st.dataframe(pd.DataFrame(ar["explanation_types"])[["label", "pct"]]
                         .rename(columns={"label": "Explanation wanted (Q45)", "pct": "%"}), hide_index=True)
        with st.expander("Every questionnaire item"):
            for x in qs["graduate_items"] + qs["employer_items"]:
                if "summary" in x:
                    s_ = x["summary"]
                    st.markdown(f"**{x['item']}** {x['description']} - mean {s_.get('mean')}, "
                                f"{s_.get('pct_agree')}% rated 4-5 (n = {s_.get('n')})")
                elif "shares" in x:
                    st.markdown(f"**{x['item']}** {x['description']} - " + "; ".join(
                        f"{s_['label']} {s_['pct']}%" for s_ in x["shares"][:6] if s_["pct"] is not None))

with tabs[8]:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**The data**")
        st.markdown(f"- Training data: **{info['training_n']} graduates**, of whom {info['events']} had found an IT job "
                    f"by the survey and {info['censored']} were still looking (*censored*).\n"
                    f"- Market data: {info['employers_n']} employer responses and {info['job_ads_n']} job ads.\n"
                    "- Final-year students are never used for training; they receive predictions only.")
        st.markdown("**The model**")
        st.markdown(f"- Selected model: **{info['selected_model_label']}** - {info['selection_reason']}.\n"
                    f"- Confidence labels come from 200 bootstrap refits: High if the typical time moves by at most "
                    f"{info['confidence_thresholds_months']['High']:g} months, Medium up to "
                    f"{info['confidence_thresholds_months']['Medium']:g}, otherwise Low.")
        for b in (r.get("llm_benchmark") or {}).get("comparisons", []):
            d = b["diff"]
            st.caption(f"Compared with a general-purpose chatbot: given the same {b['profiles']} test graduates, "
                       f"{b['llm']} reached Uno's C {b['uno_c']:.3f} against {b['model_uno_c']:.3f} for this model"
                       + (f" (difference {d[0]:+.3f}, 95% CI {d[1]:+.3f} to {d[2]:+.3f})" if d[0] is not None else "")
                       + ".")
    with c2:
        st.markdown("**Limitations**")
        st.markdown("\n".join(f"- {x}" for x in info["limitations"]))
        st.markdown("**Associations, not advice**")
        st.markdown(info["associational_note"])
        st.markdown("**Privacy**")
        st.markdown(info["privacy"])
