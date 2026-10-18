"""Grounded assistant: answers a graduate's questions about THEIR estimate using only the system's own outputs.

No language model is used and nothing is generated freely: the message is matched to one of a fixed set of intents
by keyword rules, and every reply is assembled from the same functions the dashboard uses (prediction, explanation,
what-if, suggestions, market alignment, peer benchmarks, model evaluation) and the wording in core/wording.py.
So every number in a reply is one the model actually produced, and the associational wording is kept.

Stateless: the profile and the message come with every call; nothing is stored or logged.
"""
from __future__ import annotations

import re

from core import schema, wording
from core.explain import Explainer
from core.predict import Predictor
from core.suggest import current_values, suggest
from core.whatif import WhatIfError, what_if

MAX_MESSAGE_CHARS = 500
FOLLOW_UPS = ["How long will it take me?", "Why is my estimate like this?", "What if I do a 6-month internship?",
              "Which skills should I learn?", "How can I find work sooner?", "Can I trust this estimate?"]
PROFILE_FREE_INTENTS = {"greeting", "help", "method", "privacy", "trust", "factor", "unknown"}
GENERAL_FOLLOW_UPS = ["What can I ask?", "How does the model work?", "Can I trust this estimate?",
                      "Do you store my data?", "Is GPA important?"]

# ---------------------------------------------------------------------------------------------- what-if parsing
WORD_NUMBERS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
NUM = r"(\d+(?:\.\d+)?|" + "|".join(WORD_NUMBERS) + r")"
ADD_WORDS = re.compile(r"\b(another|more|extra|additional|further|plus|add|do|get|complete|finish|take|earn|"
                       r"build|make)\b")
SET_WORDS = re.compile(r"\b(total|in all|altogether|had|have)\b")
AREA_WORDS = {
    "area_programming": r"programming|coding|java|python|javascript|oop",
    "area_web_apis": r"web|api|apis|rest",
    "area_data_ai": r"data|sql|database|analytics|ai\b|machine learning|ml\b",
    "area_cloud_devops": r"cloud|devops|docker|linux|ci/cd|git\b|aws|azure",
    "area_quality_security": r"testing|test\b|qa\b|quality|security|cyber",
    "area_soft": r"soft skill|communication|teamwork|team work|presentation|english",
}
WHATIF_TRIGGER = re.compile(r"\b(what if|if i|if i'd|suppose|would .* (help|change)|what happens|should i (do|get|"
                            r"take)|will .* (help|change)|how much (sooner|faster|quicker))\b")

CHAT_TERMS = (
    "what", "why", "estimate", "estimates", "predict", "prediction", "chance", "probability", "likely",
    "intern", "interns", "internship", "internships", "cert", "certs", "certification", "certifications",
    "project", "projects", "skill", "skills", "learn", "learning", "improve", "improvement", "faster", "sooner",
    "quicker", "suggest", "suggestion", "suggestions", "advice", "recommend", "recommendation", "recommendations",
    "employer", "employers", "graduate", "graduates", "graduated", "university", "universities", "institution",
    "institutions", "campus", "degree", "course", "programme", "program", "portfolio", "github", "hackathon",
    "hackathons", "competition", "competitions", "opensource", "gpa", "grade", "grades", "class", "technical",
    "programming", "confidence", "confident", "accuracy", "accurate", "reliable", "reliability", "trust", "valid",
    "average", "compare", "comparison", "peer", "peers", "similar", "factor", "factors", "important", "matter",
    "affect", "effect", "help", "helping", "holding", "back", "explain", "reason", "reasons", "data", "model",
    "algorithm", "survival", "trained", "privacy", "store", "stored", "saving", "save", "track", "tracked",
    "months", "month", "softer", "soft", "find", "finding", "work", "job", "jobs", "increase", "chance",
)


def _edit_distance(left: str, right: str, limit: int) -> int:
    """Damerau-Levenshtein distance, stopping once a token is beyond the small correction limit."""
    if abs(len(left) - len(right)) > limit:
        return limit + 1
    previous_previous = None
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, 1):
        current = [row]
        for column, right_char in enumerate(right, 1):
            cost = left_char != right_char
            value = min(
                previous[column] + 1,
                current[column - 1] + 1,
                previous[column - 1] + cost,
            )
            if (previous_previous is not None and row > 1 and column > 1
                    and left_char == right[column - 2] and left[row - 2] == right_char):
                value = min(value, previous_previous[column - 2] + 1)
            current.append(value)
        if min(current) > limit:
            return limit + 1
        previous_previous, previous = previous, current
    return previous[-1]


def normalize_chat_text(message: str) -> str:
    """Correct only clear, one/two-edit misspellings of known chat keywords."""
    message = re.sub(r"\bthank(?:you|u|yuo)\b", "thank you", message, flags=re.IGNORECASE)
    terms_by_length: dict[int, list[str]] = {}
    for term in set(CHAT_TERMS):
        terms_by_length.setdefault(len(term), []).append(term)

    def correct(match: re.Match) -> str:
        token = match.group(0)
        lower = token.lower()
        if len(lower) < 4 or lower in CHAT_TERMS:
            return token
        limit = 1 if len(lower) < 8 else 2
        candidates = []
        for length in range(max(4, len(lower) - limit), len(lower) + limit + 1):
            for term in terms_by_length.get(length, ()):
                distance = _edit_distance(lower, term, limit)
                if distance <= limit:
                    candidates.append((distance, term))
        if not candidates:
            return token
        best_distance = min(distance for distance, _ in candidates)
        closest = {term for distance, term in candidates if distance == best_distance}
        if len(closest) > 1:
            suffixes = ("ies", "ing", "ed", "s")
            suffix = next((ending for ending in suffixes if lower.endswith(ending)), None)
            if suffix:
                matching_suffix = {term for term in closest if term.endswith(suffix)}
                if len(matching_suffix) == 1:
                    closest = matching_suffix
        if len(closest) != 1:
            return token
        replacement = next(iter(closest))
        return replacement.capitalize() if token[0].isupper() else replacement

    return re.sub(r"[A-Za-z]+", correct, message)


def _num(token: str | None, default: float | None = None) -> float | None:
    if token is None:
        return default
    token = token.lower()
    return float(WORD_NUMBERS[token]) if token in WORD_NUMBERS else float(token)


def parse_changes(text: str, current: dict, status: str) -> tuple[dict, list[str]]:
    """Turn a what-if question into actionable changes. Returns (changes, assumptions stated in the reply)."""
    t = normalize_chat_text(text).lower()
    changes, notes = {}, []
    additive = bool(ADD_WORDS.search(t)) or not SET_WORDS.search(t)

    def bump(feat, amount, absolute=None):
        cur = current.get(feat) or 0.0
        spec = schema.ACTIONABLE_FEATURES[feat]
        new = absolute if absolute is not None else cur + amount
        changes[feat] = float(min(max(new, spec.min), spec.max))

    if "intern" in t:
        m = re.search(NUM + r"[\s-]*(?:more\s+|extra\s+)?months?", t)
        n = _num(m.group(1)) if m else None
        if n is None:
            n = 6.0
            notes.append("I assumed a 6-month internship, the most common length in the survey.")
        bump("internship_months", n, None if additive else n)
    if re.search(r"certif|\bcerts?\b", t):
        m = re.search(NUM + r"\s+(?:more\s+|extra\s+|additional\s+)?(?:\w+\s+)?(?:certif|certs?\b)", t)
        n = _num(m.group(1), 1.0) if m else 1.0
        bump("certifications_count", n, None if additive else n)
    if "project" in t:
        m = re.search(NUM + r"\s+(?:more\s+|extra\s+|additional\s+|personal\s+|side\s+)?projects?", t)
        n = _num(m.group(1), 1.0) if m else 1.0
        bump("project_count", n, None if additive else n)
    if "gpa" in t:
        m = re.search(r"gpa\D{0,12}(\d(?:\.\d+)?)", t) or re.search(r"(\d\.\d+)\s*gpa", t)
        if m:
            changes["gpa"] = float(m.group(1))
    for feat, pattern in AREA_WORDS.items():
        if re.search(pattern, t) and ("skill" in t or "improve" in t or "learn" in t or " to " in t or "better" in t):
            m = re.search(r"(?:" + pattern + r")[^.?!]*?\bto\s+(\d(?:\.\d)?)", t)
            if m:
                bump(feat, 0, float(m.group(1)))
            else:
                bump(feat, 1.0)
                notes.append(f"I assumed your {schema.ACTIONABLE_FEATURES[feat].label.lower()} goes up by one point "
                             "on the 1-5 scale.")
    return changes, notes


# ---------------------------------------------------------------------------------------------- intents
FEATURE_WORDS = {
    "gpa": r"\bgpa\b|grades?|results?|class\b", "internship_months": r"intern", "certifications_count": r"certif",
    "project_count": r"project", "technical_mean": r"technical skill|tech skill|programming skill",
    "soft_mean": r"soft skill", "university_type": r"universit|institut|campus|state uni|private",
    "degree_group": r"degree|course|programme|program", "has_portfolio": r"portfolio|github",
    "hackathon_count": r"hackathon|competition", "open_source": r"open.?source",
}
INTENTS = [  # checked in order; the first match wins
    ("skills", r"\bskills?\b.*\b(learn|need|gap|improve|focus|missing|demand)|which skills|what skills|"
               r"employers? (want|look|ask|value|expect)|job ads?|in demand|\bcourses?\b|what should i learn"),
    ("suggestions", r"improve|faster|sooner|quicker|suggest|advice|tips?\b|recommend|what should i do|"
                    r"how (can|do) i (get|find|land)|increase my chance|best thing"),
    ("explain", r"\bwhy\b|helping|holding (me )?back|reason|explain|what('s| is) (affecting|driving)|factors?"),
    ("factor", r"\b(important|matter|matters|affect|effect|count|help)\b"),
    ("trust", r"accura|trust|reliab|how (good|sure|certain)|confiden|c-?index|valid|wrong|error"),
    ("peers", r"\bpeers?\b|others like me|people like me|compare|similar graduates|other graduates|average"),
    ("estimate", r"how long|when (will|would|can) i|how many months|estimate|predict|chance|probab|likely|"
                 r"by \d+ months|time to"),
    ("method", r"how does (it|this|the (app|system|model)) work|survival|which model|what model|algorithm|"
               r"where.*data|what data|trained|how was"),
    ("privacy", r"stor(e|ed|ing)|save|privacy|my data|track|log"),
    ("greeting", r"^(h+i+o?|hello|hey|good (morning|afternoon|evening)|thanks|thank you)\b"),
    ("help", r"help|what can (you|i) (do|ask)|options"),
]


def classify(message: str) -> str:
    t = normalize_chat_text(message).lower().strip()
    if WHATIF_TRIGGER.search(t):
        return "whatif"
    for name, pattern in INTENTS:
        if re.search(pattern, t):
            if name == "factor" and not any(re.search(p, t) for p in FEATURE_WORDS.values()):
                continue
            return name
    return "unknown"


# ---------------------------------------------------------------------------------------------- replies
def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {x}" for x in items)


class Assistant:
    """Answers one message about one profile. Holds references to the loaded model objects only."""

    def __init__(self, predictor: Predictor, explainer: Explainer, reference=None):
        self.predictor, self.explainer, self.reference = predictor, explainer, reference

    def respond(self, message: str, profile: dict | None) -> dict:
        message = normalize_chat_text((message or "").strip()[:MAX_MESSAGE_CHARS])
        intent = classify(message)
        if profile is None and intent not in PROFILE_FREE_INTENTS:
            reply, sources = ("Complete your profile first so I can answer based on your own estimate.", [])
        else:
            handler = getattr(self, f"_{intent}")
            try:
                reply, sources = handler(message, profile)
            except (WhatIfError, ValueError) as e:
                reply, sources = f"I couldn't work that out: {e}", []
        return {"intent": intent, "reply": reply, "sources": sources,
                "follow_ups": [q for q in (GENERAL_FOLLOW_UPS if profile is None else FOLLOW_UPS)
                               if q.lower() != message.lower()][:4]}

    # -- intents -------------------------------------------------------------------------------------
    def _estimate(self, message, profile):
        p = self.predictor.predict(profile)
        lines = [p["headline"], "",
                 f"Confidence: **{p['confidence']}** - {p['confidence_reason']}"]
        if p.get("milestones"):
            lines += ["", _bullets([m["text"] for m in p["milestones"]][:3])]
        if p.get("cautions"):
            lines += ["", _bullets(p["cautions"])]
        return "\n".join(lines), ["prediction"]

    def _explain(self, message, profile):
        e = self.explainer.explain(profile)
        lines = []
        if e["helping"]:
            lines += ["**What's helping:**", _bullets([i["text"] for i in e["helping"]])]
        if e["hurting"]:
            lines += ([""] if lines else []) + ["**What's holding the estimate back:**",
                                                 _bullets([i["text"] for i in e["hurting"]])]
        elif e["helping"]:
            lines += ([""] if lines else []) + [wording.LOCAL_NO_HURTING]
        if not lines:
            lines = [wording.LOCAL_NO_HURTING]
        lines += ["", e["note"]]
        return "\n".join(lines), ["explanation"]

    def _factor(self, message, profile):
        t = message.lower()
        feats = [f for f, p in FEATURE_WORDS.items() if re.search(p, t)]
        global_effects = (self.explainer.global_effects() if profile is None
                          else self.explainer.explain(profile)["global"])
        effects = [g for g in global_effects if g["feature"] in feats]
        if not effects:
            return self._help(message, profile)
        return ("Across all graduates in the training data:\n" + _bullets([g["text"] for g in effects]) +
                "\n\n" + wording.ASSOCIATIONAL_NOTE), ["global effects"]

    def _whatif(self, message, profile):
        status = profile.get("respondent_status", "graduated")
        changes, notes = parse_changes(message, current_values(profile), status)
        if not changes:
            return ("I can test changes to internship months, certifications, projects, skill ratings (and GPA for "
                    "final-year students). Try: *\"What if I do a 6-month internship?\"* or *\"What if I get two "
                    "certifications?\"*"), []
        try:
            out = what_if(self.predictor, profile, changes)
        except WhatIfError as e:
            if "final-year" in str(e):
                return ("GPA is fixed once you have graduated, so I can't test a GPA change for you. I can test "
                        "internships, certifications, projects and skill ratings instead."), []
            raise
        current = current_values(profile)
        applied = [f"{schema.ACTIONABLE_FEATURES[k].label}: {current.get(k) or 0:g} -> {v:g}"
                   for k, v in out["applied_changes"].items()]
        lines = ["**Change tested:** " + "; ".join(applied), "", out["message"]]
        lines += [""] + [_bullets(notes)] if notes else []
        if out["cautions"]:
            lines += ["", _bullets(out["cautions"])]
        lines += ["", wording.ASSOCIATIONAL_NOTE]
        return "\n".join(lines), ["what-if"]

    def _suggestions(self, message, profile):
        s = suggest(self.predictor, profile)
        if not s["suggestions"]:
            return s["message"] or wording.NO_SUGGESTIONS, ["suggestions"]
        return ("**Changes linked with finding work sooner, for profiles like yours:**\n" +
                _bullets([x["text"] for x in s["suggestions"]]) + "\n\n" + s["note"]), ["suggestions"]

    def _skills(self, message, profile):
        if self.reference is None:
            return "Employer and job-advert data are not loaded.", []
        m = self.reference.alignment(profile)
        if not m["top_gaps"]:
            return m["message"] or wording.MARKET_NO_GAPS, ["market alignment"]
        items = []
        for g in m["top_gaps"]:
            res = "; ".join(f"[{r['title']}]({r['url']}) ({r['provider']})" for r in g.get("resources", [])[:2])
            items.append(g["text"] + (f" Free resources: {res}" if res else ""))
        return "**Your biggest gaps against what employers and job ads ask for:**\n" + _bullets(items) + \
            "\n\n" + m["note"], ["employer survey", "job adverts"]

    def _peers(self, message, profile):
        if self.reference is None:
            return "Peer data are not loaded.", []
        pb = self.reference.peer_benchmark(profile)
        return (_bullets([f"**{g['group_label']} - {g['level_label']}** (n = {g['n']}): {g['text']}"
                          for g in pb["groups"]]) + "\n\n" + pb["note"]), ["peer benchmarks"]

    def _trust(self, message, profile):
        b = self.predictor.bundle
        test = b.get("test_metrics")
        if profile is None:
            if test:
                return ("The model was tested on graduates it had not learned from. Its results are estimates, not "
                        "guarantees. Complete your profile to see the confidence for your own estimate."), [
                            "model evaluation"]
            return ("The model gives an estimate, not a guarantee. Complete your profile to see the confidence for "
                    "your own result."), ["model evaluation"]
        lines = []
        if test:
            from core.evaluate import plain_language_metrics
            lines += ["**How accurate the model was on graduates it had never seen:**",
                      _bullets(plain_language_metrics(test["uno_c"], {h: test[f"cal_err_{h}m"] for h in (3, 6, 12)}))]
        p = self.predictor.predict(profile)
        lines += ["", f"For your profile the confidence is **{p['confidence']}**: {p['confidence_reason']}"]
        lines += ["", "**Limits to keep in mind:**",
                  _bullets(wording.limitations(self.predictor.bundle.get("dataset_info", {})))]
        return "\n".join(lines), ["model evaluation"]

    def _method(self, message, profile):
        return ("JobSpark uses survey data from past IT graduates to estimate how long finding a first IT job may "
                "take. It gives an estimate, not a guarantee."), ["model information"]

    def _privacy(self, message, profile):
        return ("Your profile answers are used to calculate your results. Your chat messages are not saved to your "
                "account."), []

    def _greeting(self, message, profile):
        time_greeting = re.search(r"\bgood (morning|afternoon|evening)\b", message, re.IGNORECASE)
        salutation = f"Good {time_greeting.group(1).lower()}!" if time_greeting else "Hi!"
        if re.match(r"^(thanks|thank you)\b", message, re.IGNORECASE):
            salutation = "You're welcome!"
        if profile is None:
            return (f"{salutation} I'm Spark. Ask me how JobSpark works or how your data is used. Complete your "
                    "profile for answers about your own estimate."), []
        return (f"{salutation} I'm here to help with your estimate, what affects it, or what you could try."), []

    def _help(self, message, profile):
        if profile is None:
            return ("Try asking how JobSpark works or whether your chat is saved. Complete your profile for personal "
                    "estimates and suggestions."), []
        return ("I answer questions about your estimate using this system's own results. You can ask:\n" +
                _bullets(FOLLOW_UPS + ["Is GPA important?", "How do I compare with similar graduates?"])), []

    def _unknown(self, message, profile):
        if profile is None:
            return ("I'm Spark, JobSpark's helper. I can answer questions about the model, privacy, and your "
                    "employment estimate. Try asking, *\"How does JobSpark work?\"*"), []
        return ("I can only answer questions about your time-to-job estimate, the factors behind it, what-if "
                "changes, skills employers want, and how reliable the model is. I'm not a general chatbot, so I "
                "won't guess about anything outside the data.\n\n" + self._help(message, profile)[0]), []
