"""Pydantic request/response models. Ranges and categories come from core/schema.py (single source of truth)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from core import schema

_RAW = schema.RAW_FIELDS_BY_NAME


def _range(name: str, default=None, **kw):
    f = _RAW[name]
    return Field(default, ge=f.min, le=f.max, description=f"{f.description} ({f.item})", **kw)


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------
_likert = {c: (Optional[float], Field(None, ge=1, le=5, description=_RAW[c].description))
           for c in (*schema.TECH_COLUMNS, *schema.SOFT_COLUMNS)}
_areas = {f"area_{a}": (Optional[float], Field(None, ge=1, le=5, description=f"Quick-mode rating: {v['label']}"))
          for a, v in {**schema.SKILL_AREAS, **schema.SOFT_AREA}.items()}

_ProfileBase = create_model(
    "_ProfileBase",
    __config__=ConfigDict(extra="forbid"),
    respondent_status=(Literal[schema.RESPONDENT_STATUS], Field("graduated", description="graduated or final_year")),
    gpa=(Optional[float], _range("gpa")),
    degree_class=(Optional[Literal[schema.DEGREE_CLASSES]], Field(None, description="Only if the degree has no GPA")),
    degree_type=(Optional[Literal[schema.DEGREE_TYPES]], Field(None, description=_RAW["degree_type"].description)),
    university_type=(Optional[Literal[schema.UNIVERSITY_TYPES]], Field(None, description="Type of institution")),
    certifications_count=(Optional[float], _range("certifications_count")),
    internship_completed=(Optional[Literal["yes", "no"]], Field(None)),
    internship_months=(Optional[float], _range("internship_months")),
    project_count=(Optional[float], _range("project_count")),
    # Preparation before completing the degree (extended feature set; optional - imputed when missing)
    has_portfolio=(Optional[Literal["yes", "no"]], Field(None, description=_RAW["has_portfolio"].description)),
    prep_start=(Optional[Literal[schema.PREP_START]], Field(None, description=_RAW["prep_start"].description)),
    self_learning_freq=(Optional[float], _range("self_learning_freq")),
    hackathon_count=(Optional[float], _range("hackathon_count")),
    open_source=(Optional[Literal["yes", "no"]], Field(None, description=_RAW["open_source"].description)),
    sdlc_familiarity=(Optional[float], _range("sdlc_familiarity")),
    months_since_graduation=(Optional[float], Field(None, ge=0, le=36, description=(
        "Graduates still looking for their first IT job: months since degree completion. Makes the prediction "
        "conditional on not having found work yet."))),
    **_likert,
    **_areas,
)


class Profile(_ProfileBase):
    """A graduate or final-year profile. Give skill items individually (Detailed mode) or one rating per area
    (Quick mode); an individual item always overrides its area rating. Unknown fields are rejected."""

    @model_validator(mode="after")
    def _months_only_for_graduates(self):
        if self.months_since_graduation and self.respondent_status != "graduated":
            raise ValueError("months_since_graduation only applies to graduates who are still looking")
        return self

    def to_core(self) -> dict:
        return self.model_dump(exclude_none=True)


class WhatIfRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profile: Profile
    changes: dict[str, float] = Field(..., description="Actionable features only: "
                                                       + ", ".join(schema.ACTIONABLE_FEATURES))


class Extras(BaseModel):
    """Optional preparation answers. They tailor the survey insights only - they never reach the model."""
    model_config = ConfigDict(extra="forbid")
    has_portfolio: Optional[bool] = Field(None, description="Has a portfolio (GitHub / website)")
    hackathons: Optional[bool] = Field(None, description="Has taken part in a hackathon or competition")
    open_source: Optional[bool] = Field(None, description="Has contributed to open source")
    self_learning: Optional[bool] = Field(None, description="Self-learns often outside coursework")
    early_prep: Optional[bool] = Field(None, description="Started preparing for IT jobs by the 2nd/3rd year")


class InsightsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profile: Profile
    extras: Extras = Extras()


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------
class MonthValue(BaseModel):
    months: Optional[int]
    more_than: bool
    display: str


class Uncertainty(BaseModel):
    median_interval_80: list[MonthValue]
    width_months: Optional[float]
    n_bootstrap: int
    share_beyond_horizon: float


class RemainingMonths(BaseModel):
    """Months from now (only for graduates still looking)."""
    low: MonthValue
    median: MonthValue
    high: MonthValue


class Curve(BaseModel):
    months: list[int]
    still_searching: list[float]


class Milestone(BaseModel):
    """Chance of having found an IT job by a given month (80% bootstrap range)."""
    months: int
    employed_pct: int
    low_pct: int
    high_pct: int
    text: str


class PredictionResponse(BaseModel):
    low: MonthValue
    median: MonthValue
    high: MonthValue
    uncertainty: Uncertainty
    confidence: Literal["High", "Medium", "Low"]
    confidence_reason: str
    status: str
    time_reference: str
    headline: str
    horizon_months: int
    months_since_graduation: Optional[int] = None
    remaining: Optional[RemainingMonths] = None
    cautions: list[str] = []
    curve: Optional[Curve] = None
    milestones: list[Milestone] = []
    is_synthetic: bool = True


class GlobalEffect(BaseModel):
    feature: str
    level: Optional[str]
    column: str
    kind: Literal["hazard_ratio", "time_ratio", "shap"]
    step: Optional[float] = None
    ratio: Optional[float] = None
    lower: Optional[float] = None
    upper: Optional[float] = None
    mean_abs_shap: Optional[float] = None
    direction: Literal["sooner", "later"]
    uncertain: bool
    text: str


class LocalItem(BaseModel):
    feature: str
    name: str
    value: str
    score: float
    strength: str
    side: Literal["helping", "hurting"]
    text: str


class ExplainResponse(BaseModel):
    global_effects: list[GlobalEffect]
    helping: list[LocalItem]
    hurting: list[LocalItem]
    note: str
    method: str


class ChangeMonths(BaseModel):
    low: Optional[int]
    median: Optional[int]
    high: Optional[int]


class WhatIfResponse(BaseModel):
    original: PredictionResponse
    changed: PredictionResponse
    change_months: ChangeMonths
    message: str
    applied_changes: dict[str, float]
    cautions: list[str]


class LearningResource(BaseModel):
    title: str
    provider: str
    url: str


class SkillAlignment(BaseModel):
    skill: str
    label: str
    kind: Literal["technical", "soft"]
    area: str
    your_rating: Optional[float]
    employer_importance: Optional[float]
    ads_pct: Optional[float]
    demand: Optional[float]
    gap: Optional[float]
    text: Optional[str] = None
    resources: list[LearningResource] = []


class MarketResponse(BaseModel):
    skills: list[SkillAlignment]
    top_gaps: list[SkillAlignment]
    message: Optional[str]
    note: str


class Suggestion(BaseModel):
    changes: dict[str, float]
    effort: int
    months_sooner: int
    new_median_months: Optional[int]
    text: str


class SuggestionsResponse(BaseModel):
    suggestions: list[Suggestion]
    message: Optional[str]
    note: str


class PeerGroup(BaseModel):
    group: str
    group_label: str
    level: str
    level_label: str
    n: int
    suppressed: bool
    median: Optional[MonthValue] = None
    median_interval: Optional[list[str]] = None
    employed_by_6m: Optional[float] = None
    text: str


class PeerResponse(BaseModel):
    groups: list[PeerGroup]
    note: str
    min_group_n: int


class ModelMetrics(BaseModel):
    model: str
    label: str
    uno_c_mean: float
    uno_c_sd: float
    harrell_c_mean: float
    ibs_mean: float
    cal_err_3m: float
    cal_err_6m: float
    cal_err_12m: float
    selected: bool


class ModelInfoResponse(BaseModel):
    is_synthetic: bool
    training_n: int
    events: int
    censored: int
    final_year_n: int
    employers_n: int
    job_ads_n: int
    selected_model: str
    selected_model_label: str
    selection_reason: str
    hyperparameters: dict
    horizon_months: int
    cv_metrics: list[ModelMetrics]                 # validation-set comparison (one row per model)
    test_metrics: Optional[dict] = None             # the selected model on the locked test set (reported result)
    modelling_n: int = 0
    validation_n: int = 0
    test_n: int = 0
    confidence_thresholds_months: dict[str, float]
    plain_metrics: list[str]
    feature_set: str
    disclaimer: str
    synthetic_banner: Optional[str]
    associational_note: str
    privacy: str
    limitations: list[str]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    bootstrap_models: int
    is_synthetic: bool
