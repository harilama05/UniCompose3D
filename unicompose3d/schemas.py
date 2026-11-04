from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
import math


UnitFloat = Annotated[float, Field(ge=0, le=1)]
DegreeFloat = Annotated[float, Field(ge=-360, le=360)]
OffsetFloat = Annotated[float, Field(ge=-2, le=2)]


class Transform(BaseModel):
    position: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0], min_length=3, max_length=3)
    rotation: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0], min_length=3, max_length=3)
    scale: list[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0], min_length=3, max_length=3)

    @field_validator("position", "rotation", "scale")
    @classmethod
    def finite(cls, value, info):
        if not all(math.isfinite(v) and abs(v) < 1e5 for v in value):
            raise ValueError("Invalid transform.")
        if info.field_name == "scale" and not all(0.001 <= v <= 1000 for v in value):
            raise ValueError("Scale must be between 0.001 and 1000.")
        return value




class GenerationRequest(BaseModel):
    source_id: str
    name: str = Field(default="Object", max_length=100)
    views: dict[str, str] = Field(default_factory=dict)
    selected_images: list[str] = Field(default_factory=list, max_length=16)
    force_new: bool = False
    provider: Literal["hunyuan"] = "hunyuan"
    use_cutouts: bool = False
    confirm_cutouts: bool = False
    texture: bool = True
    target: str = Field(default="", max_length=200)
    cutout_files: dict[str, str] = Field(default_factory=dict)


class TargetLocation(BaseModel):
    found: bool
    confidence: float = Field(ge=0, le=1)
    box: list[float] = Field(min_length=4, max_length=4)
    explanation: str = Field(max_length=1000)

    @field_validator("box")
    @classmethod
    def normalized_box(cls, value):
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in value) or value[0] >= value[2] or value[1] >= value[3]:
            raise ValueError("The box must be left, top, right, bottom in [0,1].")
        return value


class SegmentRequest(BaseModel):
    filename: str = Field(max_length=100)
    target: str = Field(min_length=1, max_length=200)
    box: list[float] | None = Field(default=None, min_length=4, max_length=4)
    points: list[list[float]] = Field(default_factory=list, max_length=32)
    labels: list[Literal[0, 1]] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def prompts(self):
        if self.box is not None:
            TargetLocation(found=True, confidence=1, box=self.box, explanation="user")
        if len(self.points) != len(self.labels):
            raise ValueError("Each point requires a keep/remove label.")
        if any(len(p) != 2 or not all(math.isfinite(v) and 0 <= v <= 1 for v in p) for p in self.points):
            raise ValueError("The selected point is outside the image.")
        return self


class MaskStroke(BaseModel):
    points: list[list[UnitFloat]] = Field(min_length=1, max_length=512)
    radius: float = Field(ge=0.001, le=0.15)
    keep: bool = False
    mode: Literal["pixels", "region"] = "pixels"
    tolerance: int = Field(default=18, ge=3, le=45)

    @field_validator("points")
    @classmethod
    def pairs(cls, value):
        if any(len(point) != 2 for point in value):
            raise ValueError("Each brush point needs two coordinates.")
        return value


class MaskEditRequest(BaseModel):
    filename: str = Field(max_length=100)
    cutout_file: str = Field(max_length=100)
    strokes: list[MaskStroke] = Field(default_factory=list, max_length=64)
    undo: bool = False

    @model_validator(mode="after")
    def action(self):
        if self.undo == bool(self.strokes):
            raise ValueError("Choose undo or provide a brush stroke.")
        return self


class CutoutExportRequest(BaseModel):
    files: list[str] = Field(min_length=1, max_length=16)
    target: str = Field(min_length=1, max_length=200)
    cutout_files: dict[str, str]
    confirmed: bool = False


class Capture(BaseModel):
    slot: Literal["A", "B"]
    image: str = Field(max_length=4_000_000)
    projection: list[float] = Field(min_length=16, max_length=16)
    world: list[float] = Field(min_length=16, max_length=16)

    @field_validator("projection", "world")
    @classmethod
    def finite_matrix(cls, value):
        if not all(math.isfinite(v) and abs(v) < 1e9 for v in value):
            raise ValueError("Invalid camera.")
        return value


class RegionHints(BaseModel):
    A_point: list[float] | None = Field(default=None, min_length=3, max_length=3)
    B_point: list[float] | None = Field(default=None, min_length=3, max_length=3)
    A_end: list[float] | None = Field(default=None, min_length=3, max_length=3)
    B_end: list[float] | None = Field(default=None, min_length=3, max_length=3)

    @field_validator("A_point", "B_point", "A_end", "B_end")
    @classmethod
    def finite(cls, value):
        if value is not None and not all(math.isfinite(v) and abs(v) < 1e5 for v in value):
            raise ValueError("Invalid selected region or direction.")
        return value


class SpatialFeature(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,31}$")
    slot: Literal["A", "B"]
    kind: Literal["point", "surface", "axis", "section", "aperture", "cavity", "region"]
    source: Literal["model", "user"] = "model"
    point_ref: str | None = Field(default=None,pattern=r"^L[0-9]{2}$")
    end_ref: str | None = Field(default=None,pattern=r"^L[0-9]{2}$")
    point: list[UnitFloat] = Field(default_factory=lambda: [0.5, 0.5, 0.5], min_length=3, max_length=3)
    end: list[UnitFloat] | None = Field(default=None, min_length=3, max_length=3)
    extent: list[UnitFloat] = Field(default_factory=lambda: [1, 1, 1], min_length=3, max_length=3)
    view: int = Field(default=-1, ge=-1, le=7)
    uv: list[UnitFloat] | None = Field(default=None, min_length=2, max_length=2)
    end_view: int = Field(default=-1, ge=-1, le=7)
    end_uv: list[UnitFloat] | None = Field(default=None, min_length=2, max_length=2)

    @model_validator(mode="after")
    def opening_centers_are_volumetric(self):
        if self.kind in {"aperture","cavity"} and (self.point_ref is not None or self.end_ref is not None):
            raise ValueError("Surface landmark IDs cannot define an empty opening centre; use local bbox coordinates.")
        return self

    @field_validator("point", "end", "extent", "uv", "end_uv")
    @classmethod
    def unit_values(cls, value):
        if value is not None and not all(math.isfinite(v) and 0 <= v <= 1 for v in value):
            raise ValueError("Feature parameters must be in [0, 1].")
        return value


class SpatialConstraint(BaseModel):
    kind: Literal["coincident", "contact", "align", "clearance", "containment", "offset", "proportion"]
    a: str = Field(max_length=32)
    b: str = Field(max_length=32)
    weight: float = Field(default=1, ge=0.1, le=5)
    tolerance: float = Field(default=0.02, ge=0.005, le=0.2)
    target: float = Field(default=0, ge=0, le=5)
    direction: Literal["parallel", "opposed", "either"] = "either"
    frame_a: str | None = Field(default=None,max_length=32)
    frame_b: str | None = Field(default=None,max_length=32)
    offset: list[OffsetFloat] = Field(default_factory=lambda: [0, 0, 0], min_length=3, max_length=3)

    @field_validator("offset")
    @classmethod
    def finite_offset(cls, value):
        if not all(math.isfinite(v) and abs(v) <= 2 for v in value):
            raise ValueError("Constraint offset is out of range.")
        return value


class SpatialProgram(BaseModel):
    features: list[SpatialFeature] = Field(min_length=2, max_length=12)
    constraints: list[SpatialConstraint] = Field(min_length=1, max_length=16)
    fit_topology: Literal["auto", "through", "cover"] = "auto"
    deformation_required: bool = False
    uncertainty: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def valid_references(self):
        features = {feature.id: feature for feature in self.features}
        if len(features) != len(self.features):
            raise ValueError("Duplicate feature ID.")
        for constraint in self.constraints:
            a, b = features.get(constraint.a), features.get(constraint.b)
            if a is None or b is None or a.slot != "A" or b.slot != "B":
                raise ValueError("Each constraint must reference existing A and B features.")
            if constraint.frame_a is not None or constraint.frame_b is not None:
                fa,fb=features.get(constraint.frame_a),features.get(constraint.frame_b)
                if constraint.kind!="align" or fa is None or fb is None or fa.slot!="A" or fb.slot!="B" or fa.kind in {"point","region"} or fb.kind in {"point","region"}:
                    raise ValueError("Planar heading needs an existing axis frame on each matching slot.")
            if constraint.kind == "contact" and (a.kind != "surface" or b.kind != "surface"):
                raise ValueError("Contact requires two surface features.")
            if constraint.kind == "clearance" and (not ("section" in {a.kind,b.kind} and bool({a.kind,b.kind}&{"aperture","cavity"})) or constraint.target < 1.015):
                raise ValueError("Clearance requires a section and opening on different objects, with target >= 1.015.")
            if constraint.kind == "containment" and a.kind != "region":
                raise ValueError("Containment requires an A region.")
        return self


class SearchBudget(BaseModel):
    max_starts: int = Field(default=16,ge=2,le=48)
    max_seconds: float = Field(default=60,ge=5,le=180)
    max_programs: int = Field(default=4,ge=1,le=8)

class ComposeRequest(BaseModel):
    base_id: str
    product_id: str
    base_transform: Transform = Field(default_factory=Transform)
    instruction: str = Field(default="", max_length=1500)
    captures: list[Capture] = Field(default_factory=list, max_length=8)
    use_cache: bool = True
    candidate_count: int = Field(default=1, ge=1, le=5)
    search_budget: SearchBudget = Field(default_factory=SearchBudget)
    hints: RegionHints = Field(default_factory=RegionHints)
    refinement_of: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    clarification_of: str | None = Field(default=None,pattern=r"^[a-f0-9]{32}$")
    clarification_answer: str = Field(default="",max_length=500)



class AgentQuestion(BaseModel):
    question: str = Field(min_length=5,max_length=300)
    options: list[str] = Field(min_length=2,max_length=4)
    @field_validator("options")
    @classmethod
    def simple_options(cls,value):
        if len(set(value))!=len(value) or any(not x.strip() or len(x)>140 for x in value):
            raise ValueError("Question options must be distinct short choices.")
        return value

class SemanticIntent(BaseModel):
    target_region: str = Field(max_length=200)
    moving_region: str = Field(max_length=200)
    spatial_relationship: str = Field(max_length=200)
    orientation: str = Field(max_length=200)
    avoid: list[str] = Field(default_factory=list,max_length=5)

class SpatialHypothesis(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,31}$")
    reason: str = Field(max_length=400)
    program: SpatialProgram

class PlacementPlan(BaseModel):
    relation: str = Field(min_length=1, max_length=80)
    base_anchor: list[UnitFloat] = Field(min_length=3, max_length=3)
    product_anchor: list[UnitFloat] = Field(min_length=3, max_length=3)
    rotation_degrees: list[DegreeFloat] = Field(min_length=3, max_length=3)
    relative_scale: float = Field(ge=0.02, le=5)
    clearance: float = Field(default=0.005, ge=-0.1, le=0.2)
    confidence: float = Field(ge=0, le=1)
    explanation: str = Field(max_length=2000)
    deform: bool = False
    base_view: int = Field(default=-1, ge=-1, le=7)
    base_uv: list[UnitFloat] | None = Field(default=None, min_length=2, max_length=2)
    product_view: int = Field(default=-1, ge=-1, le=7)
    product_uv: list[UnitFloat] | None = Field(default=None, min_length=2, max_length=2)
    fit_mode: Literal["generic", "aperture"] = "generic"
    base_axis_end: list[UnitFloat] | None = Field(default=None, min_length=3, max_length=3)
    base_axis_end_view: int = Field(default=-1, ge=-1, le=7)
    base_axis_end_uv: list[UnitFloat] | None = Field(default=None, min_length=2, max_length=2)
    program: SpatialProgram | None = None
    clarification: AgentQuestion | None = None
    intent: SemanticIntent | None = None
    hypotheses: list[SpatialHypothesis] = Field(default_factory=list,max_length=3)

    @field_validator("base_anchor", "product_anchor")
    @classmethod
    def unit_points(cls, value):
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in value):
            raise ValueError("Anchor points must be in [0, 1].")
        return value

    @field_validator("rotation_degrees")
    @classmethod
    def angles(cls, value):
        if not all(math.isfinite(v) and abs(v) <= 360 for v in value):
            raise ValueError("Invalid rotation.")
        return value

    @field_validator("base_axis_end", "base_uv", "product_uv", "base_axis_end_uv")
    @classmethod
    def optional_unit_points(cls, value):
        if value is not None and not all(math.isfinite(v) and 0 <= v <= 1 for v in value):
            raise ValueError("Direction/image coordinates must be in [0, 1].")
        return value


class ApertureFitRequest(BaseModel):
    base_id: str
    product_id: str
    base_transform: Transform = Field(default_factory=Transform)
    product_transform: Transform = Field(default_factory=Transform)
    base_point: list[float] = Field(min_length=3, max_length=3)
    axis_end: list[float] = Field(min_length=3, max_length=3)

    @field_validator("base_point", "axis_end")
    @classmethod
    def finite_points(cls, value):
        if not all(math.isfinite(v) and abs(v) < 1e5 for v in value):
            raise ValueError("Invalid direction point.")
        return value

class CandidateRender(BaseModel):
    candidate_id: str = Field(pattern=r"^c[0-4]$")
    image: str = Field(max_length=4_000_000)
    layout: Literal["front-side-back-oblique-v1","front-side-back-oblique-contact-v2"] = "front-side-back-oblique-v1"


class EvaluateRequest(BaseModel):
    renders: list[CandidateRender] = Field(default_factory=list, max_length=5)


class CandidateReview(BaseModel):
    candidate_id: str = Field(pattern=r"^c[0-4]$")
    intent_satisfied: bool | None = None
    failure_cause: Literal["none","grounding","direction","position","scale","shape","occlusion","unknown"] = "unknown"
    repair_action: str = Field(default="",max_length=500)
    acceptable: bool
    contact_score: float = Field(ge=0, le=1)
    placement_score: float = Field(ge=0, le=1)
    proportion_score: float = Field(ge=0, le=1)
    appearance_score: float = Field(ge=0, le=1)
    explanation: str = Field(max_length=1200)
    issues: list[str] = Field(default_factory=list, max_length=6)

    @field_validator("issues")
    @classmethod
    def short_issues(cls, value):
        if any(len(v) > 300 for v in value):
            raise ValueError("Review text is too long.")
        return value


class VisualEvaluation(BaseModel):
    reviews: list[CandidateReview] = Field(min_length=1, max_length=5)
    summary: str = Field(max_length=2000)


class SceneRequest(BaseModel):
    base_id: str
    product_id: str
    base_transform: Transform = Field(default_factory=Transform)
    product_transform: Transform = Field(default_factory=Transform)
    title: str = Field(default="Composition scene", max_length=100)
    notes: str = Field(default="", max_length=5000)
    hints: RegionHints = Field(default_factory=RegionHints)


class AnchorRequest(BaseModel):
    base_id: str
    product_id: str
    base_point: list[float] = Field(min_length=3, max_length=3)
    product_point: list[float] = Field(min_length=3, max_length=3)
    base_transform: Transform
    product_transform: Transform

    @field_validator("base_point", "product_point")
    @classmethod
    def finite_point(cls, value):
        if not all(math.isfinite(v) and abs(v) < 1e5 for v in value):
            raise ValueError("Invalid anchor point.")
        return value


class DeformRequest(BaseModel):
    asset_id: str
    point: list[float] = Field(min_length=3, max_length=3)
    delta: list[float] = Field(min_length=3, max_length=3)
    radius: float = Field(default=0.3, ge=0.01, le=2)
    name: str = Field(default="Sculpted product", max_length=100)

    @field_validator("point", "delta")
    @classmethod
    def bounded(cls, value):
        if not all(math.isfinite(v) and abs(v) <= 5 for v in value):
            raise ValueError("Deformation exceeds the allowed limit.")
        return value





