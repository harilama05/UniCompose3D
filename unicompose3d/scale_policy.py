"""Category-independent scale evidence and inferred geometric fit constraints."""
import numpy as np
from .schemas import SpatialConstraint

class ScaleEvidenceError(ValueError):
    pass

def complete_fit_constraints(program):
    program = program.model_copy(deep=True)
    features = {feature.id:feature for feature in program.features}
    pairs = {(constraint.a,constraint.b) for constraint in program.constraints}
    additions=[]
    for a_id,b_id in sorted(pairs):
        a,b=features[a_id],features[b_id]
        if not ("section" in {a.kind,b.kind} and bool({a.kind,b.kind}&{"aperture","cavity"})):
            continue
        kinds={constraint.kind for constraint in program.constraints if (constraint.a,constraint.b)==(a_id,b_id)}
        # These measured features define a fit; position/alignment alone cannot
        # specify whether their sizes are compatible.
        for kind in ("coincident","align","clearance"):
            if kind not in kinds:
                additions.append(SpatialConstraint(kind=kind,a=a_id,b=b_id,
                    target=1.05 if kind=="clearance" else 0))
    if len(program.constraints)+len(additions)>16:
        raise ScaleEvidenceError('Too many constraints after scale completion; simplify the plan.')
    program.constraints.extend(additions)
    return program, [constraint.model_dump() for constraint in additions]

def scale_evidence(program,features):
    explicit=[constraint.kind for constraint in program.constraints
              if constraint.kind in {"clearance","proportion"}]
    if explicit:
        return {"verified":True,"source":"measured constraints","constraints":explicit}
    landmarks=[constraint for constraint in program.constraints
               if constraint.kind=="coincident"]
    for i,one in enumerate(landmarks):
        for two in landmarks[i+1:]:
            a=np.linalg.norm(features[one.a]["center"]-features[two.a]["center"])
            b=np.linalg.norm(features[one.b]["center"]-features[two.b]["center"])
            if min(a,b)>1e-6:
                return {"verified":True,"source":"multiple distinct landmarks"}
    return {"verified":False,"source":"LLM size prior only"}

def landmark_scales(program,features):
    landmarks=[constraint for constraint in program.constraints if constraint.kind=="coincident"]
    ratios=[]
    for i,one in enumerate(landmarks):
        for two in landmarks[i+1:]:
            a=np.linalg.norm(features[one.a]["center"]-features[two.a]["center"])
            b=np.linalg.norm(features[one.b]["center"]-features[two.b]["center"])
            if min(a,b)>1e-6:ratios.append(a/b)
    return ratios
