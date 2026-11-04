"""Structured feature failures: preserve successful tool measurements too."""
import numpy as np

def serializable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {key:serializable(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [serializable(item) for item in value]
    return value

class FeatureResolutionError(ValueError):
    def __init__(self,issues,resolved):
        super().__init__(" ".join(issue["error"] for issue in issues))
        self.measurements={"feature_issues":serializable(issues),"resolved_features":serializable(resolved)}
