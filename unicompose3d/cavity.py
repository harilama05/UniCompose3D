"""Category-independent open-mouth, closed-end cavity measurements."""
from __future__ import annotations
import numpy as np
from . import aperture,geometry

def _frame(mesh,origin,axis,size,region_center=None,max_distance=None):
    loops=aperture.sections(mesh,origin,axis)
    pairs=[]
    for inner in loops:
        for outer in loops:
            if inner is outer or inner["radius"]>=outer["radius"]*.985:continue
            separation=np.linalg.norm(inner["center"]-outer["center"])
            if separation>inner["radius"]*.2 or inner["roundness"]>.4:continue
            if inner["min_radius"]<size*.025:continue
            point=inner["center"]
            if region_center is not None and max_distance is not None and np.linalg.norm(point-region_center)>max_distance:continue
            # Empty path to the exterior, finite end inward: not a through-hole.
            if geometry.ray_hit(mesh,point-axis*size*1e-5,-axis) is not None:continue
            roof=geometry.ray_hit(mesh,point+axis*size*1e-5,axis)
            if roof is None:continue
            depth=float(np.dot(roof[0]-point,axis))
            if depth<inner["radius"]*.15 or depth>size*1.5:continue
            pairs.append(((-np.linalg.norm(point-region_center) if region_center is not None else inner["radius"]),{"center":point,"axis":axis,
                "inner_radius":inner["min_radius"],"outer_radius":outer["max_radius"],
                "depth":depth,"roof":roof[0],"roundness":inner["roundness"]}))
    return max(pairs,key=lambda item:item[0])[1] if pairs else None

def cavity_frame(mesh, preferred_axis=None,region_center=None,max_distance=None):
    region_center=np.asarray(region_center) if region_center is not None else None
    size=float(max(mesh.extents));vertices=np.asarray(mesh.vertices)
    def measure(origin,axis):return _frame(mesh,origin,axis,size,region_center,max_distance)
    directions=[]
    if preferred_axis is not None and np.linalg.norm(preferred_axis)>1e-8:
        directions.append(np.asarray(preferred_axis)/np.linalg.norm(preferred_axis))
    _,principal=np.linalg.eigh(np.cov(vertices.T))
    for axis in [*np.eye(3),*principal.T]:
        if not any(abs(np.dot(axis,previous))>.995 for previous in directions):directions.append(axis)
    for direction in directions:
        for sign in (1,-1):
            axis=direction*sign
            projected=vertices@axis;low,high=float(projected.min()),float(projected.max())
            center=mesh.bounds.mean(axis=0);center=center-axis*np.dot(center,axis)
            previous=.015
            for fraction in (.08,.14,.22,.34,.48,.62):
                origin=center+axis*(low+(high-low)*fraction)
                frame=measure(origin,axis)
                if frame is None:previous=fraction;continue
                left,right=previous,fraction
                for _ in range(4):
                    middle=(left+right)/2
                    trial=measure(center+axis*(low+(high-low)*middle),axis)
                    if trial is None:left=middle
                    else:right=middle;frame=trial
                # Move a small distance inside the rim to avoid a numerically
                # degenerate slice exactly along a boundary edge.
                inset=min(frame["depth"]*.035,size*.012)
                interior=measure(frame["center"]+axis*inset,axis)
                if interior is None:continue
                frame=interior
                check=measure(frame["center"]+axis*min(frame["depth"]*.12,size*.04),axis)
                if check is None:continue
                return {**frame,"axial_span":frame["depth"],
                    "topology":"open-mouth-closed-end","mouth_verified":True,
                    "measurement":"nested wall sections, clear exterior ray, finite inner end",
                    "watertight":bool(mesh.is_watertight)}
    error=ValueError("No verified open-mouth cavity with a measured closed end was found.")
    error.measurements={"required_topology":"open mouth with internal wall and closed end",
        "through_aperture_required":False}
    raise error
