"""Generic aperture/target fitting from mesh cross sections, never sizing.

Verify nested loops or a radially supported partial opening in B. Select A's
local cross section in the requested region, then align the hole axis and
center. Rings, bracelets and tubular attachments share these constraints.
An unsupported topology fails explicitly instead of accepting surface contact.
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from . import geometry
from .schemas import Transform


def plane_basis(axis):
    axis = np.asarray(axis, dtype=float)
    length = np.linalg.norm(axis)
    if length < 1e-8:
        raise ValueError("The two direction points on A are too close.")
    axis = axis / length
    helper = np.eye(3)[np.argmin(np.abs(axis))]
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    return axis, np.column_stack([u, np.cross(axis, u)])


def sections(mesh, origin, axis):
    axis, basis = plane_basis(axis)
    cut = mesh.section(plane_origin=origin, plane_normal=axis)
    if cut is None:
        return []
    loops = []
    for line in cut.discrete:
        line = np.asarray(line)
        if len(line) < 9 or np.linalg.norm(line[0] - line[-1]) > max(mesh.extents) * 1e-4:
            continue
        xy = (line[:-1] - origin) @ basis
        # Resample uniformly by arc length: tessellation density must not move
        # the fitted center toward a high-poly side of a curved object.
        closed = np.vstack([xy, xy[0]])
        lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        cumulative = np.r_[0, np.cumsum(lengths)]
        keep = np.r_[True, np.diff(cumulative) > 1e-12]
        if cumulative[-1] < 1e-8:
            continue
        values = np.linspace(0, cumulative[-1], 128, endpoint=False)
        xy = np.column_stack([np.interp(values, cumulative[keep], closed[keep, j]) for j in range(2)])
        system = np.column_stack([2 * xy, np.ones(len(xy))])
        solution = np.linalg.lstsq(system, np.sum(xy * xy, axis=1), rcond=None)[0]
        center2 = solution[:2]
        radii = np.linalg.norm(xy - center2, axis=1)
        if np.min(radii) < 1e-8:
            continue
        loops.append({"center": np.asarray(origin) + basis @ center2, "min_radius": float(np.min(radii)), "max_radius": float(np.max(radii)), "radius": float(np.mean(radii)), "roundness": float(np.std(radii) / np.mean(radii)), "line": line})
    return loops



def raw_opening(mesh, origin, axis, size):
    segments = trimesh.intersections.mesh_plane(mesh, np.asarray(axis), np.asarray(origin))
    if not len(segments):
        return None
    return radial_opening([{"line":segment} for segment in segments],origin,axis,size)


def radial_opening(loops, origin, axis, size):
    """Measured radial entry/exit pairs also support a C-shaped rigid opening."""
    _, basis = plane_basis(axis)
    lines = [np.asarray(loop["line"]) for loop in loops]
    if not lines:
        return None
    points = np.concatenate(lines)
    xy = (points - origin) @ basis
    solution = np.linalg.lstsq(np.column_stack([2*xy, np.ones(len(xy))]),
                              np.sum(xy*xy, axis=1), rcond=None)[0]
    center2 = solution[:2]
    starts = np.concatenate([(line[:-1]-origin) @ basis for line in lines])
    edges = np.concatenate([np.diff(line,axis=0) @ basis for line in lines])
    if len(starts) == 0:
        return None
    def cross(a,b):
        return a[...,0]*b[...,1]-a[...,1]*b[...,0]
    relative = starts-center2
    inner, outer = [], []
    for angle in np.linspace(0,2*np.pi,96,endpoint=False):
        direction = np.array([np.cos(angle), np.sin(angle)])
        denominator = cross(direction,edges)
        valid = np.abs(denominator)>1e-12
        distances = np.full(len(edges),np.inf)
        fractions = np.full(len(edges),np.inf)
        distances[valid]=cross(relative[valid],edges[valid])/denominator[valid]
        fractions[valid]=cross(relative[valid],direction)/denominator[valid]
        hits = np.sort(distances[valid & (distances>size*1e-6) &
                                    (fractions>=-1e-8) & (fractions<=1+1e-8)])
        if len(hits)<2:
            continue
        hits = hits[np.r_[True,np.diff(hits)>size*1e-5]]
        if len(hits)>=2 and hits[1]-hits[0]>size*.002:
            inner.append(hits[0]);outer.append(hits[1])
    coverage = len(inner)/96
    if coverage<.70:
        return None
    inner, outer = np.asarray(inner), np.asarray(outer)
    if inner.min()<=size*.04 or inner.max()>=size*.55 or inner.std()/inner.mean()>.22:
        return None
    if np.median(outer/inner)<1.02:
        return None
    return {"axis":axis, "center":np.asarray(origin)+basis@center2,
            "inner_radius":float(inner.min()),"outer_radius":float(outer.max()),
            "angular_coverage":coverage, "roundness":float(inner.std()/inner.mean())}


def hole_frame(mesh,region_center=None,max_distance=None):
    vertices = np.asarray(mesh.vertices)
    covariance = np.cov(vertices.T)
    _, axes = np.linalg.eigh(covariance)
    directions = [*axes.T, *np.eye(3)]
    best = None
    candidates=[]
    planes = []
    center = mesh.bounds.mean(axis=0)
    size = max(mesh.extents)
    for axis in directions:
        for offset in (0, -0.04, 0.04):
            origin = center + axis * size * offset
            loops = sections(mesh, origin, axis)
            planes.append((origin, axis, loops))
            for inner in loops:
                for outer in loops:
                    if inner is outer or inner["radius"] >= outer["radius"] * 0.94:
                        continue
                    separation = np.linalg.norm(inner["center"] - outer["center"])
                    if separation > inner["radius"] * 0.12 or inner["roundness"] > 0.20 or outer["roundness"] > 0.25:
                        continue
                    # A radial opening in an annular section is necessary,
                    # although it cannot prove all reconstructed topology.
                    if not size * 0.04 < inner["min_radius"] < size * 0.55:
                        continue
                    score = inner["roundness"] + outer["roundness"] + separation / inner["radius"] + abs(offset)
                    if best is None or score < best[0]:
                        best = (score, {"axis": axis, "center": inner["center"], "inner_radius": inner["min_radius"], "outer_radius": outer["max_radius"]})
                    candidates.append((score,{"axis":axis,"center":inner["center"],"inner_radius":inner["min_radius"],"outer_radius":outer["max_radius"]}))
    for origin, axis, loops in planes:
        candidate = radial_opening(loops, origin, axis, size)
        if candidate is None:
            candidate = raw_opening(mesh,origin,axis,size)
        if candidate is not None:
            score = candidate["roundness"] + (1-candidate["angular_coverage"])*.1
            if best is None or score<best[0]:
                best = (score,candidate)
            candidates.append((score,candidate))
    if best is None:
        raise ValueError("No through opening was verified on mesh B. It may be filled or distorted; use a better GLB or edit it.")
    errors=[]
    if region_center is not None:
        region_center=np.asarray(region_center)
        if max_distance is not None:candidates=[item for item in candidates if np.linalg.norm(item[1]["center"]-region_center)<=max_distance]
    for score,result in sorted(candidates,key=lambda item:np.linalg.norm(item[1]["center"]-region_center) if region_center is not None else item[0]):
        try:return verify_aperture(mesh,result)
        except ValueError as error:errors.append(str(error))
    raise ValueError(errors[0] if errors else "No aperture candidate passed verification.")


def verify_aperture(mesh,result):
    result=dict(result)
    vertices=np.asarray(mesh.vertices)
    size=max(mesh.extents)
    # Require the aperture to remain present on adjacent sections; a single
    # decorative groove or a capped cavity must not count as a through hole.
    axis = result["axis"]
    span = np.ptp(vertices @ axis)
    # A closed bottom/cap along the purported hole axis is not a through opening.
    if geometry.ray_hit(mesh, result["center"]-axis*(size+span), axis) is not None:
        error=ValueError("The opening axis in B is blocked by a surface; it is not verified as through.")
        error.measurements={"through_axis_blocked":True,"alternative_feature":"cavity","instruction":"Use cavity only when the requested interaction is a cover or insertion into a closed-end space; do not waive a through requirement."}
        raise error
    for offset in (-0.22, 0.22):
        section_origin = result["center"] + axis * span * offset
        loops = sections(mesh, section_origin, axis)
        if result.get("angular_coverage") is not None:
            adjacent = radial_opening(loops, section_origin, axis, size)
            if adjacent is None:
                adjacent = raw_opening(mesh,section_origin,axis,size)
            if adjacent is None or abs(adjacent["inner_radius"]/result["inner_radius"]-1)>.35:
                raise ValueError("The opening in B is not continuous through its thickness.")
            result["inner_radius"] = min(result["inner_radius"], adjacent["inner_radius"])
            continue
        matched = [loop for loop in loops if np.linalg.norm(loop["center"] - (result["center"] + axis * span * offset)) < result["inner_radius"] * 0.2 and abs(loop["min_radius"] / result["inner_radius"] - 1) < 0.35]
        if not matched:
            raise ValueError("The opening through B has not been verified; surrounding placement will not be applied automatically.")
        result["inner_radius"] = min(result["inner_radius"], min(loop["min_radius"] for loop in matched))
    result["axial_span"] = float(span)
    result["measurement"] = "mesh-plane segments, radial pairs and neighbouring planes"
    result["watertight"] = bool(mesh.is_watertight)
    return result


def target_section(mesh, point, axis):
    loops = sections(mesh, point, axis)
    if not loops:
        raise ValueError("No closed section was found at the target region. Choose an isolated cross section away from junctions.")
    _,basis=plane_basis(axis)
    def contains_origin(loop):
        xy=(loop["line"]-point)@basis
        a,b=xy[:-1],xy[1:]
        crossing=(a[:,1]>0)!=(b[:,1]>0)
        if not crossing.any():return False
        a,b=a[crossing],b[crossing]
        x=a[:,0]+(-a[:,1])*(b[:,0]-a[:,0])/(b[:,1]-a[:,1])
        return bool(np.count_nonzero(x>0)%2)
    enclosing=[loop for loop in loops if contains_origin(loop)]
    loop=min(enclosing,key=lambda item:item["radius"]) if enclosing else min(
        loops,key=lambda item:np.min(np.linalg.norm(item["line"]-point,axis=1)))
    if loop["max_radius"] > max(mesh.extents) * 0.6 or loop["roundness"] > 0.35:
        raise ValueError("The selected section is unsuitable for the near-circular opening fitter. Revise its location or direction.")
    return loop


def fit(base, product, base_transform, point, axis_end, initial_rotation, clearance=1.05):
    world = base.copy()
    matrix = geometry.transform_matrix(base_transform)
    world.apply_transform(matrix)
    point, end = trimesh.transform_points([point, axis_end], matrix)
    axis, _ = plane_basis(end - point)
    section = target_section(world, point, axis)
    hole = hole_frame(product)
    radius = section["max_radius"]
    scale = radius / hole["inner_radius"] * clearance
    # Check the full band's thickness, not just its middle slice.
    for offset in (-0.4, 0.4):
        cross = target_section(world, section["center"] + axis * hole["axial_span"] * scale * offset, axis)
        radius = max(radius, cross["max_radius"] + np.linalg.norm(cross["center"] - (section["center"] + axis * hole["axial_span"] * scale * offset)))
    scale = radius / hole["inner_radius"] * clearance
    current = Rotation.from_euler("XYZ", initial_rotation)
    correction, _ = Rotation.align_vectors([axis], [current.apply(hole["axis"])])
    rotation = correction * current
    center = section["center"]
    position = center - rotation.apply(hole["center"] * scale)
    transform = Transform(position=position.tolist(), rotation=rotation.as_euler("XYZ").tolist(), scale=[scale] * 3)
    diagnostics = {"method": "annular sections and local target sections", "hole_verified": True, "target_center_world": center.tolist(), "target_axis_world": axis.tolist(), "hole_center_local": hole["center"].tolist(), "hole_axis_local": hole["axis"].tolist(), "target_radius_relative": radius, "hole_radius_local": hole["inner_radius"], "visual_clearance_factor": clearance, "size_measurement": False}
    return transform, diagnostics


def check_pose(transform, diagnostics):
    matrix = geometry.transform_matrix(transform)
    center = (matrix @ [*diagnostics["hole_center_local"], 1])[:3]
    axis = matrix[:3, :3] @ diagnostics["hole_axis_local"]
    axis /= np.linalg.norm(axis)
    target = np.asarray(diagnostics["target_center_world"])
    radius = diagnostics["target_radius_relative"]
    error = np.linalg.norm(center - target) / max(radius, 1e-8)
    alignment = abs(float(np.dot(axis, diagnostics["target_axis_world"])))
    clearance = diagnostics["hole_radius_local"] * min(transform.scale) / radius
    failures = []
    if error > 0.18:
        failures.append("The opening center in B does not match the target section center.")
    if alignment < 0.96:
        failures.append("The opening axis in B is not aligned with the target direction.")
    if clearance < 1.015:
        failures.append("The opening in B is smaller than the target section; material penetration is possible.")
    return {"aperture_center_error_in_target_radii": float(error), "aperture_axis_alignment": alignment, "aperture_clearance_ratio": float(clearance)}, failures

def local_section(mesh, point, end, support_extent=None, allow_search=True):
    """First isolated section along the supplied local landmark interval."""
    point,end=np.asarray(point),np.asarray(end)
    axis,length=(end-point),float(np.linalg.norm(end-point))
    axis=axis/max(length,1e-12)
    search_axis=axis.copy()
    # The first point may be at a junction. Estimate direction on the more
    # isolated second landmark, then search back along the original interval.
    if allow_search:
        try:
            endpoint=target_section(mesh,end,axis)
            if support_extent is None or endpoint["max_radius"]*2<=support_extent*1.5:
                _,axis=refine_section_axis(mesh,endpoint,axis,support_extent)
        except ValueError:
            pass
    offsets=np.linspace(0,length*.9,13) if allow_search else [0.]
    failures=[]
    for offset in offsets:
        origin=point+search_axis*offset
        try:
            section=target_section(mesh,origin,axis)
            diameter=2*section["max_radius"]
            boundary=float(np.min(np.linalg.norm(section["line"]-origin,axis=1)))
            if support_extent is not None and diameter>support_extent*1.5:
                failures.append({"offset":float(offset),"diameter":diameter,"reason":"outside support"})
                continue
            if support_extent is not None and boundary>support_extent*.35:
                failures.append({"offset":float(offset),"diameter":diameter,"reason":"different local part"})
                continue
            if allow_search:
                section,refined_axis=refine_section_axis(mesh,section,axis,support_extent)
            else:
                refined_axis=axis
            return {**section,"axis":refined_axis,"requested_start":point,"requested_end":end,
                    "search_offset":float(offset),"support_extent":support_extent,
                    "section_trials":failures}
        except ValueError as error:
            failures.append({"offset":float(offset),"reason":str(error)})
    error=ValueError("No isolated section was found in the selected region/direction; revise the landmarks or axis.")
    error.measurements={"requested_start":point.tolist(),"requested_end":end.tolist(),
        "support_extent":support_extent,"section_trials":failures}
    raise error


def fit_band_section(mesh, section_feature, opening_feature, target_ratio, opening_on_product):
    """Fit the whole band's thickness without growing onto a neighbouring part."""
    start=np.asarray(section_feature.get("requested_start",section_feature["center"]))
    end=np.asarray(section_feature.get("requested_end",start+section_feature["axis"]*.001))
    axis=np.asarray(section_feature["axis"])
    search_axis=(end-start)/max(np.linalg.norm(end-start),1e-12)
    extent=section_feature.get("section_support")
    distance=float(np.linalg.norm(end-start))
    search=bool(section_feature.get("allow_axis_search",False))
    base_offset=float(section_feature.get("search_offset",0))
    offsets=np.linspace(base_offset,max(base_offset,distance*.98),17) if search else [base_offset]
    trials=[]
    for offset in offsets:
        origin=start+search_axis*offset
        try:
            section=target_section(mesh,origin,axis)
            radius=section["max_radius"]
            if extent is not None and radius*2>extent*1.5:
                continue
            if extent is not None and np.min(np.linalg.norm(section["line"]-origin,axis=1))>extent*.35:
                continue
            scale=radius*target_ratio/opening_feature["radius"] if opening_on_product else opening_feature["radius"]/(radius*target_ratio)
            valid=True
            for _ in range(2):
                full_radius=radius
                thickness=opening_feature["axial_span"]*scale if opening_on_product else opening_feature["axial_span"]/scale
                for fraction in (-.4,.4):
                    target=section["center"]+axis*thickness*fraction
                    side=target_section(mesh,target,axis)
                    drift=float(np.linalg.norm(side["center"]-target))
                    # A sudden merge into a wider body or neighbouring branch is not a
                    # reason to inflate the product around the entire base.
                    if drift>radius*.35 or side["max_radius"]>radius*1.3:
                        valid=False;break
                    full_radius=max(full_radius,side["max_radius"]+drift)
                if not valid:break
                radius=full_radius
                scale=radius*target_ratio/opening_feature["radius"] if opening_on_product else opening_feature["radius"]/(radius*target_ratio)
            trials.append({"offset":float(offset),"radius":float(radius),"compatible_band":valid})
            if valid:
                section_feature.update(center=section["center"],radius=float(radius),
                    extent=2*float(radius),search_offset=float(offset),band_trials=trials)
                return float(scale)
        except ValueError:
            trials.append({"offset":float(offset),"compatible_band":False})
    error=ValueError("The opening does not fit across its thickness. Do not enlarge it around neighboring regions.")
    error.measurements={"band_trials":trials}
    raise error


def refine_section_axis(mesh, frame, axis, support_extent=None):
    """Recover a local centreline tangent from neighbouring measured sections."""
    axis=np.asarray(axis,dtype=float);center=np.asarray(frame["center"])
    original=axis.copy();radius=float(frame["max_radius"])
    for _ in range(2):
        step=max(radius*.35,1e-5)
        try:
            low=target_section(mesh,center-axis*step,axis)
            high=target_section(mesh,center+axis*step,axis)
            if max(low["max_radius"],high["max_radius"])>radius*1.4:
                break
            tangent=high["center"]-low["center"]
            tangent=tangent/max(np.linalg.norm(tangent),1e-12)
            if np.dot(tangent,original)<np.cos(np.deg2rad(35)):
                break
            measured=target_section(mesh,center,tangent)
            if support_extent is not None and 2*measured["max_radius"]>support_extent*1.5:
                break
            axis,frame,center=tangent,measured,measured["center"]
            radius=float(frame["max_radius"])
        except ValueError:
            break
    return frame,axis

