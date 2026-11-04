"""Declarative spatial programs: one solver, arbitrary object semantics.

AI chooses grounded features and combines bounded geometric primitives. The
runtime never dispatches on object names, user text, or the relation label.
No generated Python is executed. All candidates share the same solver with bounded relative-scale refinement.
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.optimize import least_squares,minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from . import aperture, geometry
from .schemas import RegionHints, SpatialConstraint, SpatialFeature, SpatialProgram, Transform

VERSION = "spatial-program-8-branched-grounding"


def compatibility_program(plan):
    """Translate saved v1.x plans/QA fixtures; new provider responses require a program."""
    a = SpatialFeature(id="a", slot="A", kind="point", point=plan.base_anchor, view=plan.base_view, uv=plan.base_uv)
    b = SpatialFeature(id="b", slot="B", kind="point", point=plan.product_anchor, view=plan.product_view, uv=plan.product_uv)
    constraints = [SpatialConstraint(kind="coincident", a="a", b="b")]
    if plan.relation in {"on", "attached"}:
        a.kind = b.kind = "surface"
        constraints = [SpatialConstraint(kind="contact", a="a", b="b", target=max(0, plan.clearance))]
    if plan.fit_mode == "aperture":
        a.kind, a.end, a.end_view, a.end_uv = "section", plan.base_axis_end, plan.base_axis_end_view, plan.base_axis_end_uv
        b.kind = "aperture"
        constraints += [SpatialConstraint(kind="align", a="a", b="b"), SpatialConstraint(kind="clearance", a="a", b="b", target=1.05)]
    if plan.relation == "inside":
        a.kind = "region"
        constraints += [SpatialConstraint(kind="containment", a="a", b="b")]
    return SpatialProgram(features=[a, b], constraints=constraints, deformation_required=plan.deform)


class ProgramSolver:
    def __init__(self, base_path, product_path, base_transform, plan, captures, hints=None, require_scale=False, search_context=None):
        from .composition import solid_mesh, surface_samples
        self.search_context=search_context
        if search_context:search_context.check()
        from .scale_policy import complete_fit_constraints
        self.program, self.inferred_constraints = complete_fit_constraints(plan.program or compatibility_program(plan))
        self.require_scale = require_scale and plan.program is not None
        self.deformation_required = self.program.deformation_required or plan.deform
        cache_key=(str(base_path),str(product_path),tuple(base_transform.position),tuple(base_transform.rotation),tuple(base_transform.scale))
        cached=search_context.geometry_cache.get(cache_key) if search_context else None
        if cached:
            self.__dict__.update(cached)
        else:
            self.base_local, self.product = geometry.flat_mesh(base_path), geometry.flat_mesh(product_path)
            self.base_matrix = geometry.transform_matrix(base_transform)
            self.base = self.base_local.copy(); self.base.apply_transform(self.base_matrix)
            self.base_solid, self.closed = solid_mesh(self.base)
            self.product_solid, self.product_closed = solid_mesh(self.product)
            self.volume_available = self.closed
            self.product_volume_available = self.product_closed
            self.size = max(float(max(self.base.extents)), 1e-6)
            self.points, _ = surface_samples(self.product, 128, seed=421)
            base_points, normals = surface_samples(self.base, 3072, seed=42)
            self.tree, self.base_points, self.normals = cKDTree(base_points), base_points, normals
            self.proxy_available = self.closed and self.base_solid.is_winding_consistent and self.base_solid.volume > 0
            if search_context:search_context.geometry_cache[cache_key]={name:getattr(self,name) for name in ['base_local', 'product', 'base_matrix', 'base', 'base_solid', 'closed', 'product_solid', 'product_closed', 'volume_available', 'product_volume_available', 'size', 'points', 'tree', 'base_points', 'normals', 'proxy_available']}
        self.fit_interfaces = {}
        self.hints = hints or RegionHints()
        self.captures = captures
        from .geometry_feedback import FeatureResolutionError
        self.features={};issues=[]
        for feature in self.program.features:
            if search_context:search_context.check()
            try:self.features[feature.id]=self.resolve(feature)
            except ValueError as error:
                issues.append({"feature_id":feature.id,"slot":feature.slot,"kind":feature.kind,
                    "request":feature.model_dump(),"error":str(error),
                    "measurements":getattr(error,"measurements",{})})
        if issues:raise FeatureResolutionError(issues,self.features)
        self.validate_types()
        from .orientation import conflicts
        self.orientation_issues,self.suspended_alignments=conflicts(self.features,self.program.constraints)
        from .scale_policy import scale_evidence
        self.scale_evidence = scale_evidence(self.program,self.features)
        self.scale_prior = plan.relative_scale * float(max(self.base_local.extents * base_transform.scale)) / max(float(max(self.product.extents)), 1e-8)
        self.base_rotation = Rotation.from_euler("XYZ", base_transform.rotation)
        self.rotation_prior = self.base_rotation * Rotation.from_euler("XYZ", plan.rotation_degrees, degrees=True)

    def resolve(self, feature):
        mesh = self.base_local if feature.slot == "A" else self.product
        def point(fraction, view, uv, ref=None):
            if ref is not None:
                from .spatial_context import reference
                return reference(mesh,ref)
            capture = self.captures[view] if 0 <= view < len(self.captures) and self.captures[view].slot == feature.slot else None
            center, hit = geometry.choose_anchor(mesh, fraction, capture, uv)
            if uv is not None and capture is not None and not hit:
                from .grounding import snap_camera_anchor
                corrected = snap_camera_anchor(mesh, capture, uv)
                if corrected is not None:
                    center, hit = corrected[0], True
            if uv is not None and (capture is None or not hit):
                error=ValueError(f"Feature {feature.id}: the selected image point does not hit mesh {feature.slot}. Reselect the target region.")
                from .grounding import nearest_camera_landmark
                error.measurements={"view":view,"requested_uv":uv,
                    "nearest_surface":nearest_camera_landmark(mesh,capture,uv) if capture else None}
                raise error
            return center
        if feature.source == "user":
            selected = getattr(self.hints, feature.slot + "_point")
            if selected is None:
                raise ValueError(f"Feature {feature.id} requires a user-selected point on {feature.slot}.")
            center = np.asarray(selected)
            selected_end = getattr(self.hints, feature.slot + "_end")
            end = np.asarray(selected_end) if selected_end is not None else None
        elif feature.kind in {"aperture","cavity"}:
            # A through-hole centre is empty space, not a surface landmark.
            # hole_frame below verifies the actual opening independently.
            center = geometry.anchor_point(mesh, feature.point)
            end = geometry.anchor_point(mesh,feature.end) if feature.end is not None else None
        else:
            center = point(feature.point, feature.view, feature.uv,feature.point_ref)
            end = point(feature.end or feature.point, feature.end_view, feature.end_uv,feature.end_ref) if feature.end is not None or feature.end_ref is not None else None
        local_point = np.asarray(center).copy()
        # Resolve A's sections after its world transform, including nonuniform scale.
        working = self.base if feature.slot == "A" else mesh
        if feature.slot == "A":
            center = trimesh.transform_points([center], self.base_matrix)[0]
            if end is not None: end = trimesh.transform_points([end], self.base_matrix)[0]
        local_extents = mesh.extents * np.asarray(feature.extent)
        measured_extent = float(max(np.linalg.norm(self.base_matrix[:3,:3] @ np.diag(local_extents),axis=0))) if feature.slot=="A" else float(max(local_extents))
        result = {"slot": feature.slot, "kind": feature.kind, "center": np.asarray(center), "axis": None, "radius": None,
                  "extent":max(measured_extent,1e-8), "extent_source":"selected local support",
                  "support_extent":max(measured_extent,1e-8)}
        if feature.kind == "surface":
            contact=any(c.kind=="contact" and feature.id in (c.a,c.b) for c in self.program.constraints)
            if contact:result["center"],result["axis"]=geometry.surface_frame(working,center)
            else:
                radius=min(self.size*.075,max(self.size*.01,result["support_extent"]*.12))
                result["center"],result["axis"]=geometry.surface_patch_frame(working,center,radius)
                result["normal_method"]="area-weighted same-facing local patch"
        elif feature.kind in {"axis", "section"}:
            if end is None or np.linalg.norm(end - center) < self.size * 0.01:
                raise ValueError(f"Feature {feature.id} lacks two sufficiently separated direction points.")
            axis = (end - center) / np.linalg.norm(end - center)
            result["axis"] = axis
            result.update(extent=float(np.linalg.norm(end-center)),extent_source="landmark distance")
            if feature.kind == "axis":
                surface,normal=geometry.surface_patch_frame(working,center,min(self.size*.075,max(self.size*.01,result["support_extent"]*.12)))
                if np.linalg.norm(surface-center)<self.size*.02:
                    result["alternative_surface_frame"]={"center":surface,"normal":normal,
                        "status":"measured tool alternative, not a semantic substitution",
                        "use":"If intent means outward facing, consider kind=surface here. Do not replace a genuine tangent/along direction."}
            if feature.kind == "section":
                support=result["support_extent"] if any(value<.999 for value in feature.extent) else None
                cavity_pair=any(c.kind=="clearance" and feature.id in (c.a,c.b) and
                    next(f for f in self.program.features if f.id==(c.b if c.a==feature.id else c.a)).kind=="cavity"
                    for c in self.program.constraints)
                if cavity_pair:
                    section=aperture.target_section(working,center,axis)
                    section.update(axis=axis,search_offset=0.,section_trials=[])
                    # Extent is a proposal, not permission to move the interface
                    # towards a narrower endpoint of the same object.
                    support=None
                else:
                    section=aperture.local_section(working,center,end,support,allow_search=feature.source!="user")
                result.update(center=section["center"],radius=section["max_radius"],
                    extent=2*section["max_radius"],axis=section["axis"],extent_source="measured local section diameter",
                    requested_start=np.asarray(center),requested_end=np.asarray(end),
                    search_offset=section["search_offset"],section_support=support,
                    allow_axis_search=feature.source!="user" and not cavity_pair,section_trials=section["section_trials"])
                if cavity_pair:
                    sides=[]
                    for sign in (1,-1):
                        direction=axis*sign
                        roof=geometry.ray_hit(working,section["center"]+direction*self.size*1e-5,direction)
                        depth=float(np.dot(roof[0]-section["center"],direction)) if roof is not None else 0
                        if depth>self.size*1e-4:sides.append({"sign":sign,"depth":depth})
                    if not sides:raise ValueError("Target insertion depth could not be measured on either side.")
                    result["insertion_sides"]=sides
                    result["insertion_depth"]=next((v["depth"] for v in sides if v["sign"]==1),sides[0]["depth"])
        elif feature.kind == "aperture":
            key=(feature.slot,feature.kind,tuple(feature.point),tuple(feature.end or []),tuple(feature.extent))
            frame=self.search_context.measurements.get(key) if self.search_context else None
            if frame is None:
                frame = aperture.hole_frame(working,region_center=center,max_distance=result["support_extent"]*.75)
                if self.search_context:self.search_context.measurements[key]=frame
            result.update(center=frame["center"], axis=frame["axis"], radius=frame["inner_radius"], axial_span=frame["axial_span"], extent=2*frame["inner_radius"],extent_source="measured opening diameter",aperture_measurement=frame)
        elif feature.kind == "cavity":
            from .cavity import cavity_frame
            key=(feature.slot,feature.kind,tuple(feature.point),tuple(feature.end or []),tuple(feature.extent))
            frame=self.search_context.measurements.get(key) if self.search_context else None
            if frame is None:
                frame=cavity_frame(working,end-center if end is not None else None,region_center=center,max_distance=result["support_extent"]*.75)
                if self.search_context:self.search_context.measurements[key]=frame
            result.update(center=frame["center"],axis=frame["axis"],radius=frame["inner_radius"],
                axial_span=frame["depth"],depth=frame["depth"],roof=frame["roof"],
                extent=2*frame["inner_radius"],extent_source="measured cavity mouth",
                aperture_measurement=frame)
        elif feature.kind == "region":
            # Region bounds are expressed in the object's canonical frame.
            local_center = local_point
            extent = np.maximum(mesh.extents * feature.extent, 1e-6)
            result["region_inverse"] = np.linalg.inv(self.base_matrix) if feature.slot == "A" else np.eye(4)
            result["region_center"] = local_center
            result["region_half"] = extent / 2
            result["extent"] = max(measured_extent,1e-8)
        return result

    def validate_types(self):
        for constraint in self.program.constraints:
            a, b = self.features[constraint.a], self.features[constraint.b]
            if constraint.kind in {"align", "contact"} and (a["axis"] is None or b["axis"] is None):
                raise ValueError("Directional constraints require an axis or surface normal.")
            if constraint.kind == "clearance" and (a["radius"] is None or b["radius"] is None):
                raise ValueError("Clearance requires a verified section and opening.")

    def branches(self):
        """The normal of a section is unsigned; an inward cavity axis is signed."""
        import copy,itertools
        options=[]
        for c in self.program.constraints:
            if c.kind!="clearance":continue
            a,b=self.features[c.a],self.features[c.b]
            section_id=c.a if a["kind"]=="section" else c.b
            opening=b if a["kind"]=="section" else a
            if opening["kind"]!="cavity":continue
            directions=[k for k in self.program.constraints if k.kind=="align" and (k.a,k.b)==(c.a,c.b)]
            if any(k.direction!="either" for k in directions):continue
            options.append((section_id,self.features[section_id].get("insertion_sides",[{"sign":1,"depth":self.features[section_id]["insertion_depth"]}])))
        if not options:return [self]
        branches=[]
        for combination in itertools.islice(itertools.product(*(x[1] for x in options)),4):
            branch=copy.copy(self);branch.features=copy.deepcopy(self.features);branch.fit_interfaces={};branch.direction_branch={}
            for (name,_),side in zip(options,combination):
                branch.features[name]["axis"]=np.asarray(self.features[name]["axis"])*side["sign"]
                branch.features[name]["insertion_depth"]=side["depth"]
                branch.direction_branch[name]=side["sign"]
            from .orientation import conflicts
            branch.orientation_issues,branch.suspended_alignments=conflicts(branch.features,branch.program.constraints)
            branches.append(branch)
        def plausibility(branch):
            ratios=[]
            for c in self.program.constraints:
                if c.kind!="clearance":continue
                a,b=branch.features[c.a],branch.features[c.b]
                opening=b if a["kind"]=="section" else a;section=a if a["kind"]=="section" else b
                if opening["kind"]!="cavity":continue
                ratio=section["insertion_depth"]*opening["radius"]*c.target/(opening["depth"]*section["radius"])
                ratios.append((ratio>1.03,abs(np.log(max(ratio,1e-8)))))
            return max(ratios) if ratios else (False,0.)
        return sorted(branches,key=plausibility)

    def insertion_feasibility(self):
        """Conservative depth bound before expensive full profile fitting."""
        failures=[]
        for c in self.program.constraints:
            if c.kind!="clearance":continue
            a,b=self.features[c.a],self.features[c.b]
            opening=b if a["kind"]=="section" else a;section=a if a["kind"]=="section" else b
            if opening["kind"]!="cavity":continue
            tolerances=[k.tolerance for k in self.program.constraints if k.kind=="coincident" and (k.a,k.b)==(c.a,c.b)]
            if not tolerances:continue
            margin=self.size*max(.01,2.6*max(tolerances))
            if a["kind"]=="section":
                fitted=opening["depth"]*section["radius"]/opening["radius"]
                minimum,maximum=fitted*.70/1.35,fitted*1.30*1.35
                impossible=section["insertion_depth"]>maximum+margin or minimum>section["insertion_depth"]*1.5+margin
            else:
                fitted=section["insertion_depth"]*opening["radius"]/(section["radius"]*c.target)
                minimum,maximum=fitted*.70/1.35,fitted*1.30*1.35
                impossible=minimum>opening["depth"]+margin or opening["depth"]>maximum*1.5+margin
            if impossible:failures.append({"code":"insertion_depth_bound","features":[c.a,c.b],"depth_interval":[minimum,maximum],"position_margin":margin,"section_depth":section["insertion_depth"],"cavity_depth":opening["depth"]})
        return failures

    def seed(self):
        from .scale_policy import ScaleEvidenceError, landmark_scales
        if self.require_scale and not self.scale_evidence["verified"]:
            raise ScaleEvidenceError("The plan locates objects but lacks scale constraints. Use measured clearance, local proportion, or two distinct landmark pairs.")
        scale, rotation = self.scale_prior, self.rotation_prior
        proportions=[c for c in self.program.constraints if c.kind=="proportion"]
        ratios=[self.features[c.a]["extent"]*max(c.target,.02)/self.features[c.b]["extent"] for c in proportions]
        if not ratios:ratios=landmark_scales(self.program,self.features)
        if ratios:scale=float(np.median(ratios))
        directional = [c for c in self.program.constraints if c.kind in {"align", "contact"} and (c.a,c.b) not in self.suspended_alignments]
        primary={(c.a,c.b) for c in self.program.constraints if c.kind=="clearance"}
        directional.sort(key=lambda c:(c.a,c.b) not in primary)
        if directional:
            sources, targets, weights = [], [], []
            for constraint in directional:
                a, b = self.features[constraint.a], self.features[constraint.b]
                sign = -1 if constraint.kind == "contact" or constraint.direction == "opposed" else 1
                if constraint.direction == "either" and constraint.kind != "contact" and "cavity" not in {a["kind"],b["kind"]} and np.dot(rotation.apply(b["axis"]), a["axis"]) < 0:
                    sign = -1
                av,bv=np.asarray(a["axis"]),np.asarray(b["axis"])
                if constraint.frame_a is not None:
                    na,nb=self.features[constraint.frame_a]["axis"],self.features[constraint.frame_b]["axis"]
                    av=av-na*np.dot(av,na);bv=bv-nb*np.dot(bv,nb)
                    if min(np.linalg.norm(av),np.linalg.norm(bv))<.1:raise ValueError("Projected heading has no measurable in-plane direction.")
                    av=av/np.linalg.norm(av);bv=bv/np.linalg.norm(bv)
                sources.append(rotation.apply(bv)); targets.append(av * sign); weights.append(np.inf if not weights and (constraint.a,constraint.b) in primary else constraint.weight)
            if len(sources) > 1 and np.linalg.matrix_rank(np.asarray(sources), tol=1e-6) < 2:
                sources, targets, weights = sources[:1], targets[:1], weights[:1]
            correction, _ = Rotation.align_vectors(targets, sources, weights=weights)
            rotation = correction * rotation
        clearances = [c for c in self.program.constraints if c.kind == "clearance"]
        if clearances:
            scales=[]
            for constraint in clearances:
                a,b=self.features[constraint.a],self.features[constraint.b]
                opening=b if a["kind"]=="section" else a
                section=a if a["kind"]=="section" else b
                if opening["kind"]=="cavity":
                    scales.append(section["radius"]*constraint.target/opening["radius"]
                        if a["kind"]=="section" else opening["radius"]/(section["radius"]*constraint.target))
                elif a["kind"]=="section":
                    scales.append(aperture.fit_band_section(self.base,a,b,constraint.target,True))
                else:
                    scales.append(aperture.fit_band_section(self.product,b,a,constraint.target,False))
            scale=float(np.median(scales))
        self.scale_prior = max(float(scale),1e-8)
        positions, weights = [], []
        for constraint in self.program.constraints:
            if constraint.kind in {"coincident", "contact", "offset", "containment"}:
                a, b = self.features[constraint.a], self.features[constraint.b]
                target = a["center"] + self.base_rotation.apply(constraint.offset) * self.size
                if constraint.kind == "contact": target = target + a["axis"] * constraint.target * self.size
                positions.append(target - rotation.apply(b["center"] * scale)); weights.append(constraint.weight)
        position = np.average(positions, axis=0, weights=weights) if positions else self.base.bounds.mean(axis=0)
        transform=Transform(position=position.tolist(), rotation=rotation.as_euler("XYZ").tolist(), scale=[scale] * 3)
        from .interface_fit import MeasuredInterface
        if self.orientation_issues:return transform  # impossible graph: diagnostic poses only, never eligible
        for constraint in clearances:
            a,b=self.features[constraint.a],self.features[constraint.b]
            forward=a["kind"]=="section"
            self.fit_interfaces[(constraint.a,constraint.b)]=MeasuredInterface(
                self.base if forward else self.product,self.product if forward else self.base,
                a if forward else b,b if forward else a,transform,forward,constraint.target)
        return transform

    def value(self, constraint, transform):
        a, b = self.features[constraint.a], self.features[constraint.b]
        rotation = Rotation.from_euler("XYZ", transform.rotation)
        scales=np.asarray(transform.scale)
        scale=float(np.exp(np.mean(np.log(scales))))
        def transformed_axis(feature):
            if feature["axis"] is None:return None
            vector=np.asarray(feature["axis"])
            vector=vector/scales if feature["kind"] in {"surface","aperture","cavity"} else vector*scales
            return rotation.apply(vector/np.linalg.norm(vector))
        center = rotation.apply(b["center"] * scales) + transform.position
        tolerance = constraint.tolerance
        if constraint.kind in {"coincident", "contact", "offset"}:
            target = a["center"] + self.base_rotation.apply(constraint.offset) * self.size
            if constraint.kind == "contact": target = target + a["axis"] * constraint.target * self.size
            residual = (center - target) / (self.size * tolerance)
            if constraint.kind == "contact": residual = np.r_[residual, (transformed_axis(b) + a["axis"]) / 0.2]
        elif constraint.kind == "align":
            axis = transformed_axis(b);target_axis=np.asarray(a["axis"])
            if constraint.frame_a is not None:
                na=self.features[constraint.frame_a]["axis"];nb=transformed_axis(self.features[constraint.frame_b])
                target_axis=target_axis-na*np.dot(target_axis,na);axis=axis-nb*np.dot(axis,nb)
                if min(np.linalg.norm(target_axis),np.linalg.norm(axis))<.1:return np.ones(3)*100
                target_axis=target_axis/np.linalg.norm(target_axis);axis=axis/np.linalg.norm(axis)
            if constraint.direction == "either": residual = np.cross(axis, target_axis) / 0.2
            else: residual = (axis - target_axis * (-1 if constraint.direction == "opposed" else 1)) / 0.2
        elif constraint.kind == "clearance":
            ratio = b["radius"] * scale / a["radius"] if a["kind"]=="section" else a["radius"]/(b["radius"]*scale)
            # A fitting clearance constrains the actual opening, not merely its
            # radius. Global AABB tolerances can exceed the whole target section.
            difference=center-a["center"]
            axial=a["axis"]*np.dot(difference,a["axis"])
            local_radius=max(min(a["radius"],b["radius"]*scale),1e-8)
            center_error=(difference-axial)/(local_radius*.10)+axial/(local_radius*.40)
            axis_error = ((transformed_axis(b)-a["axis"]) if "cavity" in {a["kind"],b["kind"]} else np.cross(transformed_axis(b), a["axis"])) / 0.05
            measured=self.fit_interfaces.get((constraint.a,constraint.b))
            residual = np.r_[measured.residuals(transform,tolerance) if measured else
                [np.log(max(ratio,1e-8)/constraint.target)/tolerance],center_error,axis_error]
        elif constraint.kind == "proportion":
            ratio = b["extent"] * scale / a["extent"]
            residual = np.array([np.log(max(ratio, 1e-8) / max(constraint.target, 0.02)) / max(tolerance, 0.08)])
        elif constraint.kind == "containment":
            points = trimesh.transform_points(self.points[:48], geometry.transform_matrix(transform))
            local = trimesh.transform_points(points, a["region_inverse"])
            residual = np.maximum(np.abs(local - a["region_center"]) - a["region_half"], 0).ravel() / (self.size * tolerance)
        else:
            raise ValueError("Unsupported constraint.")
        return residual

    def residuals(self, transform):
        if self.search_context:self.search_context.check()
        values = [self.value(c,transform)*np.sqrt(c.weight)/(1 if c.kind=="clearance" and (c.a,c.b) in self.fit_interfaces else np.sqrt(len(self.value(c,transform)))) for c in self.program.constraints if not (c.kind=="align" and (c.a,c.b) in self.suspended_alignments)]
        # A soft prior prevents unconstrained shrinking; explicit scale/clearance
        # constraints can override it. All semantics use the same collision term.
        values.append(np.array([0.04 * np.log(np.exp(np.mean(np.log(transform.scale))) / self.scale_prior)]))
        points = trimesh.transform_points(self.points, geometry.transform_matrix(transform))
        if self.proxy_available:
            _, nearest = self.tree.query(points)
            signed = np.einsum("ij,ij->i", points - self.base_points[nearest], self.normals[nearest])
            values.append(np.maximum(-signed / self.size - 0.003, 0) * 8 / np.sqrt(len(points)))
        return np.concatenate(values)

    def optimize(self, seed, _coarse=True):
        original=seed
        initial_cost=float(np.mean(self.residuals(seed)**2))
        coarse_evaluations=0
        if self.fit_interfaces and _coarse:
            constraint=next(c for c in self.program.constraints if c.kind=="clearance")
            a,b=self.features[constraint.a],self.features[constraint.b]
            original_rotation=Rotation.from_euler("XYZ",seed.rotation)
            anchor=original_rotation.apply(b["center"]*np.asarray(seed.scale))+seed.position
            best_cost=initial_cost
            # A bounded roll sweep avoids a stationary wrong alignment for
            # polygonal/elliptical interfaces. It changes no semantic labels.
            for angle in np.arange(24)*np.pi/12:
                rotated=Rotation.from_rotvec(np.asarray(a["axis"])*angle)*original_rotation
                for factor in (.8,.9,1.):
                    scales=np.asarray(original.scale)*factor
                    candidate=Transform(position=(anchor-rotated.apply(b["center"]*scales)).tolist(),
                        rotation=rotated.as_euler("XYZ").tolist(),scale=scales.tolist())
                    cost=float(np.mean(self.residuals(candidate)**2));coarse_evaluations+=1
                    if cost<best_cost-1e-9:seed,best_cost=candidate,cost
        seed_rotation = Rotation.from_euler("XYZ", seed.rotation)
        def pose(x):
            rotation = Rotation.from_rotvec(x[3:6]) * seed_rotation
            scale = seed.scale[0] * np.exp(x[6])
            return Transform(position=(np.asarray(seed.position) + x[:3] * self.size).tolist(), rotation=rotation.as_euler("XYZ").tolist(), scale=[scale] * 3)
        bounds = ([-0.8] * 3 + [-np.pi] * 3 + [max(-1.5, np.log(0.001 / seed.scale[0]))], [0.8] * 3 + [np.pi] * 3 + [min(1.5, np.log(1000 / seed.scale[0]))])
        before = float(np.mean(self.residuals(seed) ** 2))
        result = least_squares(lambda x: self.residuals(pose(x)), np.zeros(7), bounds=bounds, max_nfev=40, ftol=0.002, xtol=0.002, gtol=0.002)
        transform = pose(result.x)
        after = float(np.mean(self.residuals(transform) ** 2))
        if after > before: transform, after = seed, before
        evaluations=int(result.nfev)
        mode="uniform relative scale"
        if self.fit_interfaces:
            # First solve unrestricted uniform scale. Then allow symmetric,
            # bounded per-axis changes around that measured solution.
            uniform=transform;origin=np.asarray(uniform.position)
            initial_rotation=Rotation.from_euler("XYZ",uniform.rotation)
            def shaped_pose(x):
                rotated=Rotation.from_rotvec(x[3:6])*initial_rotation
                scales=np.asarray(uniform.scale)*np.exp(x[6:9])
                return Transform(position=(origin+x[:3]*self.size).tolist(),
                    rotation=rotated.as_euler("XYZ").tolist(),scale=scales.tolist())
            def shaped_residual(x):
                shape=x[6:9]-np.mean(x[6:9])
                return np.r_[self.residuals(shaped_pose(x)),shape*.35]
            limit=np.log(1.15)
            refined=least_squares(shaped_residual,np.zeros(9),
                bounds=([-.2]*3+[-.35]*3+[-limit]*3,[.2]*3+[.35]*3+[limit]*3),
                max_nfev=60,ftol=.001,xtol=.001,gtol=.001)
            candidate=shaped_pose(refined.x)
            shaped_cost=float(np.mean(self.residuals(candidate)**2))
            if shaped_cost<after:
                transform,after=candidate,shaped_cost
                mode="measured relative scale plus bounded per-axis refinement"
            evaluations+=int(refined.nfev)
            if not all(interface.measure(transform)["eligible"] for interface in self.fit_interfaces.values()):
                # Least-squares may trade a small acceptance violation for a
                # lower soft loss. Finish with the SAME fixed inequalities.
                current=transform
                x0=np.r_[(np.asarray(current.position)-origin)/self.size,
                    (Rotation.from_euler("XYZ",current.rotation)*initial_rotation.inv()).as_rotvec(),
                    np.log(np.asarray(current.scale)/np.asarray(uniform.scale))]
                x0=np.clip(x0,[-.2]*3+[-.35]*3+[-limit]*3,[.2]*3+[.35]*3+[limit]*3)
                def margins(x):
                    posed=shaped_pose(x);values=[]
                    for interface in self.fit_interfaces.values():values.extend(interface.feasibility_margins(posed))
                    for constraint in self.program.constraints:
                        value=self.value(constraint,posed)
                        if constraint.kind=="clearance":values.extend(1.5-np.abs(value[-6:])-1e-4)
                        else:values.append(1.5-float(np.sqrt(np.mean(value**2)))-1e-4)
                    values.append(1.35-max(posed.scale)/min(posed.scale)-1e-4)
                    return np.asarray(values)
                from .spatial_search import SearchDeadline
                try:
                    feasible=minimize(lambda x:float(np.mean(shaped_residual(x)**2)),x0,method="SLSQP",
                        bounds=list(zip([-.2]*3+[-.35]*3+[-limit]*3,[.2]*3+[.35]*3+[limit]*3)),
                        constraints=[{"type":"ineq","fun":margins}],options={"maxiter":20,"ftol":1e-6})
                    candidate=shaped_pose(feasible.x)
                    candidate_valid=all(interface.measure(candidate)["eligible"] for interface in self.fit_interfaces.values())
                    if candidate_valid and np.min(margins(feasible.x))>=-1e-6:
                        transform=candidate;after=float(np.mean(self.residuals(candidate)**2));mode="bounded scale and fixed-inequality feasibility refinement"
                    evaluations+=int(feasible.nfev)
                except SearchDeadline:pass
        if _coarse and coarse_evaluations and seed!=original:
            alternative,info=self.optimize(original,_coarse=False)
            valid=all(item.measure(transform)["eligible"] for item in self.fit_interfaces.values())
            other_valid=all(item.measure(alternative)["eligible"] for item in self.fit_interfaces.values())
            if (not other_valid,info["objective_after"])<(not valid,after):
                transform,after,mode=alternative,info["objective_after"],info["scale_mode"]
            evaluations+=info["evaluations"]
        return transform, {"performed": True, "method": "shared constrained surface fit",
            "scale_mode":mode,"max_axis_ratio":1.35,"objective_before":initial_cost,
            "objective_after":after,"evaluations":evaluations,"coarse_evaluations":coarse_evaluations}

    def check(self, transform):
        from .collision import penetration_fraction
        checks, failures, warnings = [], [], []
        if self.orientation_issues:
            failures.append("Directional landmarks are mutually inconsistent; re-ground the axes before accepting this placement.")
        for constraint in self.program.constraints:
            residual = self.value(constraint, transform)
            error = float(np.max(np.abs(residual)) if constraint.kind == "clearance" else np.sqrt(np.mean(residual ** 2)))
            passed = error <= 1.5
            interface=self.fit_interfaces.get((constraint.a,constraint.b)) if constraint.kind=="clearance" else None
            if interface is not None:
                fit=interface.measure(transform);limits=fit["limits"]
                error=max(max(0,fit["gap_median_ratio"])/limits["median_gap_ratio"],
                    max(0,fit["gap_p90_ratio"])/limits["p90_gap_ratio"],
                    fit["intrusion_fraction"]/limits["intrusion_fraction"],
                    (1-fit["supported_fraction"])/(1-limits["supported_fraction"]),
                    float(np.max(np.abs(residual[-6:]))))
                passed=fit["eligible"] and error<=1.5
            if not passed: failures.append(f"Unsatisfied {constraint.kind}: {constraint.a} ↔ {constraint.b}.")
            checks.append({"kind": constraint.kind, "a": constraint.a, "b": constraint.b, "normalized_error": round(error, 6), "passed": passed})
            if constraint.kind == "containment": warnings.append("Containment uses regions and mesh samples; a complete empty cavity is not proven.")
        axis_ratio=max(transform.scale)/min(transform.scale)
        if axis_ratio>1.35+1e-5:
            failures.append("Per-axis scaling exceeds the shape-preservation limit.")
        interface_checks=[]
        for key,interface in self.fit_interfaces.items():
            measured=interface.measure(transform)
            measured.update(a=key[0],b=key[1])
            interface_checks.append(measured)
            if not measured["eligible"]:
                failures.append("The measured interface is loose or penetrates the target; fit requirements were not met.")
        points = trimesh.transform_points(self.points, geometry.transform_matrix(transform))
        penetration = None
        if self.volume_available:
            penetration = penetration_fraction(self.base_solid, points, self.size * 0.003)
            if penetration > 0.05: failures.append("B penetrates solid material in A beyond the sample limit.")
        else:
            warnings.append("Mesh A is open: solid-material containment cannot be verified reliably.")
        reverse_penetration = None
        if self.product_volume_available:
            local_base = trimesh.transform_points(self.base_points[:128], np.linalg.inv(geometry.transform_matrix(transform)))
            reverse_penetration = penetration_fraction(self.product_solid, local_base, self.size * 0.003 / min(transform.scale))
            if reverse_penetration > 0.05: failures.append("A penetrates solid material in B beyond the sample limit.")
        if not self.product_closed: warnings.append("Mesh B is open: inspect topology and interior surfaces.")
        if self.deformation_required:
            failures.append("Required deformation cannot be solved by rigid placement.")
            warnings.append("This request needs deformation. Use a suitable simulator or edit the mesh.")
        if self.program.uncertainty: warnings.append(self.program.uncertainty)
        quality = float(np.exp(-np.mean([c["normalized_error"] ** 2 for c in checks]) / 2))
        collision = [v for v in (penetration, reverse_penetration) if v is not None]
        if collision: quality *= max(0, 1 - max(collision) * 4)
        return {"eligible": not failures, "geometry_score": round(quality, 6), "failures": failures, "warnings": list(dict.fromkeys(warnings)), "metrics": {"orientation_issues":self.orientation_issues,"direction_branch":getattr(self,"direction_branch",{}),"constraint_checks": checks, "penetration_fraction": max(v for v in (penetration, reverse_penetration) if v is not None) if penetration is not None or reverse_penetration is not None else None, "product_inside_base_fraction": penetration, "base_inside_product_fraction": reverse_penetration, "base_watertight": self.closed, "product_watertight": self.product_closed, "base_volume_test_available": self.volume_available, "surface_samples_product": len(self.points), "scale_evidence":self.scale_evidence, "scale_factor":float(np.exp(np.mean(np.log(transform.scale)))),"scale_axes":transform.scale,"axis_scale_ratio":axis_ratio,"interface_fit":interface_checks}}


def _build_program_candidates(base_path, product_path, plan, captures, base_transform, count=5, hints=None, goal_contract=None, search_context=None):
    baseline, baseline_diagnostics = geometry.solve_placement(base_path, product_path, plan, captures, base_transform)
    diagnostics = {"architecture": VERSION, "program_source": "AI" if plan.program else "legacy translation", "program": (plan.program or compatibility_program(plan)).model_dump(), "baseline": baseline_diagnostics}
    diagnostics["goal_contract"]=goal_contract
    try:
        from .goal_contract import enforce
        enforce(plan,goal_contract)
        solver = ProgramSolver(base_path, product_path, base_transform, plan, captures, hints, require_scale=True,search_context=search_context)
        from .spatial_search import SearchDeadline
        branch_seeds=[];branch_errors=[];pruned_directions=[]
        for branch in solver.branches():
            rejected=branch.insertion_feasibility()
            if rejected:
                pruned_directions.append({"branch":getattr(branch,"direction_branch",{}),"reasons":rejected});continue
            try:branch_seeds.append((branch,branch.seed()))
            except SearchDeadline as error:
                branch_errors.append(error);break
            except ValueError as error:branch_errors.append(error)
        if not branch_seeds:raise (branch_errors[0] if branch_errors else ValueError("No insertion side can satisfy measured depth and bounded visual fit."))
    except ValueError as exc:
        # Preserve a reviewable result, never pretend feature extraction succeeded.
        from .scale_policy import ScaleEvidenceError
        from .goal_contract import GoalContractError
        from .spatial_search import SearchDeadline
        kind="budget_error" if isinstance(exc,SearchDeadline) else "goal_error" if isinstance(exc,GoalContractError) else "scale_error" if isinstance(exc,ScaleEvidenceError) else "feature_error"
        diagnostics[kind] = str(exc)
        from .geometry_feedback import serializable
        diagnostics.update(serializable(getattr(exc,"measurements",{})))
        if "solver" in locals():
            diagnostics["resolved_features"]=serializable(solver.features)
        return [{"id": f"c{i}", "label": "Constraints unresolved: revise target or direction", "baseline": i == 0, "transform": baseline.model_dump(), "optimization": {"performed": False, "method": "feature extraction failed"}, "eligible": False, "geometry_score": 0, "failures": [str(exc)], "warnings": ["Unverified features are never applied automatically."], "metrics": {"constraint_checks": []}} for i in range(1)], diagnostics
    diagnostics["scale_evidence"] = solver.scale_evidence
    diagnostics["inferred_constraints"] = solver.inferred_constraints
    diagnostics["effective_program"] = solver.program.model_dump()
    from .geometry_feedback import serializable
    diagnostics["resolved_features"] = serializable(solver.features)
    diagnostics["orientation_issues"]=solver.orientation_issues
    pool=[]
    for branch,branch_seed in branch_seeds:
        from .spatial_search import SearchDeadline
        roll_count=max(1,min(8,search_context.quota//max(1,len(branch_seeds)))) if search_context else (1 if count==1 else 4)
        angles=[0. if i<3 else 2*np.pi*(((i-2)*.6180339887498949)%1) for i in range(roll_count)]
        for start_index,yaw in enumerate(angles):
            if search_context:
                try:
                    if not search_context.start():break
                except SearchDeadline:break
            variant=branch_seed.model_copy(deep=True)
            factor=1. if start_index==0 else np.exp(-.15) if start_index==1 else np.exp(.15) if start_index==2 else np.exp(.2*(2*((start_index*.41421356237)%1)-1))
            variant.scale=(np.asarray(branch_seed.scale)*factor).tolist()
            first=next((c for c in branch.program.constraints if c.kind in {"coincident","contact","clearance","offset"}),None)
            if first:
                feature=branch.features[first.b];rot=Rotation.from_euler("XYZ",branch_seed.rotation)
                anchor=rot.apply(feature["center"]*np.asarray(branch_seed.scale))+branch_seed.position
                variant.position=(anchor-rot.apply(feature["center"]*np.asarray(variant.scale))).tolist()
            if yaw:
                first=next((c for c in branch.program.constraints if c.kind in {"align","contact","clearance"}),None)
                axis=branch.features[first.a]["axis"] if first else branch.base_rotation.apply([0,1,0])
                if axis is None:axis=branch.base_rotation.apply([0,1,0])
                rotated=Rotation.from_rotvec(np.asarray(axis)*yaw)*Rotation.from_euler("XYZ",variant.rotation)
                if first:
                    b=branch.features[first.b];anchor=Rotation.from_euler("XYZ",variant.rotation).apply(b["center"]*np.asarray(variant.scale))+variant.position
                    variant.position=(anchor-rotated.apply(b["center"]*np.asarray(variant.scale))).tolist()
                variant.rotation=rotated.as_euler("XYZ").tolist()
            if branch.orientation_issues:
                transform=variant
                objective=0. if search_context and search_context.exhausted else float(np.mean(branch.residuals(variant)**2))
                optimization={"performed":False,"method":"diagnostic direction variant; re-ground contradictory axes first",
                    "objective_before":objective,"objective_after":objective,"evaluations":0}
            else:
                try:transform,optimization=branch.optimize(variant,_coarse=count==1 and search_context is None)
                except SearchDeadline:
                    transform=variant;optimization={"performed":False,"method":"search deadline; diagnostic seed retained","objective_before":0.,"objective_after":1e9,"evaluations":0}
            checked=branch.check(transform)
            from .orientation import repair_feedback
            checked["repair_feedback"]=repair_feedback(checked)
            pool.append({"transform":transform.model_dump(),"optimization":optimization,**checked,
                "resolved_features":serializable(branch.features),"search_seed":{"branch":getattr(branch,"direction_branch",{}),"yaw":yaw,"scale_factor":factor}})
    if not pool:
        branch,seed=branch_seeds[0]
        pool=[{"transform":seed.model_dump(),"optimization":{"performed":False,"method":"budget exhausted; verified seed only","objective_after":1e9},
            **branch.check(seed),"resolved_features":serializable(branch.features),"search_seed":{"branch":getattr(branch,"direction_branch",{}),"yaw":0}}]
    pool.sort(key=lambda c:(not c["eligible"],len(c["failures"]),-c["geometry_score"],c["optimization"]["objective_after"]))
    def equivalent(first,second):
        a,b=first["transform"],second["transform"]
        return (np.linalg.norm(np.asarray(a["position"])-b["position"])/solver.size<.015 and
            np.max(np.abs(np.log(np.asarray(a["scale"])/b["scale"])))<.035 and
            (Rotation.from_euler("XYZ",a["rotation"])*Rotation.from_euler("XYZ",b["rotation"]).inv()).magnitude()<np.deg2rad(8))
    unique=[]
    for candidate in pool:
        if not any(equivalent(candidate,old) for old in unique):unique.append(candidate)
    proposals=unique[:max(1,count-1)] if count>1 else unique[:1]
    candidates=[]
    if count>1:
        objective=0. if search_context is not None else float(np.mean(solver.residuals(baseline)**2))
        candidates.append({"id":"c0","label":"Single-pass baseline","baseline":True,"transform":baseline.model_dump(),
            "optimization":{"performed":False,"method":"single-pass baseline","objective_before":objective,"objective_after":objective,"evaluations":0},
            "resolved_features":serializable(branch_seeds[0][0].features),**branch_seeds[0][0].check(baseline)})
    for candidate in proposals:
        candidate.update(id="c"+str(len(candidates)),label="Measured spatial proposal",baseline=False)
        candidates.append(candidate)
    diagnostics["search"]={"direction_branches":len(solver.branches()),"pruned_directions":pruned_directions,"optimized_starts":sum(c["optimization"]["performed"] for c in pool),"sampled_starts":len(pool),"distinct_poses":len(unique),"returned":len(candidates),"deduplicated":len(pool)-len(unique)}
    return candidates,diagnostics


def build_candidates(base_path,product_path,plan,captures,base_transform,count=5,hints=None,goal_contract=None,budget=None):
 """Beam over spatial programs + local anchor variants, then continuous pose search."""
 import copy,json
 from .schemas import SearchBudget
 from .spatial_search import SearchContext,SearchDeadline,local_programs
 options=SearchBudget.model_validate(budget or {})
 context=SearchContext(max_starts=options.max_starts,max_seconds=options.max_seconds,max_programs=options.max_programs)
 queue=[("primary","LLM primary program",plan)]
 for item in plan.hypotheses:
  proposal=plan.model_copy(update={"program":item.program,"hypotheses":[]})
  queue.append((item.id,item.reason,proposal))
 visited=set();pool=[];trace=[];main=None;baseline=None
 while queue and context.started_programs<context.max_programs and context.starts<context.max_starts:
  name,reason,proposal=queue.pop(0)
  signature=json.dumps((proposal.program or compatibility_program(proposal)).model_dump(),sort_keys=True)
  if signature in visited:continue
  visited.add(signature)
  try:context.program(context.max_programs-context.started_programs)
  except SearchDeadline:break
  try:
   found,diag=_build_program_candidates(base_path,product_path,proposal,captures,base_transform,max(2,count),hints,goal_contract,context)
  except SearchDeadline:
   context.exhausted=True;break
  if main is None:main=diag
  trace.append({"id":name,"source":reason,"eligible":sum(c["eligible"] for c in found),
   "error":diag.get("feature_error") or diag.get("goal_error") or diag.get("scale_error"),"search":diag.get("search",{})})
  for candidate in found:
   candidate=copy.deepcopy(candidate)
   candidate["hypothesis"]={"id":name,"source":reason}
   candidate["executed_program"]=(proposal.program or compatibility_program(proposal)).model_dump()
   if candidate.get("baseline"):
    if baseline is None:baseline=candidate
   else:pool.append(candidate)
  if context.exhausted:break
  # New local variants of the current best interpretation combine frame repair
  # with anchor/region changes. They are not only +/- normal flips.
  if proposal.program and not any(c["eligible"] for c in found):
   best=next((c for c in sorted(found,key=lambda c:(not c["eligible"],bool(c.get("metrics",{}).get("orientation_issues")),-c["geometry_score"],len(c["failures"]))) if not c.get("baseline")),found[0])
   refinements=local_programs(proposal,base_path,product_path,best.get("resolved_features",diag.get("resolved_features",{})),base_transform)
   from .spatial_search import within_original_support
   refinements=[item for item in refinements if within_original_support(plan,item[2],base_path,product_path)]
   queue=(queue+[(name+"/"+n,why,p) for n,why,p in refinements]) if plan.hypotheses else ([(name+"/"+n,why,p) for n,why,p in refinements]+queue)
  elif proposal.program and len(pool)<count:
   queue+=local_programs(proposal,base_path,product_path,diag.get("resolved_features",{}),base_transform)
 if main is None:
  # A valid request always gets a reviewable legacy pose, never a successful
  # result invented because its search budget ended.
  old,details=geometry.solve_placement(base_path,product_path,plan,captures,base_transform)
  main={"architecture":VERSION,"budget_exhausted":True,"baseline":details}
  baseline={"id":"c0","baseline":True,"label":"Search budget exhausted","transform":old.model_dump(),"eligible":False,
   "geometry_score":0.,"failures":["Search budget ended before a spatial program was verified."],"warnings":[],
   "metrics":{},"optimization":{"performed":False}}
 # Keep a single diagnostic baseline if all programs failed extraction.
 pool.sort(key=lambda c:(not c["eligible"],bool(c.get("metrics",{}).get("orientation_issues")),-c["geometry_score"],len(c["failures"])))
 unique=[]
 size=max(float(max(geometry.flat_mesh(base_path).extents*np.asarray(base_transform.scale))),1e-8)
 def equivalent(a,b):
  x,y=a["transform"],b["transform"]
  return (np.linalg.norm(np.asarray(x["position"])-y["position"])<size*.015 and
   np.max(np.abs(np.log(np.asarray(x["scale"])/y["scale"])))<.035 and
   (Rotation.from_euler("XYZ",x["rotation"])*Rotation.from_euler("XYZ",y["rotation"]).inv()).magnitude()<np.deg2rad(8))
 for candidate in pool:
  if not any(equivalent(candidate,old) for old in unique):unique.append(candidate)
 output=([baseline] if count>1 and baseline is not None else [])+unique[:max(1,count-(1 if count>1 and baseline else 0))]
 if not output and baseline is not None:output=[baseline]
 for index,candidate in enumerate(output):candidate["id"]="c"+str(index)
 best=next((c for c in sorted(output,key=lambda c:(not c["eligible"],bool(c.get("metrics",{}).get("orientation_issues")),-c["geometry_score"],len(c["failures"]))) if not c.get("baseline")),output[0])
 main["primary_orientation_issues"]=main.get("orientation_issues",[])
 main["resolved_features"]=best.get("resolved_features",main.get("resolved_features",{}))
 main["orientation_issues"]=best.get("metrics",{}).get("orientation_issues",[])
 main["best_geometric_hypothesis"]=best.get("hypothesis")
 if best.get("metrics",{}).get("scale_evidence"):main["scale_evidence"]=best["metrics"]["scale_evidence"]
 main["hypothesis_search"]={**context.report(),"programs_tested":trace,"distinct_poses":len(unique),"pending_programs":len(queue),
  "dimensions":["semantic region","anchor","feature interpretation","axis sign","rotation","translation","uniform scale","bounded per-axis scale"]}
 if any(c["eligible"] for c in output):
  for key in ("feature_error","scale_error","goal_error"):main.pop(key,None)
 return output,main
