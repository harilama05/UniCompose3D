"""Measured interface gaps on original mesh sections, independent of object labels.

Profiles are sampled once; optimization evaluates interpolated local boundaries.
Uniform relative scale is the default. Optional independent axis scales change
shape explicitly; neither mode fills holes, edits vertices or certifies sizing.
"""
from __future__ import annotations
import numpy as np
import trimesh
from . import aperture,geometry

ANGLES=96
FIT_LIMITS={"median_gap_ratio":.10,"p90_gap_ratio":.22,"intrusion_fraction":.05,"supported_fraction":.95}

def radial_distances(segments, origin, axis, outer=False):
    """Positive ray/segment intersections in a plane, preserving missing sectors."""
    _,basis=aperture.plane_basis(axis)
    starts=(segments[:,0]-origin)@basis
    edges=(segments[:,1]-segments[:,0])@basis
    def cross(a,b):return a[...,0]*b[...,1]-a[...,1]*b[...,0]
    values=np.full(ANGLES,np.nan)
    for i,angle in enumerate(np.arange(ANGLES)*2*np.pi/ANGLES):
        direction=np.array([np.cos(angle),np.sin(angle)])
        den=cross(direction,edges);valid=np.abs(den)>1e-12
        distance=np.zeros(len(edges));fraction=np.zeros(len(edges))
        distance[valid]=cross(starts[valid],edges[valid])/den[valid]
        fraction[valid]=cross(starts[valid],direction)/den[valid]
        hits=distance[valid&(distance>1e-9)&(fraction>=-1e-8)&(fraction<=1+1e-8)]
        if len(hits):values[i]=np.max(hits) if outer else np.min(hits)
    return values


def radial_target_frame(mesh,point,axis,reference_radius):
    segments=trimesh.intersections.mesh_plane(mesh,axis,point)
    if not len(segments):raise ValueError("No target plane intersections.")
    radii=radial_distances(segments,point,axis)
    valid=np.isfinite(radii)
    if np.mean(valid)<.96:raise ValueError("Insufficient target angular support.")
    index=np.arange(ANGLES)
    radii=np.interp(index,index[valid],radii[valid],period=ANGLES)
    if radii.min()<reference_radius*.12 or radii.max()>reference_radius*1.4:
        raise ValueError("Radial target differs from the measured local part.")
    _,basis=aperture.plane_basis(axis)
    angles=index*2*np.pi/ANGLES
    line=point+np.column_stack([np.cos(angles)*radii,np.sin(angles)*radii])@basis.T
    return {"center":point,"line":np.vstack([line,line[0]]),
        "max_radius":float(radii.max()),"radial_recovery":True}

class MeasuredInterface:
    def __init__(self,section_mesh,opening_mesh,section,opening,seed,opening_on_product,target):
        self.forward=opening_on_product
        self.cavity=opening["kind"]=="cavity"
        self.insertion_depth=float(section.get("insertion_depth",0))
        self.roof=np.asarray(opening.get("roof",opening["center"]))
        if self.cavity and self.insertion_depth<=0:
            raise ValueError("A closed-end fit needs a measured insertion depth.")
        self.center=np.asarray(section["center"])
        self.axis,self.basis=aperture.plane_basis(section["axis"])
        self.target=target-1
        self.radius=float(section["radius"])
        self.a,self.b=section,opening
        axis=np.asarray(opening["axis"]);origin=np.asarray(opening["center"])
        _,basis=aperture.plane_basis(axis)
        angles=np.arange(ANGLES)*2*np.pi/ANGLES
        directions=np.column_stack([np.cos(angles),np.sin(angles)])
        points=[];slices=[];rim=[]
        # Centre plus independently measured neighbouring planes.
        for fraction in ((.02,.15,.35,.55,.75) if self.cavity else (-.3,-.15,0,.15,.3)):
            cut_origin=origin+axis*opening["axial_span"]*fraction
            segments=trimesh.intersections.mesh_plane(opening_mesh,axis,cut_origin)
            if not len(segments):continue
            radii=radial_distances(segments,cut_origin,axis)
            valid=np.isfinite(radii)
            if np.mean(valid)<.6:continue
            sample=cut_origin+(directions[valid]*radii[valid,None])@basis.T
            points.extend(sample);rim.extend([fraction<=.151]*len(sample))
            slices.append({"fraction":fraction,"coverage":float(np.mean(valid)),
                           "inner_radius_p10":float(np.quantile(radii[valid],.1)),
                           "inner_radius_p90":float(np.quantile(radii[valid],.9))})
        if len(slices)<3:raise ValueError("Insufficient measured opening contours across its thickness.")
        self.points=np.asarray(points);self.slices=slices;self.rim=np.asarray(rim,dtype=bool)
        mapped=self.map(seed);z=(mapped-self.center)@self.axis
        spread=max(float(np.max(np.abs(z))),self.radius*.15)+self.radius*.2
        grid=np.linspace(-spread*.2,spread,21) if self.cavity else np.linspace(-spread,spread,11)
        profiles=[];centers=[];valid_grid=[];self.target_recoveries=0;self.profile_trials=[]
        for value in grid:
            trial={"offset":float(value)};self.profile_trials.append(trial)
            point=self.center+self.axis*value
            try:
                try:frame=aperture.target_section(section_mesh,point,self.axis)
                except ValueError:frame=radial_target_frame(section_mesh,point,self.axis,self.radius)
                if frame["max_radius"]>self.radius*1.4 or np.linalg.norm(frame["center"]-point)>self.radius*.5:
                    # A tiny closed reconstruction fragment can be selected in
                    # place of the local outer contour. Verify raw triangle/plane
                    # intersections before declaring a real support gap.
                    frame=radial_target_frame(section_mesh,point,self.axis,self.radius)
                    trial["recovery"]="raw measured angular intersections"
                line=frame["line"]
                segments=np.stack([line[:-1],line[1:]],axis=1)
                radii=radial_distances(segments,frame["center"],self.axis,outer=True)
                if not np.all(np.isfinite(radii)):
                    frame=radial_target_frame(section_mesh,point,self.axis,self.radius)
                    line=frame["line"];segments=np.stack([line[:-1],line[1:]],axis=1)
                    radii=radial_distances(segments,frame["center"],self.axis,outer=True)
                if frame.get("radial_recovery"):self.target_recoveries+=1
                if not np.all(np.isfinite(radii)):continue
                profiles.append(radii);centers.append(frame["center"]);valid_grid.append(value);trial["verified"]=True
            except ValueError as error:trial["error"]=str(error)
        if len(valid_grid)<3:raise ValueError("Insufficient independent local sections for fitting.")
        grid=np.asarray(valid_grid)
        # A missing middle section must not be silently bridged.
        step=float(grid[1]-grid[0])
        nearest=int(np.argmin(np.abs(grid)))
        lo=hi=nearest
        while lo>0 and grid[lo]-grid[lo-1]<step*1.1:lo-=1
        while hi<len(grid)-1 and grid[hi+1]-grid[hi]<step*1.1:hi+=1
        if hi-lo<2:raise ValueError("The section region is discontinuous; choose another target or direction.")
        self.grid=grid[lo:hi+1]
        self.profiles=np.asarray(profiles)[lo:hi+1]
        self.centers=np.asarray(centers)[lo:hi+1]
        self.samples=len(self.points)

    def map(self,transform):
        matrix=geometry.transform_matrix(transform)
        if not self.forward:matrix=np.linalg.inv(matrix)
        return trimesh.transform_points(self.points,matrix)

    def gaps(self,transform):
        points=self.map(transform)
        z=(points-self.center)@self.axis
        zi=np.clip(z,self.grid[0],self.grid[-1])
        high=np.clip(np.searchsorted(self.grid,zi,side="right"),1,len(self.grid)-1)
        low=high-1;mix=(zi-self.grid[low])/(self.grid[high]-self.grid[low])
        centers=self.centers[low]*(1-mix[:,None])+self.centers[high]*mix[:,None]
        xy=(points-centers)@self.basis
        angle=(np.arctan2(xy[:,1],xy[:,0])%(2*np.pi))*ANGLES/(2*np.pi)
        left=np.floor(angle).astype(int)%ANGLES;right=(left+1)%ANGLES;blend=angle-np.floor(angle)
        rlow=self.profiles[low,left]*(1-blend)+self.profiles[low,right]*blend
        rhigh=self.profiles[high,left]*(1-blend)+self.profiles[high,right]*blend
        target=rlow*(1-mix)+rhigh*mix
        gaps=(np.linalg.norm(xy,axis=1)-target)/np.maximum(target,1e-8)
        outside=np.abs(z-zi)/max(self.radius,1e-8)
        return gaps,outside

    def active(self,transform):
        if not self.cavity:return np.ones(self.samples,dtype=bool)
        z=(self.map(transform)-self.center)@self.axis
        return (z>=-self.radius*.12)&(z<=self.insertion_depth*.92)

    def roof_gap(self,transform):
        matrix=geometry.transform_matrix(transform)
        if not self.forward:matrix=np.linalg.inv(matrix)
        roof=trimesh.transform_points([self.roof],matrix)[0]
        return float((np.dot(roof-self.center,self.axis)-self.insertion_depth)/self.insertion_depth)

    def residuals(self,transform,tolerance):
        gaps,outside=self.gaps(transform)
        error=gaps-self.target
        # Prevent least-squares from buying a smaller average gap with intrusion.
        active=self.active(transform)
        wanted=np.where(self.rim,error,np.minimum(gaps,0)*8) if self.cavity else error
        weighted=np.where(gaps<0,error*8,wanted)*active
        residual=np.r_[weighted/max(tolerance,.025),outside*active/.04]/np.sqrt(max(self.samples,1))
        scored=active&self.rim if self.cavity else active
        if np.any(scored):
            fitted=gaps[scored]
            # The optimizer must target the same fixed acceptance criteria as
            # the independent checker, rather than merely matching a soft prior.
            residual=np.r_[residual,max(0,float(np.median(fitted))-FIT_LIMITS["median_gap_ratio"])/.025,
                max(0,float(np.quantile(fitted,.9))-FIT_LIMITS["p90_gap_ratio"])/.04]
        else:residual=np.r_[residual,10.,10.]
        if self.cavity:
            gap=self.roof_gap(transform)
            residual=np.r_[residual,(gap*8 if gap<0 else max(0,gap-.5))/.12,
                max(0,.35-float(np.mean(active)))*30]
        return residual

    def feasibility_margins(self,transform):
        """Continuous quantile counterparts of the fixed acceptance tests."""
        gaps,outside=self.gaps(transform);active=self.active(transform)
        scored=active&self.rim if self.cavity else active
        if not np.any(active) or not np.any(scored):return np.full(7,-10.)
        margin=1e-4
        values=[FIT_LIMITS["median_gap_ratio"]-float(np.median(gaps[scored]))-margin,
            FIT_LIMITS["p90_gap_ratio"]-float(np.quantile(gaps[scored],.9))-margin,
            float(np.quantile(gaps[active],.05))+.025-margin,
            .02-float(np.quantile(outside[active],.95))-margin]
        if self.cavity:
            roof=self.roof_gap(transform)
            values.extend([.5-roof-margin,roof+.03-margin,float(np.mean(active))-.35-margin])
        else:values.extend([1.,1.,1.])
        return np.asarray(values)

    def measure(self,transform):
        gaps,outside=self.gaps(transform)
        active=self.active(transform)
        active_fraction=float(np.mean(active))
        if not active.any():active=np.ones(self.samples,dtype=bool)
        all_gaps=gaps[active]
        scored=active&self.rim if self.cavity else active
        if not scored.any():scored=active
        scored_gaps=gaps[scored]
        outside=outside[active]
        gaps=scored_gaps
        median=float(np.median(gaps));p90=float(np.quantile(gaps,.9))
        penetration=float(np.mean(all_gaps<-.025))
        supported=float(np.mean(outside<.02))
        limits=dict(FIT_LIMITS)
        snug=median<=limits["median_gap_ratio"] and p90<=limits["p90_gap_ratio"]
        safe=penetration<=limits["intrusion_fraction"] and supported>=limits["supported_fraction"]
        roof_gap=self.roof_gap(transform) if self.cavity else None
        if self.cavity:safe=safe and -.03<=roof_gap<=.5 and active_fraction>=.35
        return {"method":"multi-section radial surface gaps","sample_count":self.samples,
            "opening_topology":"closed-end cavity" if self.cavity else "through opening",
            "roof_gap_ratio":roof_gap,"active_fraction":active_fraction,"fit_region":"mouth" if self.cavity else "full interface","wall_gap_median_ratio":float(np.median(all_gaps)),
            "opening_slices":self.slices,"target_section_count":len(self.grid),"target_profile_range":[float(self.grid[0]),float(self.grid[-1])],"target_profile_trials":self.profile_trials,"target_radial_recoveries":self.target_recoveries,
            "gap_p10_ratio":float(np.quantile(gaps,.1)),"gap_median_ratio":median,
            "gap_p90_ratio":p90,"intrusion_fraction":penetration,"supported_fraction":supported,
            "target_gap_ratio":self.target,"limits":limits,"snug":snug,
            "eligible":bool(snug and safe),"shape_mismatch":bool(safe and not snug),
            "scale_policy":"relative to the fixed base; common A/B scaling does not change fit",
            "scope":"sampled local interface; not a global solid collision proof"}
