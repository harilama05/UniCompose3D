"""Deterministic geometry cases with expected outcomes, no category dispatch."""
import numpy as np,trimesh
from unicompose3d.schemas import PlacementPlan,SpatialProgram,Transform
def cup():
 return trimesh.creation.revolve(np.array([[.30,0],[.34,0],[.34,.50],[0,.50],[0,.46],[.30,.46],[.30,0]]),sections=48)
def plan(features,constraints,**kwargs):
 return PlacementPlan(relation="generic spatial intent",base_anchor=[.5]*3,product_anchor=[.5]*3,
  rotation_degrees=[0]*3,relative_scale=1,confidence=.9,explanation="benchmark",
  program=SpatialProgram(features=features,constraints=constraints,**kwargs))
def cases():
 box=trimesh.creation.box()
 contact=plan([dict(id="a",slot="A",kind="surface",point=[.5,1,.5]),dict(id="b",slot="B",kind="surface",point=[.5,0,.5])],
  [dict(kind="contact",a="a",b="b",target=.005),dict(kind="proportion",a="a",b="b",target=.3)])
 yield "support",box,box,contact,Transform(),True
 yield "support_rotated_base",box,box,contact,Transform(rotation=[.4,.3,.2],scale=[1.2,.8,1.]),True
 offset=plan([dict(id="a",slot="A",kind="point"),dict(id="b",slot="B",kind="point")],
  [dict(kind="offset",a="a",b="b",offset=[1.1,0,0]),dict(kind="proportion",a="a",b="b",target=.25)])
 yield "beside",box,box,offset,Transform(),True
 attached=contact.model_copy(deep=True);attached.program.features[0].point=[1,.5,.5];attached.program.features[1].point=[0,.5,.5]
 yield "side_attachment",box,box,attached,Transform(),True
 body=trimesh.creation.cylinder(radius=.25,height=.4,sections=48);body.apply_translation([0,0,.2])
 for reverse in (False,True):
  for sign in (1,-1):
   section=dict(id="section",slot="B" if reverse else "A",kind="section",point=[1,.5,.12],end=[1,.5,.9 if sign>0 else 0])
   opening=dict(id="opening",slot="A" if reverse else "B",kind="cavity",point=[.5,.5,.04],end=[.5,.5,.8])
   a,b=("opening","section") if reverse else ("section","opening")
   p=plan([section,opening],[dict(kind="coincident",a=a,b=b),dict(kind="align",a=a,b=b,direction="either"),dict(kind="clearance",a=a,b=b,target=1.03)],fit_topology="cover")
   yield "cavity_"+("reverse_" if reverse else "")+("positive" if sign>0 else "negative"),cup() if reverse else body,body if reverse else cup(),p,Transform(),True
 through=plan([dict(id="a",slot="A",kind="section",point=[1,.5,.5],end=[1,.5,.9]),dict(id="b",slot="B",kind="aperture")],
  [dict(kind="coincident",a="a",b="b"),dict(kind="align",a="a",b="b"),dict(kind="clearance",a="a",b="b",target=1.03)],fit_topology="through")
 yield "through",body,trimesh.creation.torus(major_radius=.3,minor_radius=.04),through,Transform(),True
 yield "missing_through_opening",body,box,through,Transform(),False

 # Containment needs empty space, not merely an enclosing bounding box.
 region=plan([dict(id="a",slot="A",kind="region",point=[.5,.5,.35],extent=[.65,.65,.5]),dict(id="b",slot="B",kind="point")],
  [dict(kind="containment",a="a",b="b"),dict(kind="proportion",a="a",b="b",target=.25)])
 yield "empty_containment",cup(),box,region,Transform(),True
 yield "solid_is_not_container",box,box,region,Transform(),False
 conflict=plan([dict(id="s",slot="A",kind="section",point=[1,.5,.12],end=[1,.5,.9]),
  dict(id="c",slot="B",kind="cavity",point=[.5,.5,.04],end=[.5,.5,.8]),
  dict(id="x",slot="A",kind="axis",point=[.5,.5,.12],end=[.5,.5,.8]),
  dict(id="y",slot="B",kind="axis",point=[.1,.5,.2],end=[.9,.5,.2])],
  [dict(kind="coincident",a="s",b="c"),dict(kind="align",a="s",b="c"),dict(kind="clearance",a="s",b="c",target=1.03),dict(kind="align",a="x",b="y",direction="parallel")],fit_topology="cover")
 yield "contradictory_directions",body,cup(),conflict,Transform(),False

 for topology in ("cavity","aperture"):
  item=cup() if topology=="cavity" else trimesh.creation.torus(major_radius=.3,minor_radius=.04)
  left,right=item.copy(),item.copy();left.apply_translation([-.65,0,0]);right.apply_translation([.65,0,0])
  both=trimesh.util.concatenate([left,right])
  for sign in (-1,1):
   x=.5+sign*.65/float(both.extents[0])
   point=[x,.5,.04 if topology=="cavity" else .5]
   opening=dict(id="opening",slot="B",kind=topology,point=point,end=[x,.5,.8],extent=[.4,.7,1])
   section=dict(id="section",slot="A",kind="section",point=[1,.5,.12 if topology=="cavity" else .5],end=[1,.5,.9])
   p=plan([section,opening],[dict(kind="coincident",a="section",b="opening"),dict(kind="align",a="section",b="opening"),dict(kind="clearance",a="section",b="opening",target=1.03)],fit_topology="cover" if topology=="cavity" else "through")
   object.__setattr__(p,"_benchmark_region_check",{"feature":"opening","axis":0,"sign":sign,"minimum":.3})
   yield "selected_"+topology+("_left" if sign<0 else "_right"),body,both,p,Transform(),True
