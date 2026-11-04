"""Stable surface landmark IDs and visibility-checked canonical view labels."""
from functools import lru_cache
from pathlib import Path
import base64,io
import numpy as np
from PIL import Image,ImageDraw
from . import geometry

def landmarks(mesh,count=24):
 vertices=np.asarray(mesh.vertices)
 points=vertices[np.linspace(0,len(vertices)-1,min(12000,len(vertices)),dtype=int)]
 points=np.unique(points,axis=0)
 selected=[int(np.argmin(points[:,1]))]
 distances=np.full(len(points),np.inf)
 for _ in range(min(count,len(points))-1):
  distances=np.minimum(distances,np.sum((points-points[selected[-1]])**2,axis=1))
  distances[selected]=-1
  selected.append(int(np.argmax(distances)))
 lower=mesh.bounds[0];extent=np.maximum(mesh.extents,1e-8)
 return [{"id":f"L{i:02d}","point":p.tolist(),"bbox":((p-lower)/extent).tolist()}
  for i,p in enumerate(points[selected])]

@lru_cache(maxsize=4)
def _catalog(path,modified,size):
 return landmarks(geometry.flat_mesh(Path(path)))
def catalog(path):
 path=Path(path);info=path.stat()
 return _catalog(str(path),info.st_mtime_ns,info.st_size)

def reference(mesh,name):
 values={item["id"]:item["point"] for item in landmarks(mesh)}
 if name not in values:raise ValueError("Unknown measured landmark ID: "+str(name))
 return np.asarray(values[name])

def label_captures(paths,captures):
 output=[];meshes={}
 for capture in captures:
  path=paths[capture.slot]
  mesh=meshes.setdefault(capture.slot,geometry.flat_mesh(path)) if capture.slot not in meshes else meshes[capture.slot]
  try:
   source=Image.open(io.BytesIO(base64.b64decode(capture.image.split(",")[-1]))).convert("RGB")
   projection=np.asarray(capture.projection).reshape(4,4,order="F")
   world=np.asarray(capture.world).reshape(4,4,order="F")
   eye=world[:3,3]
   draw=ImageDraw.Draw(source)
   marked=[]
   for item in catalog(path):
    point=np.asarray(item["point"])
    clip=projection@np.linalg.inv(world)@np.r_[point,1]
    if clip[3]<=0:continue
    ndc=clip[:3]/clip[3]
    if np.max(np.abs(ndc[:2]))>.96 or abs(ndc[2])>1:continue
    direction=point-eye;distance=np.linalg.norm(direction)
    if distance<1e-8:continue
    hit=geometry.ray_hit(mesh,eye,direction/distance)
    if hit is None or np.linalg.norm(hit[0]-point)>max(mesh.extents)*.018:continue
    x,y=(ndc[0]+1)*source.width/2,(1-ndc[1])*source.height/2
    if any((x-u)**2+(y-v)**2<22**2 for u,v in marked):continue
    marked.append((x,y));draw.ellipse((x-3,y-3,x+3,y+3),fill="yellow",outline="black")
    draw.text((x+4,y-10),item["id"],fill="black",stroke_width=2,stroke_fill="white")
   stream=io.BytesIO();source.save(stream,"PNG")
   output.append(capture.model_copy(update={"image":"data:image/png;base64,"+base64.b64encode(stream.getvalue()).decode()}))
  except (ValueError,OSError,KeyError,np.linalg.LinAlgError):output.append(capture)
 return output


def annotate_vectors(captures,features,base_transform):
 """Show what raycast landmarks REALLY mean in 3D, not the plan's prose."""
 from .schemas import Transform
 output=[]
 inverse=np.linalg.inv(geometry.transform_matrix(Transform.model_validate(base_transform)))
 for view,capture in enumerate(captures):
  selected=[(name,f) for name,f in features.items() if f.get("slot")==capture.slot and f.get("axis") is not None]
  if not selected:continue
  try:
   image=Image.open(io.BytesIO(base64.b64decode(capture.image.split(",")[-1]))).convert("RGB")
   draw=ImageDraw.Draw(image)
   matrix=np.asarray(capture.projection).reshape(4,4,order="F")@np.linalg.inv(np.asarray(capture.world).reshape(4,4,order="F"))
   count=0;quality=0.
   for name,feature in selected[:6]:
    start=np.asarray(feature["center"]);end=start+np.asarray(feature["axis"])*min(.25,max(.08,feature["extent"]*.5))
    if capture.slot=="A":start,end=geometry.trimesh.transform_points([start,end],inverse)
    def project(p):
     clip=matrix@np.r_[p,1]
     if clip[3]<=0:return None
     return ((clip[0]/clip[3]+1)*image.width/2,(1-clip[1]/clip[3])*image.height/2)
    a,b=project(start),project(end)
    if a is None or b is None:continue
    draw.line([a,b],fill="red",width=3);draw.ellipse((b[0]-4,b[1]-4,b[0]+4,b[1]+4),fill="red")
    draw.text(a,name,fill="black",stroke_width=2,stroke_fill="white");count+=1;quality+=float(np.linalg.norm(np.asarray(a)-b))
   if not count:continue
   stream=io.BytesIO();image.save(stream,"PNG")
   output.append({"candidate_id":"measured_axes_"+str(view),"kind":"axes","view":view,
     "image":base64.b64encode(stream.getvalue()).decode(),"slot":capture.slot,"quality":quality})
  except (ValueError,OSError,KeyError,np.linalg.LinAlgError):continue
 return [max([x for x in output if x["slot"]==slot],key=lambda x:x["quality"]) for slot in ("A","B") if any(x["slot"]==slot for x in output)]
