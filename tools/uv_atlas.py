"""Bounded-memory UV atlas on all original faces; no mesh simplification."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import trimesh

POLICY="spatial-patches-20000-parallel-v3"

def partitions(mesh,max_faces=20000):
    centers=np.asarray(mesh.triangles_center,dtype=np.float32)
    pending=[np.arange(len(mesh.faces))];result=[]
    while pending:
        ids=pending.pop()
        if len(ids)<=max_faces:result.append(ids);continue
        axis=int(np.argmax(np.ptp(centers[ids],axis=0)))
        split=len(ids)//2;ordered=np.argpartition(centers[ids,axis],split)
        pending.extend([ids[ordered[:split]],ids[ordered[split:]]])
    return result

def rectangles(weights):
    weights=np.asarray(weights,float);boxes={}
    def split(ids,rect):
        if len(ids)==1:boxes[ids[0]]=rect;return
        total=weights[ids].sum();partial=np.cumsum(weights[ids])
        index=int(np.clip(np.searchsorted(partial,total/2)+1,1,len(ids)-1))
        first,second=ids[:index],ids[index:];fraction=float(weights[first].sum()/total)
        x,y,w,h=rect
        if w>=h:split(first,(x,y,w*fraction,h));split(second,(x+w*fraction,y,w*(1-fraction),h))
        else:split(first,(x,y,w,h*fraction));split(second,(x,y+h*fraction,w,h*(1-fraction)))
    split(list(range(len(weights))),(0.,0.,1.,1.))
    return boxes


def _parametrize(payload):
    import xatlas
    vertices,local_faces=payload
    positions,first,weld=np.unique(vertices,axis=0,return_index=True,return_inverse=True)
    local_faces=weld[local_faces]
    mapping,indices,uv=xatlas.parametrize(np.asarray(positions,np.float32),local_faces.astype(np.uint32))
    return first[np.asarray(mapping)],np.asarray(indices),np.asarray(uv)

def _requests(mesh,groups):
    for ids in groups:
        used,inverse=np.unique(mesh.faces[ids],return_inverse=True)
        yield np.asarray(mesh.vertices[used]),inverse.reshape(-1,3)

def unwrap(mesh,directory=None,resolution=1024,progress=None,max_faces=20000):
    import xatlas
    if isinstance(mesh,trimesh.Scene):mesh=mesh.to_geometry()
    began=time.monotonic()
    identity=hashlib.sha256(np.asarray(mesh.vertices,dtype=np.float32).tobytes()+
        np.asarray(mesh.faces,dtype=np.uint32).tobytes()+POLICY.encode()).hexdigest()
    cache=Path(directory)/"uv-atlas.npz" if directory else None
    report={"policy":POLICY,"original_faces":len(mesh.faces),"decimation":False}
    if cache and cache.is_file():
        with np.load(cache,allow_pickle=False) as saved:
            if str(saved["identity"])==identity:
                output=mesh.copy();output.vertices=np.asarray(mesh.vertices)[saved["mapping"]]
                output.faces=saved["faces"];output.visual=trimesh.visual.texture.TextureVisuals(uv=saved["uv"])
                report.update(cache_reused=True,seconds=time.monotonic()-began)
                return output,report
    groups=partitions(mesh,max_faces)
    weights=[max(float(mesh.area_faces[ids].sum()),1e-12) for ids in groups]
    boxes=rectangles(weights);mappings=[];faces=[];uvs=[];offset=0
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    from contextlib import nullcontext
    import os
    workers=min(2,os.cpu_count() or 1) if len(groups)>3 else 1
    manager=ProcessPoolExecutor(max_workers=workers,mp_context=get_context("spawn")) if workers>1 else nullcontext()
    with manager as executor:
      results=executor.map(_parametrize,_requests(mesh,groups)) if executor else map(_parametrize,_requests(mesh,groups))
      for index,(ids,result) in enumerate(zip(groups,results)):
        used=np.unique(mesh.faces[ids])
        local_faces=mesh.faces[ids]
        mapping,indices,uv=result
        mapping=used[mapping]
        indices=np.asarray(indices)
        # Verify exact triangle correspondence before accepting a new UV atlas.
        if indices.shape!=local_faces.shape or not np.array_equal(np.asarray(mesh.vertices)[mapping[indices]],mesh.triangles[ids]):
            raise ValueError("UV atlas changed source triangle correspondence.")
        uv=np.asarray(uv,dtype=np.float64)
        if not np.isfinite(uv).all():raise ValueError("UV atlas contains non-finite coordinates.")
        x,y,w,h=boxes[index]
        padding=min(2/resolution,w*.025,h*.025)
        uv=uv*np.array([w-2*padding,h-2*padding])+[x+padding,y+padding]
        mappings.append(mapping);faces.append(indices+offset);uvs.append(uv)
        offset+=len(mapping)
        if progress:progress(index+1,len(groups))
    mapping=np.concatenate(mappings);indices=np.concatenate(faces);uv=np.concatenate(uvs)
    output=mesh.copy();output.vertices=np.asarray(mesh.vertices)[mapping];output.faces=indices
    output.visual=trimesh.visual.texture.TextureVisuals(uv=uv)
    assert len(output.faces)==len(mesh.faces)
    report.update(workers=workers,patches=len(groups),max_faces_per_patch=max_faces,output_faces=len(output.faces),
        seconds=round(time.monotonic()-began,3),cache_reused=False)
    if cache:
        temporary=cache.with_suffix(".tmp.npz")
        np.savez_compressed(temporary,identity=np.array(identity),mapping=mapping,faces=indices,uv=uv)
        temporary.replace(cache)
        cache.with_suffix(".json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return output,report
