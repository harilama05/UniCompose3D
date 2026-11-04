"""Bounded screen correction plus explicit nearest-surface tool feedback."""
import numpy as np

def nearest_camera_landmark(mesh,capture,uv):
    projection=np.asarray(capture.projection).reshape(4,4,order="F")
    world=np.asarray(capture.world).reshape(4,4,order="F")
    matrix=projection@np.linalg.inv(world)
    vertices=np.asarray(mesh.vertices);target=np.asarray(uv);best=None
    for offset in range(0,len(vertices),32768):
        chunk=vertices[offset:offset+32768]
        clip=np.column_stack([chunk,np.ones(len(chunk))])@matrix.T
        front=clip[:,3]>1e-8
        ndc=clip[:,:3]/np.maximum(clip[:,3,None],1e-8)
        image=np.column_stack([(ndc[:,0]+1)/2,(1-ndc[:,1])/2])
        distance=np.linalg.norm(image-target,axis=1)
        distance[~front|(np.abs(ndc[:,2])>1)]=np.inf
        index=int(np.argmin(distance))
        if best is None or distance[index]<best[0]:
            best=(float(distance[index]),image[index],chunk[index])
    if best is None or not np.isfinite(best[0]):return None
    return {"uv":best[1].tolist(),"distance_uv":best[0],"point":best[2].tolist()}

def snap_camera_anchor(mesh,capture,uv,max_distance=.012):
    from .geometry import camera_anchor
    candidate=nearest_camera_landmark(mesh,capture,uv)
    if candidate is None or candidate["distance_uv"]>max_distance:return None
    return camera_anchor(mesh,capture,candidate["uv"])
