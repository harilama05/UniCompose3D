"""Bounded composition agent state: original intent, measurements and rendered feedback."""
import base64
from pathlib import Path

MAX_ROUNDS = 3

def parent_state(engine, request):
    if not request.refinement_of:
        return None
    parent = engine.store.read("compositions",request.refinement_of)
    same = (parent["base_id"]==request.base_id and parent["product_id"]==request.product_id
        and parent["base_transform"]==request.base_transform.model_dump()
        and parent["instruction"]==request.instruction
        and parent.get("hints",{})==request.hints.model_dump())
    if not same:
        raise ValueError("A revision must preserve A/B, the base pose, original instruction, and user hints.")
    if int(parent.get("agent_round",1)) >= MAX_ROUNDS:
        raise ValueError("The agent reached its three-round limit. Review the failures before starting another run.")
    return parent

def grounding_images(captures,diagnostics):
    """Show the failed request and nearest measured surface, without moving it."""
    import io
    from PIL import Image,ImageDraw
    issues=list(diagnostics.get("feature_issues",[]))
    for attempt in diagnostics.get("previous_attempts",[]):
        issues+=attempt.get("diagnostics",{}).get("feature_issues",[])
    images=[]
    for issue in issues[:3]:
        measured=issue.get("measurements",{})
        request=issue.get("request",{})
        view=measured.get("view",request.get("view",-1))
        uv=measured.get("requested_uv",request.get("uv"))
        if uv is None or not 0<=view<len(captures):continue
        capture=captures[view]
        if capture.slot!=issue["slot"]:continue
        try:
            blob=base64.b64decode(capture.image.split(",",1)[-1],validate=True)
            with Image.open(io.BytesIO(blob)) as source:image=source.convert("RGB")
            draw=ImageDraw.Draw(image)
            def marker(point,color):
                x,y=point[0]*image.width,point[1]*image.height
                draw.ellipse((x-7,y-7,x+7,y+7),outline=color,width=3)
            marker(uv,"red")
            nearest=measured.get("nearest_surface")
            if nearest:marker(nearest["uv"],"lime")
            stream=io.BytesIO();image.save(stream,"PNG")
            images.append({"candidate_id":"grounding_"+issue["feature_id"],"kind":"grounding",
                "view":view,"image":base64.b64encode(stream.getvalue()).decode()})
        except (ValueError,OSError):continue
    return images


def feedback(engine, parent, captures=None):
    state={key:parent.get(key) for key in ("instruction","plan","diagnostics","candidates",
        "summary","warnings","base_transform","evaluation_status","api_calls","goal_contract")}
    images=[]
    chosen=parent.get("selected_candidate_id") or parent.get("suggested_candidate_id")
    for item in parent.get("renders",[]):
        # Failed alternatives also contain evidence needed to repair the plan.
        path=engine.store.path("compositions",parent["id"])/(item["candidate_id"]+".png")
        if path.is_file():
            images.append({"candidate_id":item["candidate_id"],
                "image":base64.b64encode(path.read_bytes()).decode()})
    if captures:
        images+=grounding_images(captures,parent.get("diagnostics",{}))
        from .spatial_context import annotate_vectors
        images+=annotate_vectors(captures,parent.get("diagnostics",{}).get("resolved_features",{}),parent.get("base_transform",{}))
    return state,images

def history_entry(parent):
    chosen=next((candidate for candidate in parent["candidates"]
        if candidate["id"]==(parent.get("selected_candidate_id") or parent.get("suggested_candidate_id"))),None)
    return {"id":parent["id"],"round":parent.get("agent_round",1),
        "summary":parent.get("summary",""),"api_calls":parent.get("api_calls",{}),
        "geometry_score":chosen.get("geometry_score") if chosen else None,
        "visual_score":chosen.get("visual_score") if chosen else None,
        "failures":chosen.get("failures",[]) if chosen else []}


def best_result(engine,ident):
    """Keep an accepted earlier round when a later attempt regresses."""
    last=engine.store.read("compositions",ident)
    records=[]
    for item in last.get("agent_history",[]):
        record=engine.store.read("compositions",item["id"])
        if (record["base_id"],record["product_id"],record["instruction"],record["base_transform"]) != (
            last["base_id"],last["product_id"],last["instruction"],last["base_transform"]):
            raise ValueError("Agent history does not match the original request.")
        records.append(record)
    records.append(last)
    accepted=[]
    for record in records:
        for candidate in record["candidates"]:
            review=candidate.get("visual_review") or {}
            if candidate.get("eligible") and review.get("acceptable"):
                accepted.append((candidate.get("combined_score",0),record,candidate))
    if accepted:
        _,selected,candidate=max(accepted,key=lambda item:item[0])
        result=dict(selected)
        result["selected_candidate_id"]=candidate["id"]
        result["suggested_candidate_id"]=candidate["id"]
    else:result=dict(last)
    result["agent_run_id"]=ident
    result["agent_attempts"]=[history_entry(record) for record in records]
    result["agent_selected_round"]=result.get("agent_round",1)
    result["agent_total_rounds"]=last.get("agent_round",1)
    result["agent_selection_reason"]="best accepted measured candidate" if accepted else "no accepted measured candidate"
    return result
