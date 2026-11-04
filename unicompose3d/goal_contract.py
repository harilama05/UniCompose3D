"""Preserve geometric interaction requirements across agent revisions."""
def derive(plan):
    program=plan.program
    if program is None:return {"version":1,"fits":[],"contact":False,"containment":False}
    features={feature.id:feature for feature in program.features};fits=[]
    for constraint in program.constraints:
        a,b=features[constraint.a],features[constraint.b]
        if "section" in {a.kind,b.kind} and bool({a.kind,b.kind}&{"aperture","cavity"}):
            pair={"section_slot":a.slot if a.kind=="section" else b.slot,
                  "opening_slot":a.slot if a.kind in {"aperture","cavity"} else b.slot}
            if pair not in fits:fits.append(pair)
    return {"version":3,"intent_status":"proposed interpretation; original user instruction is authoritative","intent":plan.intent.model_dump() if plan.intent else None,"fits":fits,"fit_topology":program.fit_topology,
        "contact":any(c.kind=="contact" for c in program.constraints),
        "containment":any(c.kind=="containment" for c in program.constraints)}

def inherited(engine,parent):
    if not parent:return None
    if parent.get("goal_contract"):return parent["goal_contract"]
    records=[]
    for item in parent.get("agent_history",[]):
        try:records.append(engine.store.read("compositions",item["id"]))
        except FileNotFoundError:continue
    records.append(parent)
    from .schemas import PlacementPlan
    contracts=[derive(PlacementPlan.model_validate(record["plan"])) for record in records]
    return {"version":1,"fits":[pair for contract in contracts for pair in contract["fits"]],
        "contact":any(contract["contact"] for contract in contracts),
        "containment":any(contract["containment"] for contract in contracts)}

class GoalContractError(ValueError):pass

def enforce(plan,contract):
    if not contract:return
    proposed=derive(plan)
    required_topology=contract.get("fit_topology","auto")
    if required_topology in {"through","cover"}:
        kinds={f.kind for f in plan.program.features} if plan.program else set()
        required_kind="aperture" if required_topology=="through" else "cavity"
        if proposed.get("fit_topology")!=required_topology or required_kind not in kinds:
            raise GoalContractError("The original interaction topology must be preserved.")
    for required in contract.get("fits",[]):
        if required not in proposed["fits"]:
            raise GoalContractError("The original goal requires matching a section and opening. Do not replace it with point-only alignment or surface overlap.")
    for kind in ("contact","containment"):
        if contract.get(kind) and not proposed[kind]:
            raise GoalContractError("The revision dropped the required "+kind+" from the original goal.")
