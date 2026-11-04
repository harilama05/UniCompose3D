"""Bounded user clarification, separate from geometric tool selection."""
def answers(engine,request):
    if not request.clarification_of:return [],0
    prior=engine.store.read("compositions",request.clarification_of)
    if prior.get("status")!="needs_input":
        raise ValueError("This question is no longer waiting for an answer.")
    if (prior["base_id"],prior["product_id"],prior["base_transform"],prior["instruction"],prior["hints"]) != (
        request.base_id,request.product_id,request.base_transform.model_dump(),request.instruction,request.hints.model_dump()):
        raise ValueError("The objects or instruction changed. Start a new composition.")
    response=(request.clarification_answer or "").strip()
    if not response:raise ValueError("Choose an option or enter an answer.")
    history=list(prior.get("clarification_answers",[]))
    if len(history)>=3:raise ValueError("Three questions have been asked. Add the missing preference to the instruction.")
    history.append({"question":prior["question"]["question"],"answer":response})
    return history,int(prior.get("api_calls",{}).get("plan",0))
