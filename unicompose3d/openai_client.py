"""OpenAI Responses adapter with validated, bounded spatial-plan repair."""
from __future__ import annotations
import copy
import json
import os
import requests
from pydantic import ValidationError

from .clients import Gemini, ProviderError, safe_error, PROMPT, EVALUATION_PROMPT
from .vision import LOCATE_PROMPT
from .schemas import PlacementPlan, TargetLocation

PLAN_REPAIRS = 1


def strict_schema(model):
    schema = copy.deepcopy(model.model_json_schema())
    if model is PlacementPlan:
        # Legacy fitting switches are not part of a declarative spatial plan.
        for name in ("fit_mode", "base_axis_end", "base_axis_end_view", "base_axis_end_uv"):
            schema["properties"].pop(name, None)
        schema["properties"]["program"] = {"anyOf":[{"$ref":"#/$defs/SpatialProgram"},{"type":"null"}]}
        constraint = schema["$defs"]["SpatialConstraint"]
        clearance, other = copy.deepcopy(constraint), copy.deepcopy(constraint)
        clearance["properties"]["kind"]["enum"] = ["clearance"]
        clearance["properties"]["target"]["minimum"] = 1.015
        other["properties"]["kind"]["enum"].remove("clearance")
        schema["$defs"]["SpatialConstraint"] = {"anyOf": [clearance, other]}
    def walk(node):
        if isinstance(node, dict):
            node.pop("default", None)
            if node.get("type") == "object" or "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}))
            for value in list(node.values()):
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
    walk(schema)
    return schema


def validation_details(error):
    issues = error.errors(include_input=False, include_context=False, include_url=False)
    return "; ".join((".".join(map(str, issue["loc"])) or "response") + ": " + issue["msg"]
                     for issue in issues[:5])[:1100]


class OpenAIClient(Gemini):
    name = "OpenAI GPT-6 Luna"
    provider = "openai"

    def __init__(self, key=None, model=None, transport=None):
        self.key = (key if key is not None else os.getenv("OPENAI_API_KEY", "")).strip()
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-6-luna")
        self.http = transport or requests.Session()

    def _post_text(self, body, locating):
        try:
            response = self.http.post("https://api.openai.com/v1/responses",
                headers={"Authorization": "Bearer " + self.key}, json=copy.deepcopy(body),
                timeout=(5, 25) if locating else (15, 120))
        except requests.RequestException:
            raise ProviderError("OpenAI did not respond in time. Network failures are not retried automatically.") from None
        try:
            payload = response.json()
        except ValueError:
            raise ProviderError(f"OpenAI HTTP {response.status_code}: response is not JSON.") from None
        if not isinstance(payload, dict):
            raise ProviderError("OpenAI returned an invalid response structure.")
        if response.status_code >= 400:
            detail = str((payload.get("error") or {}).get("message") or "Request rejected.").replace(self.key, "[redacted]")
            hint = ("Check quota, balance, and rate limits." if response.status_code == 429 else
                    "Check the key and model access." if response.status_code in (401, 403, 404) else
                    "Service unavailable." if response.status_code >= 500 else "Check request configuration.")
            raise ProviderError(f"OpenAI HTTP {response.status_code} · model {self.model}: {safe_error(detail)} {hint}")
        if payload.get("status") == "incomplete":
            raise ProviderError("OpenAI output is incomplete. " + safe_error(payload.get("incomplete_details", {})))
        if payload.get("status") not in (None, "completed"):
            detail = str(payload.get("error") or payload.get("status")).replace(self.key, "[redacted]")
            raise ProviderError("OpenAI did not complete the request: " + safe_error(detail))
        texts = []
        for item in payload.get("output", []):
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for block in item.get("content", []):
                if block.get("type") == "refusal":
                    raise ProviderError("OpenAI refused the request. Refusals are not retried.")
                if block.get("type") == "output_text" and isinstance(block.get("text"), str):
                    texts.append(block["text"])
        if not texts or not "".join(texts).strip():
            raise ProviderError("OpenAI completed without JSON output. The current scene is retained.")
        return "".join(texts)

    def _generate(self, parts, schema):
        if not self.key:
            raise ProviderError("OPENAI_API_KEY is missing. Add a notebook secret; manual SAM prompts remain available.")
        locating = schema is TargetLocation
        instructions=None
        if parts and parts[0].get("text") in (PROMPT,EVALUATION_PROMPT,LOCATE_PROMPT):
            instructions=parts[0]["text"];parts=parts[1:]
        content = []
        for part in parts:
            if "text" in part:
                content.append({"type": "input_text", "text": part["text"]})
            elif "inline_data" in part:
                data = part["inline_data"]
                content.append({"type": "input_image", "image_url": "data:" + data["mime_type"] + ";base64," + data["data"], "detail": "auto"})
        body = {"model": self.model, "store": False,
                "input": [{"role": "user", "content": content}],
                "reasoning": {"effort": "none" if locating else os.getenv("OPENAI_COMPOSITION_REASONING_EFFORT",os.getenv("OPENAI_REASONING_EFFORT", "medium"))},
                "max_output_tokens": 512 if locating else 12288,
                "text": {"format": {"type": "json_schema", "name": schema.__name__, "schema": strict_schema(schema), "strict": True}}}
        if instructions:body["instructions"]=instructions
        attempts = 1 + PLAN_REPAIRS if schema is PlacementPlan else 1
        for attempt in range(attempts):
            text = self._post_text(body, locating)
            try:
                result = schema.model_validate_json(text)
                issue = "program: a non-null spatial program is required." if schema is PlacementPlan and result.program is None and result.clarification is None else ""
                if schema is PlacementPlan and not issue and result.clarification is None and result.intent is None:
                    issue="intent: describe target region, moving region, original relationship, orientation and avoid regions."
            except ValidationError as error:
                issue = validation_details(error)
            if not issue:
                object.__setattr__(result, "_provider_calls", attempt + 1)
                return result
            issue = safe_error(issue.replace(self.key, "[redacted]"))
            if attempt + 1 == attempts:
                raise ProviderError(f"OpenAI {schema.__name__} invalid after {attempt + 1} attempts: {issue} The current 3D scene is retained.")
            # A completed plan with invalid semantics gets one explicit repair.
            # Transport errors, refusals, incomplete output, locate/evaluate never retry.
            body["input"].extend([
                {"role": "assistant", "content": [{"type": "output_text", "text": text[:24000]}]},
                {"role": "user", "content": [{"type": "input_text", "text":
                    "Repair the previous spatial plan using the SAME objects, renders and user intent. "
                    "Validation failed: " + issue +
                    " Return the full schema-valid JSON. All constraint a references must point to slot A, "
                    "b references to slot B. Contact needs two surfaces. Clearance needs one section and one aperture or cavity in either A/B order "
                    "with radius ratio >=1.015. Never invent mesh holes or alter the task to hide a failure."}]},
            ])


def make_llm():
    provider = os.getenv("TRYON_LLM_PROVIDER", "openai").strip().lower()
    if provider == "openai":
        return OpenAIClient()
    if provider == "gemini":
        return Gemini()
    raise ValueError("TRYON_LLM_PROVIDER must be openai or gemini.")
