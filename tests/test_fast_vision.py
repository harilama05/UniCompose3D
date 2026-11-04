"""Offline contracts for OpenAI localization and real SAM feature reuse."""
import base64
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

from PIL import Image
from unicompose3d.clients import Gemini, ProviderError, safe_error
from unicompose3d.openai_client import OpenAIClient, strict_schema, make_llm
from unicompose3d.schemas import PlacementPlan, TargetLocation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location("resident_fixture", ROOT / "tools/resident_worker.py")
resident = importlib.util.module_from_spec(spec); spec.loader.exec_module(resident)


class VisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.image = Path(self.temp.name) / "photo.png"; Image.new("RGB", (1800, 1200), "blue").save(self.image)
        self.location = {"found": True, "confidence": .9, "box": [.1, .2, .8, .9], "explanation": "test region"}
        self.http = Mock()
        self.http.post.return_value = SimpleNamespace(status_code=200, json=lambda: {"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(self.location)}]}]})

    def test_openai_uses_bounded_jpeg_short_profile_and_original_is_untouched(self):
        before = hashlib.sha256(self.image.read_bytes()).hexdigest()
        client = OpenAIClient(key="fixture-secret", transport=self.http)
        result = client.locate(self.image, "hand")
        self.assertEqual(result.box, self.location["box"])
        body = self.http.post.call_args.kwargs["json"]
        self.assertEqual(body["model"], "gpt-6-luna"); self.assertEqual(body["reasoning"]["effort"], "none")
        self.assertEqual(body["max_output_tokens"], 512); self.assertFalse(body["store"])
        self.assertEqual(self.http.post.call_args.kwargs["timeout"], (5, 25))
        content = body["input"][0]["content"][-1]["image_url"]
        with Image.open(io.BytesIO(base64.b64decode(content.split(",")[1]))) as small:
            self.assertEqual(small.format, "JPEG"); self.assertEqual(small.size, (1024, 683))
        self.assertEqual(hashlib.sha256(self.image.read_bytes()).hexdigest(), before)
        self.assertEqual(result.box, [.1, .2, .8, .9])

    def test_recursive_strict_schema_forbids_extra_fields_and_requires_all_properties(self):
        schema = strict_schema(PlacementPlan)
        def check(node):
            if isinstance(node, dict):
                if node.get("type") == "object":
                    self.assertFalse(node["additionalProperties"])
                    self.assertEqual(set(node["required"]), set(node["properties"]))
                self.assertNotIn("default", node)
                for value in node.values(): check(value)
            elif isinstance(node, list):
                for value in node: check(value)
        check(schema)

    def test_http_error_refusal_incomplete_and_invalid_geometry_are_not_silently_applied(self):
        client = OpenAIClient(key="fixture-secret", transport=self.http)
        for code, payload in [(401, {"error": {"message": "bad fixture-secret"}}),
             (200, {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}),
             (200, {"output": [{"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}]}),
             (200, {"output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps({**self.location, "box": [0, 0, 2, 1]})}]}]})]:
            self.http.post.return_value = SimpleNamespace(status_code=code, json=lambda data=payload: data)
            with self.assertRaises(ProviderError) as error: client.locate(self.image, "object")
            self.assertNotIn("fixture-secret", str(error.exception))
        self.assertEqual(self.http.post.call_count, 4)

    def test_gemini_optional_uses_its_own_short_profile_and_model_factory_is_explicit(self):
        self.http.post.return_value = SimpleNamespace(status_code=200, json=lambda: {"candidates": [{"content": {"parts": [{"text": json.dumps(self.location)}]}}]})
        client = Gemini(key="fixture", transport=self.http); client.locate(self.image, "object")
        body = self.http.post.call_args.kwargs["json"]
        self.assertEqual(body["generationConfig"]["maxOutputTokens"], 512)
        self.assertEqual(body["generationConfig"]["thinkingConfig"]["thinkingLevel"], "low")
        self.assertEqual(body["contents"][0]["parts"][-1]["inline_data"]["mime_type"], "image/jpeg")
        with patch.dict(os.environ, {"TRYON_LLM_PROVIDER": "openai", "OPENAI_API_KEY": ""}):
            self.assertIsInstance(make_llm(), OpenAIClient)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "private-openai", "HF_TOKEN": "private-hf"}):
            message = safe_error("private-openai private-hf")
            self.assertNotIn("private-openai", message); self.assertNotIn("private-hf", message)

    def test_real_sam_encoder_is_reused_for_new_points_and_invalidated_when_image_changes(self):
        os.environ.setdefault("USE_TF", "0"); os.environ.setdefault("USE_FLAX", "0")
        import torch
        from transformers import SamModel, SamConfig, SamProcessor, SamImageProcessor
        before = torch.get_num_threads(); torch.set_num_threads(1); self.addCleanup(torch.set_num_threads, before)
        config = SamConfig(vision_config={"hidden_size": 32, "output_channels": 32, "num_pos_feats": 16, "num_hidden_layers": 1, "num_attention_heads": 4, "mlp_dim": 64, "image_size": 32, "patch_size": 16, "global_attn_indexes": [0]},
             prompt_encoder_config={"hidden_size": 32, "image_size": 32, "patch_size": 16},
             mask_decoder_config={"hidden_size": 32, "num_attention_heads": 4, "mlp_dim": 64, "num_hidden_layers": 1, "iou_head_hidden_dim": 32})
        session = resident.SamSession(Path(self.temp.name), device="cpu")
        session.model = SamModel(config).eval()
        # An untrained decoder cannot be expected to honor a positive semantic point.
        # Make its masks deterministically empty; exercise real embedding reuse with a negative point.
        with torch.no_grad():
            for mlp in session.model.mask_decoder.output_hypernetworks_mlps:
                for parameter in mlp.parameters(): parameter.zero_()
        session.processor = SamProcessor(SamImageProcessor(size={"longest_edge": 32}, pad_size={"height": 32, "width": 32}))
        encoder = Mock(wraps=session.model.get_image_embeddings); session.model.get_image_embeddings = encoder
        path = Path(self.temp.name) / "small.png"; Image.new("RGB", (64, 48), "red").save(path)
        request = {"image": str(path), "output": str(path.with_name("mask.png")), "box": [.1, .1, .9, .9], "points": [], "labels": []}
        with patch.object(resident, "emit"):
            first = session.run(request)
            second = session.run({**request, "points": [[.4, .4]], "labels": [0]})
            self.assertEqual(encoder.call_count, 1)
            self.assertTrue(second["timings"]["embedding_cached"])
            Image.new("RGB", (64, 48), "green").save(path); session.run(request)
            self.assertEqual(encoder.call_count, 2)
        with Image.open(request["output"]) as mask: self.assertEqual(mask.size, (64, 48))


if __name__ == "__main__": unittest.main()

