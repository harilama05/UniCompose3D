"""Exercise SAM's real processor/model API with tiny random CPU parameters.

This is an interface check only, not pretrained segmentation or GPU quality.
"""
import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from unicompose3d.quality import gpu_environment, QualityBackend

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('quality_worker', ROOT / 'tools/quality_worker.py')
worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)


class WorkerTests(unittest.TestCase):
    def test_gpu_environment_does_not_forward_service_credentials(self):
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'secret', 'TRIPO_API_KEY': 'secret', 'TRYON_ACCESS_TOKEN': 'secret', 'PYTHONPATH': 'wrong'}):
            env = gpu_environment()
        for name in ('GEMINI_API_KEY', 'TRIPO_API_KEY', 'TRYON_ACCESS_TOKEN', 'PYTHONPATH'):
            self.assertNotIn(name, env)
        self.assertEqual(env['USE_TF'], '0')

    def test_real_sam_processor_and_tiny_model_accept_box_and_point_edits(self):
        if importlib.util.find_spec('transformers') is None:
            self.skipTest('Transformers not installed in QA environment')
        with patch.dict(os.environ, {'USE_TF': '0', 'USE_FLAX': '0'}):
            import torch
            from transformers import SamProcessor, SamImageProcessor, SamConfig, SamModel
        processor = SamProcessor(SamImageProcessor(size={'longest_edge': 32}, pad_size={'height': 32, 'width': 32}))
        config = SamConfig(
            vision_config={'hidden_size': 32, 'output_channels': 32, 'num_pos_feats': 16, 'num_hidden_layers': 1, 'num_attention_heads': 4, 'mlp_dim': 64, 'image_size': 32, 'patch_size': 16, 'global_attn_indexes': [0]},
            prompt_encoder_config={'hidden_size': 32, 'image_size': 32, 'patch_size': 16},
            mask_decoder_config={'hidden_size': 32, 'num_attention_heads': 4, 'mlp_dim': 64, 'num_hidden_layers': 1, 'iou_head_hidden_dim': 32},
        )
        model = SamModel(config).eval()
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        image = Image.new('RGB', (100, 60), 'green')
        for points, labels in (([], []), ([[.2, .2], [.8, .8]], [1, 0])):
            prompts = worker.sam_prompts(image.size, {'box': [.1, .1, .9, .9], 'points': points, 'labels': labels})
            inputs = processor(image, return_tensors='pt', **prompts)
            self.assertEqual(tuple(inputs['input_boxes'].shape), (1, 1, 4))
            with torch.inference_mode(): outputs = model(**inputs)
            masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs['original_sizes'], inputs['reshaped_input_sizes'])[0][0]
            selected = int(outputs.iou_scores[0, 0].argmax().item())
            self.assertEqual(tuple(masks[selected].shape), (60, 100))


if __name__ == '__main__':
    unittest.main()
