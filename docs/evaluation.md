# Evaluation

The automated checks separate geometry, application integration, and external model quality.

- **Geometry benchmark:** 17/17 passing procedural cases covering contact, support, attachment, containment, cavities, through openings, scale, direction, local regions, and infeasible configurations. Programs and expected outcomes are explicit; this does not measure live LLM accuracy.
- **Regression suite:** 248 passing tests for active reconstruction, segmentation, composition, caching, import/export, security, and packaging tests. Tests for removed experimental backends are excluded.
- **Browser integration:** real HTTP, Three.js rendering, mask editing, clarification, candidate review, agent refinement, project saving, and export, with offline LLM/GPU fixtures.

The retained Hunyuan inference configuration and composition algorithms match the latest pre-cleanup Kaggle source. Package imports and application wiring were updated; unused reconstruction adapters were removed. A normalized AST comparison confirmed 32 protected geometry, composition, SAM, and Hunyuan modules retain their pre-cleanup logic.

## Reproduce

```bash
python -m unittest discover -s tests
python tools/benchmark_composition.py --output test-results/composition.json
python -m playwright install chromium
python tools/full_pipeline_browser_smoke.py
python tools/diagnostic_browser_smoke.py
python tools/agent_browser_smoke.py
```

Recorded geometry results: [benchmarks/composition.json](../benchmarks/composition.json). For an explicitly enabled paid evaluation, inspect `python tools/benchmark_luna_browser.py --help`.

## Limits

Single-image reconstruction infers hidden geometry. Open meshes, missing cavities, and noisy surfaces limit measurement and collision checks. Bounded pose and scale optimization cannot repair arbitrary topology or simulate cloth. Landmark IDs locate real surface points, but the LLM may select the wrong semantic region. Visual review can miss errors.

Reconstruction includes GPU inference and CPU mesh/UV work; hardware utilization alone does not explain total latency. This release was checked locally with offline providers, not with a new live Luna or Kaggle T4 quality benchmark. Visual composition does not establish physical fit or size.
