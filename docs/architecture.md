# Architecture

UniCompose3D keeps reconstruction, intent reasoning, geometric measurement, and visual review separate. The Python package is `unicompose3d`. `TRYON_*` environment variables and saved project identifiers remain compatible with earlier releases.

## Data flow

1. **Prepare inputs.** Upload images, extract video frames, or import a mesh. SAM uses a box or include/exclude points. Optional Luna localization translates an object description into image-space guidance. Connected-background removal and brush corrections refine alpha masks.
2. **Reconstruct.** Hunyuan3D Turbo generates geometry; Paint creates textures. Single-view and up to four explicitly assigned multiview inputs are supported. Video is sampled into images, not reconstructed as motion. Model setup and inference run in an isolated GPU environment; the Studio does not install weights at startup.
3. **Ground intent.** Luna sees the original request, canonical renders, measured dimensions, and identifiable surface landmarks. A structured spatial program proposes regions, features, and constraints. Ambiguity can trigger a question with selectable answers.
4. **Search and solve.** A bounded search explores feature choices, anchor regions, normal polarity, orientation, translation, and scale. Continuous optimization refines poses. Independent geometric checks validate contact, containment, openings, cavities, clearance, and directional constraints. Object B is scaled relative to A; bounded per-axis changes improve visual fit without unrestricted deformation.
5. **Review and refine.** Candidates are rendered from multiple views. Visual review checks the original request and supplies actionable feedback, including for failed geometric candidates. Review cannot override a failed geometry gate. Goal constraints and user answers persist across rounds.
6. **Edit and export.** Three.js supplies orbit controls, transforms, optional local deformation, candidate comparison, asset reuse, GLB export, and project save/restore. Failed proposals do not silently replace the current scene.

Search is limited by program count, starts, wall time, and agent rounds. It is not limited to two normal directions or a fixed list of object combinations.

## Source map

| Component | Location |
| --- | --- |
| HTTP API, jobs, assets, saved projects | `unicompose3d/api.py`, `unicompose3d/storage.py` |
| Agent orchestration and goal preservation | `unicompose3d/agent_loop.py`, `unicompose3d/goal_contract.py` |
| LLM transport and structured planning | `unicompose3d/openai_client.py`, `unicompose3d/clients.py`, `unicompose3d/planning_prompt.py` |
| Hypothesis search and optimization | `unicompose3d/spatial_search.py`, `unicompose3d/constraint_solver.py` |
| Local measurements and interface fitting | `unicompose3d/aperture.py`, `unicompose3d/cavity.py`, `unicompose3d/interface_fit.py` |
| Surface grounding and directions | `unicompose3d/spatial_context.py`, `unicompose3d/orientation.py` |
| Hunyuan and SAM workers | `tools/quality_worker.py`, `tools/sam_mask.py`, `tools/setup_quality.py` |
| Browser editor | `unicompose3d/static/` |
| Regression tests and benchmarks | `tests/`, `tools/composition_cases.py`, `benchmarks/` |

Hunyuan3D is the sole reconstruction backend. Earlier experimental adapters and installers have been removed. Runtime environment variables retain their existing TRYON_ names so established Kaggle environments and saved projects remain usable.

## Configuration and deployment

Use `.env.example` locally or Kaggle Secrets in a notebook. Environment variables take precedence over `.env`. Luna remains the default LLM; startup never calls a paid provider.

A random session token protects API requests and is passed in the launch URL fragment. Keep the complete link private. The launcher binds to localhost by default; `--share` explicitly enables a notebook tunnel. This is a single-user research application, not a multi-tenant hosted service.

Assets, cached results, diagnostic bundles, and project state live under `TRYON_DATA_DIR`. Download a project ZIP before ending an ephemeral Kaggle session. Do not commit this directory, API keys, or model weights.
