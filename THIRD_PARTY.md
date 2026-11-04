# Third-party notices

The root [MIT license](LICENSE) covers original UniCompose3D code. Bundled libraries, downloaded models, and external services retain their own terms.

- **Three.js 0.180.0** — MIT. [Source](https://github.com/mrdoob/three.js); bundled license: `unicompose3d/static/vendor/LICENSE-THREE.txt`.
- **Draco** — Apache-2.0. [Source](https://github.com/google/draco); bundled decoder notices: `unicompose3d/static/vendor/LICENSE-DRACO.txt`.
- **Meshoptimizer** — MIT. [Source](https://github.com/zeux/meshoptimizer); bundled license: `unicompose3d/static/vendor/LICENSE-MESHOPT.txt`.
- **Hunyuan3D-2** — Tencent community/custom license. [Code](https://github.com/Tencent-Hunyuan/Hunyuan3D-2), revision `f8db63096c8282cb27354314d896feba5ba6ff8a`; [single-view weights](https://huggingface.co/tencent/Hunyuan3D-2), revision `9cd649ba6913f7a852e3286bad86bfa9a2d83dcf`; [multiview weights](https://huggingface.co/tencent/Hunyuan3D-2mv), revision `3a761b539b29fe4ff64714813aa9560fd66f5de0`. Weights download during setup and are not included in the repository.
- **SAM ViT Base** — Apache-2.0. [Model](https://huggingface.co/facebook/sam-vit-base), revision `70c1a07f894ebb5b307fd9eaaee97b9dfc16068f`. Weights download during setup.
- **Python dependencies** retain their upstream licenses and are installed from their registries.
- **OpenAI** supplies the primary Luna LLM API. The optional Gemini adapter uses Google services. Both require user credentials and remain subject to provider terms; startup makes no paid calls.

Demo renders illustrate application behavior. Source object designs and likenesses are not relicensed by the software license.
