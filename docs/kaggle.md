# Run on Kaggle

## Prepare the upload

From the repository root:

```bash
python -m pip install -r requirements-dev.txt
python tools/build_bundle.py
```

Upload `dist/UniCompose3D.zip` as a private Kaggle dataset. Create a notebook by importing `notebooks/UniCompose3D_Kaggle.ipynb`, then attach the dataset as Input.

## Configure and run

1. Enable **Internet** and a **T4 GPU** in notebook settings.
2. Add `OPENAI_API_KEY` in **Add-ons → Secrets** and enable notebook access. No Hugging Face token is required by the standard Hunyuan setup.
3. Run notebook sections **1 → 2 → 2B → 3 → 4**. Setup downloads weights and compiles extensions; it does not call the paid LLM.
4. Open the complete Studio link, including `#token=...`, in another tab. Keep the notebook session running.

The model environment reuses Kaggle CUDA PyTorch. Model weights need approximately 11.2 GiB; start with at least 14 GiB of free disk and no competing GPU task. Actual memory peaks and latency depend on input, texture generation, and the Kaggle runtime. CPU UV unwrapping and mesh processing can take substantial time even when GPU inference is brief.

## Use the Studio

1. Add **A** and **B** as images, video frames, or existing models.
2. For image inputs, describe **Object to keep** or draw a box. Check the alpha preview; use include/exclude points or **Remove background region** for holes.
3. Confirm the masks, then reconstruct. For multiview Hunyuan, assign the camera view for each of up to four images; the application does not silently discard extra selected images.
4. Inspect the model and choose **Use as A/B**. Reuse A when replacing B.
5. Enter a short request such as “Place the vase on the table.” Click **Compose with agent** and answer any clarification question.
6. Review candidates. Adjust position, rotation, or scale if needed, and download **GLB** or **project ZIP**.

A valid GLB does not prove reconstruction accuracy. Open meshes and missing cavities can prevent reliable automatic placement; the application reports uncertainty instead of claiming a verified fit.

## Updating and troubleshooting

- Save your project ZIP before replacing the source dataset.
- Restart the session after changing code; then run from section 1.
- Setup failure: inspect the first underlying error above the final `CalledProcessError`.
- Disk full: remove redundant caches or restart a clean session; preserve your exported projects first.
- GPU memory failure: stop competing tasks and inspect the job diagnostics. Two T4 devices do not automatically pool their VRAM.
- Provider error: check API model access, quota, and the reported HTTP status. Repeatedly reconstructing objects does not fix an LLM error.
- If the tunnel fails, use the optional reconnect cell. Keep the token-bearing link private.
