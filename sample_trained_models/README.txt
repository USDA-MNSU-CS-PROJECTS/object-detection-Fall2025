Expected weights for the Gradio app (two models):
  casparian_epidermis.pt   - Casparian strip + Epidermis segmentation
  vascular_bundles.pt      - Vascular bundles

Filenames and class names are configured in src/app/config/inference_constants.py.
For local testing you may copy the same .pt file to both names.

Note: this is the current dual-model runtime layout for src/app/main.py.
Legacy single-model references such as best.pt belong to older packaging/training flows.
