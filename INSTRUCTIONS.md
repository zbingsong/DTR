# How to

First download the model's weights (no authorization needed):
```bash
# assume you are inside DTR/
wget https://github.com/birkhoffkiki/DTR/releases/download/weights/hemit_weight.pth
mkdir weights
mv hemit_weight.pth weights
```

Run the following:

```bash
conda create --name DTR python=3.9
conda activate DTR
pip install -r requirements.txt

python infer_wsi.py \
    --slide <path to .svs> \
    --output-dir <output directory> \
    --checkpoint weights/hemit_weight.pth \
    --levels <comma-separated list of WSI levels to run predictions> \
    --tile-size 512 \
    --device cuda:0 \
    --ome-quant-mode <global/tile/none>
```

`--ome-quant-mode` affects how the raw prediction logits are converted to the `ome.tiff` file:
- `tile`: logits are normalized per tile without considering other tiles;
- `global`: logits are normalized using global max/min logit values;
- `none`: do not perform quantization and save float values into the `ome.tiff` file directly.

# Important Things
- The WSI inference pipeline first split the input WSI level into non-overlapping tiles, and then identify background tiles using a heuristic: tiles where 99.9% of pixels have RGB values greater than 200 in all 3 channels are considered background (this means this tile is almost purely white).
