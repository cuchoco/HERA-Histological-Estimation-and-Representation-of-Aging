# HERA: Histological Estimation and Representation of Aging

Code for the paper *Histological clocks read shared vascular and organ-specific parenchymal
signatures of aging*.

HERA estimates age from H&E whole-slide images. It combines a **pan-organ** attention-MIL clock
trained on all 40 GTEx tissue types, 24 **per-organ** clocks, and their **organ-wise fusion**

$$\hat{y} = (1-w_o)\,\hat{y}_{\text{pan}} + w_o\,\hat{y}_{\text{per}}$$

with one weight $w_o$ per organ, chosen by cross-fitting.

```
data/gtex_splits.csv          25,702 GTEx slides: slide_id, subject_id, tissue, organ, fold (donor-level 5 folds)
data/gtex_moments.npz         GTEx feature mean / s.d. for embedding alignment of external cohorts
hera/                         model, dataset, alignment, constants
scripts/extract_features.sh   TRIDENT: tissue segmentation, 224 px patches at 20x, UNI2-h embeddings
scripts/prepare_metadata.py   adds donor age (dbGaP) to gtex_splits.csv -> data/metadata.csv
scripts/run_cv.sh             all 5 folds x (pan + 24 per-organ clocks), then fusion
train.py                      trains one clock on one fold
fuse.py                       cross-fitted organ-wise fusion -> out-of-fold predictions, w_o per organ
predict.py                    applies the fused clock to new slides (optional embedding alignment)
```

## Installation

```bash
pip install -r requirements.txt
git clone https://github.com/mahmoodlab/TRIDENT   # feature extraction only; follow its install notes
```

Tested with Python 3.11, PyTorch 2.6 and CUDA 12.4. Training a clock takes a single GPU.

## 1. Data

* **Slides.** GTEx v8 H&E whole-slide images. They are openly available from the NCI Imaging Data
  Commons (collection `gtex`, through `pip install idc-index`). IDC's slide identifier differs from
  our slide ID in the last two digits (`GTEX-1117F-0226` is `GTEX-1117F-0225` in IDC), so match the
  two on the ID without its last two digits. Name each slide file `<slide_id>.svs` (or another
  OpenSlide-readable format) so that the feature file is `<slide_id>.h5`.
* **Donor age.** Exact age is controlled-access (dbGaP phs000424). Once you have access, download
  `GTEx_Subject_Phenotypes.GRU.txt` and run

  ```bash
  python scripts/prepare_metadata.py --phenotypes /path/to/GTEx_Subject_Phenotypes.GRU.txt
  ```

  This writes `data/metadata.csv`. It contains protected data, so it is listed in `.gitignore`.

## 2. Feature extraction

```bash
TRIDENT=/path/to/TRIDENT bash scripts/extract_features.sh /path/to/gtex_slides /path/to/job_dir
# -> /path/to/job_dir/20x_224px_0px_overlap/features_uni_v2/<slide_id>.h5   (N patches x 1536)
```

UNI2-h is gated on Hugging Face (`MahmoodLab/UNI2-h`), so request access and run `huggingface-cli login` first.

## 3. Training and fusion

```bash
FEAT=/path/to/job_dir/20x_224px_0px_overlap/features_uni_v2

python train.py --clock pan    --fold 0 --features_dir $FEAT   # pan-organ clock
python train.py --clock Brain  --fold 0 --features_dir $FEAT   # one per-organ clock
python train.py --clock organs --fold 0 --features_dir $FEAT   # all 24 per-organ clocks

bash scripts/run_cv.sh $FEAT        # everything: 5 folds x 25 clocks, then fuse.py
```

Checkpoints and validation predictions are written to `checkpoints/<clock>/fold<k>/`.
`fuse.py` then writes `results/oof_predictions.csv` (out-of-fold `pan`, `per_organ` and `fused`
prediction for every slide) and `results/fusion_weights.csv` ($w_o$ per organ).

Expected out-of-fold results (25,702 slides):

| clock | MAE (years) | Pearson r | R² |
|---|---|---|---|
| pan-organ | 4.46 | 0.881 | 0.774 |
| fused | 4.16 | 0.898 | 0.806 |

## 4. Trained weights

The trained weights are not publicly available. The clocks were trained on controlled-access GTEx
donor ages, so we treat the weights as Data Derivatives under the dbGaP Data Use Certification and
share them only with approved dbGaP users of GTEx (phs000424). Without that access, you cannot
obtain the weights or the donor ages needed to retrain the clocks. With access, you can either
retrain them (sections 1-3) or request ours.

To request the weights, email the corresponding authors (Hoyeon Jeong, sodi@g.skku.edu; Yoon-La
Choi, ylachoi@skku.edu) with your name, institution, dbGaP project number and intended use. Sharing
must comply with the dbGaP approvals of both parties and may need additional NIH approval.

By receiving the weights, you agree to:

* use them for non-commercial academic research only. They were trained on UNI2-h outputs, so the
  [UNI2-h terms of use](https://huggingface.co/MahmoodLab/UNI2-h) apply to them as derivatives;
* not redistribute them, and handle them as Data Derivatives under your own dbGaP Data Use
  Certification, including destroying them at project close-out.

UNI2-h itself is not included, so you need your own access to it for feature extraction.

The package `hera_checkpoints.tar.gz` (about 6 GB) contains all 125 clocks (5 pan-organ and
24 x 5 per-organ) and the fusion weights. Extract it in the repository root:

```bash
tar -xzf hera_checkpoints.tar.gz
# -> checkpoints/<clock>/fold<k>/best_model.pt  and  results/fusion_weights.csv
```

The layout is the same as the output of section 3, so `predict.py` works with either. Running
`fuse.py` afterwards overwrites `results/fusion_weights.csv`.

## 5. Applying the clock to new slides

This step needs the checkpoints and `results/fusion_weights.csv` from section 3 or 4. Extract
features with the same `scripts/extract_features.sh`, then

```bash
# all slides in a folder are from one organ
python predict.py --features_dir $EXT_FEAT --organ Lung --out lung_pred.csv --align

# or a list with columns slide_id, organ and optionally age
python predict.py --features_dir $EXT_FEAT --slides slides.csv --out pred.csv --align
```

Each slide is scored by the five fold models of the pan-organ clock and of the matching per-organ
clock and fused with that organ's $w_o$. `--align` standardizes the patch embeddings of each organ
group of the input to the GTEx moments (unsupervised, no labels). It is recommended for slides from
other hospitals or scanners. If an `age` column is given, the script reports MAE and r, together
with the MAE after a linear recalibration on 10 labelled slides.

## Citation

> *Manuscript in preparation.*

## License

The code and data files in this repository are released under the
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license for non-commercial use (see
[LICENSE](LICENSE)). The trained weights are not covered by this license; see
[Trained weights](#4-trained-weights). UNI2-h and TRIDENT have their own licenses.
