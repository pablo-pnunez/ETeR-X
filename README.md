# **ETeR-X:** An efficient text-based recommender system with explanatory capabilities for cold-start scenarios

<br>

<p align="center">
    <img src="output/mockup.png" alt="drawing" style="width:700px;"/>
</p>

ETeR-X is a *query-based* text recommender: it learns, from the textual reviews
written by previous customers, an intrinsically explainable association between
relevant terms (nouns and adjectives, selected through part-of-speech filtering)
and items, using a single shallow classifier. Because it never builds a user
profile, it is by design robust to the *user cold-start* problem. The full
methodology, datasets and results are described in the accompanying paper.

This README covers how to set up the environment and reproduce every result
reported in the paper, from raw data to the final tables and figures.

## Repository structure

```
src/                        Core library: dataset loaders, models, metrics
  datasets/text_datasets/    One loader per data source (Amazon, POIs, Restaurants)
  models/text_models/        RSTModel (base class) and its subclasses:
                              BOW2ITM (ETeR-X itself), USEM2ITM, BERT2ITM, MOSTPOP2ITM
Main.py                     Grid-search training entry point (one model/dataset/subset at a time)
FinalTrain.py                Trains the final model with already-chosen hyperparameters
                              and measures training/test emissions with codecarbon
Baselines.py                 Collaborative-filtering baselines via Cornac
                              (MostPop, BPR, EASE^R, BiVAE, plus MF/OnlineIBPR, unused in the paper)
knn_bow_baseline.py           Non-neural kNN-BoW retrieval baseline
embedding_baseline.py         Zero-shot embedding baselines (EmbeddingGemma, Qwen3-Embedding)
mostpop_coldstart.py          MostPop re-evaluated under the user-cold-start protocol
run_one_coldstart_eval.py     Re-evaluates one trained checkpoint (ETeR-X/USEM/BERT) under
                              the user-cold-start protocol, without retraining
run_one_user_agg.py           Recomputes per-user aggregated metrics for one trained checkpoint
run_one_retrain_amazon_fix.py,
run_one_retrain_amazon_correct_cache.py
                              One-off retraining scripts for USEM/BERT on Amazon, kept for
                              provenance; see their docstrings for the specific issue each fixes
vocab_threshold_ablation.py   Sensitivity ablation for the BoW vocabulary-size threshold
min_reviews_ablation.py       Sensitivity ablation for the minimum-reviews-per-item threshold
PlotResults.ipynb             Builds the final result tables/figures directly from models/
Example.ipynb                 Minimal usage example: load a trained model, get a recommendation
Limitations.ipynb             Generates the query examples in the limitations table (tab:limitations):
                              missing adjectives/nouns and negation, one model per domain
CollectResults.py             Legacy Excel-report helper, superseded by PlotResults.ipynb
results/                      Raw, per-script output CSVs from the scripts above (Section 3)
examples/                     Assets for Example.ipynb (a pre-trained checkpoint, zipped)
output/                       Final figures (PDF) and aggregated tables (xlsx) from PlotResults.ipynb
scripts/                      Bash drivers that loop Main.py/FinalTrain.py/Baselines.py over every
                              dataset/subset (and, for Main.py, model/hyperparameter combination),
                              launching a bounded number of them in parallel across GPUs
notebooks/                    RemoveFailedRuns.ipynb: deletes incomplete checkpoint folders left
                              behind by a training run that was interrupted or crashed
models/                       Trained checkpoints, one folder per (model class, dataset, subset, config hash)
data/                         Cached, preprocessed datasets (generated on first run, gitignored)
```

## 1. Environment setup

```bash
conda create --name ETeR-X python=3.9.18
conda activate ETeR-X
pip install -r requirements.txt
```

This covers everything needed to train and evaluate ETeR-X, USEM, BERT-Tiny,
MostPop, BPR and EASE^R, and to run the ablations.

Two baselines have additional, optional requirements:

- **BiVAE and OnlineIBPR** (Cornac baselines, included in `Baselines.py` but not
  reported in the paper's main tables) need PyTorch. It is not pinned in
  `requirements.txt` because the right build depends on your CUDA version;
  install it from the instructions at <https://pytorch.org/get-started/locally/>
  before running `Baselines.py` with these two models enabled. Every model
  actually reported in the paper (MostPop, BPR, EASE^R) runs without it.
- **EmbeddingGemma and Qwen3-Embedding** (`embedding_baseline.py`) are served
  through [Ollama](https://ollama.com) rather than loaded in-process. Install
  Ollama, start it, and pull both models before running the script:
  ```bash
  ollama pull embeddinggemma
  ollama pull qwen3-embedding:0.6b
  ```
  `embedding_baseline.py` talks to Ollama over HTTP (`OLLAMA_HOST` at the top
  of the file); adjust the host/port and `OLLAMA_GPU_ID` if your Ollama
  instance runs on a different port or GPU than the one used for training.

## 2. Data

Datasets are not distributed with this repository. Each loader in
`src/datasets/text_datasets/` expects a specific, pre-existing layout under
`data_path` (a dataset-config key, see below):

- **Amazon** (`AmazonDataset`): the standard McAuley Amazon review dump, split
  by category and reorganized as `{data_path}/{subset}/data.json.gz` (reviews,
  one JSON object per line: `reviewerID`, `asin`, `overall`, `reviewText`,
  `summary`, `unixReviewTime`, ...) and `{data_path}/{subset}/metadata.json.gz`
  (`asin`, `title`, ...). `subset` is `fashion` or `digital_music` in this
  paper. The raw category dumps are publicly available from
  <https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews>; download
  the relevant categories and place them under `{subset}/` with these two
  file names.
- **Restaurants** (`RestaurantDataset`) and **POIs** (`POIDataset`): reviews
  and item metadata scraped from TripAdvisor for a set of cities, stored as
  pickled DataFrames (`{data_path}/{subset}/reviews.pkl` +
  `restaurants.pkl` for restaurants; `{data_path}/DATA_byList/{subset}/df.pickle`
  for POIs). This is an in-house crawl, not a public dataset; adapting either
  loader's `load_subset()` method to a differently-sourced dataset with the
  same columns (`reviewId`, `userId`, `itemId`/`restaurantId`, `rating`,
  `date`, `text`, `title`) is the fastest path to using your own data.

Once the raw files are in place, point `data_path` (in `Main.py`,
`FinalTrain.py`, or any of the ablation/baseline scripts) at their parent
directory. The first run for a given `(dataset, subset)` preprocesses and
caches the result under `data/<DatasetClass>/<config hash>/`; subsequent runs
with the same configuration reuse the cache instead of reprocessing.

## 3. Reproducing the pipeline

Every trained model is saved under
`models/<ModelClass>/<dataset>/<subset>/<config hash>/`, where the hash is an
MD5 of the dataset and model configuration (see `ModelClass.__get_md5__`) —
changing any hyperparameter or preprocessing option produces a new, separate
checkpoint rather than overwriting an existing one. `models/best_models.csv`
records, for every `(dataset, subset, model)` combination reported in the
paper, which of these hashes is the reference checkpoint; every script below
that needs "the" trained model for a combination reads this file rather than
guessing from the directory listing.

### 3.1 Train ETeR-X, USEM and BERT-Tiny

```bash
python Main.py -dst restaurants -sst gijon -mn BOW2ITM -stg 0   # hyperparameter search (dev split)
python Main.py -dst restaurants -sst gijon -mn BOW2ITM -stg 3   # final training + evaluation
```
`-mn` selects the model (`BOW2ITM` = ETeR-X, `USEM2ITM`, `BERT2ITM`,
`MOSTPOP2ITM`), `-dst`/`-sst` the dataset/subset. See `Main.py`'s
`parse_cmd_args` (in `src/Common.py`) for the full list of flags (learning
rate, batch size, vocabulary threshold, etc.). Repeat for each of these four
models on every dataset/subset listed in Table 1 of the paper, or use
`scripts/LaunchTests.sh` to launch the grid search for every combination at
once (bounded parallelism across GPUs; edit the `DATASETS`/`MODELS` arrays at
the top to select which ones to run).

`FinalTrain.py` re-trains the already-selected best configuration for a
combination while also measuring training/inference carbon emissions with
`codecarbon`:
```bash
python FinalTrain.py -dataset restaurants -subset gijon -model BOW2ITM
```
`scripts/FinalTrain.sh` does the same for every combination, one model at a
time (edit its `MODELS` array).

### 3.2 Collaborative-filtering baselines (MostPop, BPR, EASE^R, BiVAE)

```bash
python Baselines.py -dst restaurants -sst gijon
```
Trains and evaluates all Cornac-based baselines for one dataset/subset at
once via a single `cornac.experiment.Experiment`, and writes
`models/Baselines/<dataset>/<subset>/{results.csv,user_results.csv}`. Requires
ETeR-X/USEM/BERT to already be trained for that combination (it loads their
checkpoints via `models/best_models.csv` to build a matched evaluation
split). `BiVAE` and `OnlineIBPR` are only added to the list of models to run
if PyTorch is importable (see Section 1); neither appears in the paper's
result tables, so their absence does not affect reproducing any published
number. `scripts/LaunchBaselines.sh` runs this for every dataset/subset.

### 3.3 kNN-BoW, EmbeddingGemma, Qwen3-Embedding, MostPop under both protocols

```bash
python knn_bow_baseline.py                  # known users
python knn_bow_baseline.py --coldstart       # user cold-start

python embedding_baseline.py all             # both embedding models, known users
python embedding_baseline.py all --coldstart

python mostpop_coldstart.py                  # both protocols in one run
```
Each of these iterates over all twelve datasets on its own and writes its
own output CSV under `results/` (`knn_bow_results.csv`, `embedding_results.csv`,
`mostpop_coldstart_results.csv`, plus `*_coldstart.csv` variants); none of
them retrain ETeR-X, they only reuse its already-selected vocabulary and, for
`mostpop_coldstart.py`, its dataset split.

### 3.4 Per-user aggregated metrics and user-cold-start re-evaluation

Table 6 of the paper (`tab:results_tk1`) aggregates every model's predictions
per user rather than per review, to match Cornac's evaluation semantics for
the CF baselines. For ETeR-X/USEM/BERT this requires an extra pass over an
already-trained checkpoint:
```bash
python run_one_user_agg.py --model BOW2ITM --dataset restaurants --subset gijon --md5 <hash from best_models.csv>
```
and, for the user-cold-start comparison (Section 4.3.3 / Appendix B):
```bash
python run_one_coldstart_eval.py --model BOW2ITM --dataset restaurants --subset gijon --md5 <hash>
```
Both scripts run against a single checkpoint per invocation, in a separate
process, to avoid reusing a TensorFlow/GPU session across models. To run
every combination, loop over `models/best_models.csv`, e.g.:
```bash
python - <<'PY'
import pandas as pd, subprocess
best = pd.read_csv("models/best_models.csv")
for _, row in best[best.model.isin(["BOW2ITM", "USEM2ITM", "BERT2ITM"])].iterrows():
    subprocess.run(["python", "run_one_user_agg.py",
                     "--model", row.model, "--dataset", row.dataset,
                     "--subset", row.subset, "--md5", row.model_md5])
PY
```

### 3.5 Sensitivity ablations

```bash
python vocab_threshold_ablation.py restaurants gijon 5 dev     # find epochs for this threshold
python vocab_threshold_ablation.py restaurants gijon 5 final   # train with that many epochs
python vocab_threshold_ablation.py restaurants gijon 5 eval    # evaluate, append to results/vocab_ablation_results.csv

python min_reviews_ablation.py restaurants gijon 5 dev
python min_reviews_ablation.py restaurants gijon 5 final
python min_reviews_ablation.py restaurants gijon 5 eval        # appends to results/min_reviews_ablation_results.csv
```
Both scripts sweep one hyperparameter (`bow_pct_words` and `min_reviews_rst`
respectively) while keeping every other choice fixed at its grid-searched
value, on the three representative datasets used in the paper (Restaurants
Gijón, POIs London, Amazon Fashion). Each combination of dataset/subset/value
runs as its own process, matching the `dev`/`final`/`eval` phases above.

### 3.6 Figures and tables

`PlotResults.ipynb` reads the raw checkpoints under `models/` directly (not
the CSVs in `results/`, which are read by a human when writing the paper's
text, not by any script) and regenerates the paper's result tables and the
`output/*.pdf` figures/`.xlsx` aggregates. Run it top to bottom after training
or re-evaluating anything under `models/`; it caches intermediate aggregates
under `models/` (as `.xlsx` files) and only recomputes what changed.

## 4. Known gaps in this reproduction package

Two files under `results/` do not have a corresponding script checked in:
`wilcoxon_results.csv` (the paired significance tests in `tab:wilcoxon` and
Section 4.3.2) and `propn_ablation_results.csv` / `propn_ablation_coldstart.csv`
(the proper-noun leakage ablation discussed in the paper). Both were produced
by one-off analysis scripts that were not preserved. Their methodology is
fully described in the paper (Sections "Results" and "Threats to validity"),
and the inputs needed to reproduce them (per-user aggregated predictions from
Section 3.4, and a `pos_tagging_fix` dataset-config toggle already
implemented in `src/datasets/text_datasets/TextDataset.py`) are already
produced by the scripts above.

## 5. Getting recommendations and explanations

`Example.ipynb` shows how to load a trained model and obtain a ranked,
explained recommendation for a free-text query, without training anything
yourself: it unzips `examples/sample_model.zip`, a pre-trained ETeR-X
checkpoint for Restaurants/New York City, straight into `models/` (matching
the entry for that combination in `models/best_models.csv`). If you retrain
that combination yourself and want to refresh this bundle, re-zip the
resulting checkpoint folder together with its `dev/log.csv` (needed to
determine the number of training epochs even when only loading existing
weights) — the bulk of `dev/`, in particular its own copy of the weights, is
not needed and can be left out.

## Citing

The paper describing this system is currently under review; please contact
the authors if you would like to cite this work ahead of publication.
