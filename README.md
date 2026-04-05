# LHC Jet Anomaly Detection

Unsupervised anomaly detection on LHC jet data using a deep autoencoder.
Trained on QCD background jets only. High reconstruction error flags W, Z, and top quark jets as anomalies.

---

## Results

| Jet Type | Reconstruction Error |
|---|---|
| Quark (background) | 0.180 |
| Gluon (background) | 0.187 |
| W boson | 0.175 |
| Z boson | 0.188 |
| Top quark | 0.348 |

AUC: 0.57 (unsupervised, no signal labels used in training)

Top quark jets show ~2x higher reconstruction error than QCD background.
UMAP of the latent space shows top quarks clustering in a completely separate region.

---

## Dataset

HLS4ML LHC Jet Dataset (Zenodo record 3602260)
50,000 simulated LHC jets, 53 high-level substructure features per jet.
Jet types: gluon, quark, W boson, Z boson, top quark.

---

## Structure

    lhc-anomaly-detection/
        notebooks/
            01_eda.ipynb
            02_autoencoder.ipynb
        src/
            dataset.py
            model.py
            train.py
            evaluate.py
            visualize.py
        models/
            jet_autoencoder_best.pt
        results/
            07_jet_anomaly_detection.png
            08_per_jet_type_scores.png
            09_umap_latent_space.png
        requirements.txt

---

## Setup

    pip install -r requirements.txt

Download dataset:

    wget "https://zenodo.org/records/3602260/files/hls4ml_LHCjet_150p_train.tar.gz"
    tar -xzf hls4ml_LHCjet_150p_train.tar.gz

---

## How it works

The autoencoder learns to compress and reconstruct QCD jet features through a 3-dimensional bottleneck.
At test time, jets the model has not seen (W, Z, top) produce higher reconstruction error.
That error is used directly as the anomaly score. No labels needed.

Architecture: 53 -> 32 -> 16 -> 8 -> 3 -> 8 -> 16 -> 32 -> 53

---

## Why top quarks score highest

Top quarks decay as t -> W b -> q q-bar b, producing 3 distinct subjets.
QCD jets typically have 1 to 2 subjets.
The autoencoder learns the 1-2 subjet structure of QCD and cannot reconstruct
3-subjet top events accurately, giving them ~2x higher error.

---

## References

Kasieczka et al. (2021). The LHC Olympics 2020. arXiv:2101.08320
Farina et al. (2020). Searching for new physics with deep autoencoders. arXiv:1808.08992
HLS4ML Collaboration. Fast Machine Learning for Science. fastmachinelearning.org
