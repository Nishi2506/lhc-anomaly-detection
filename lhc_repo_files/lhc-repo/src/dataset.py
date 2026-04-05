import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# jet type indices
GLUON = 0
QUARK = 1
W     = 2
Z     = 3
TOP   = 4

def load_files(file_paths):
    # load jets array from multiple h5 files and concatenate
    jets_list  = []
    labels_list = []
    for path in file_paths:
        with h5py.File(path, "r") as f:
            jets = f["jets"][:]
            labels_list.append(jets[:, -6:])
            jets_list.append(jets)
    jets_raw   = np.concatenate(jets_list,   axis=0)
    labels_raw = np.concatenate(labels_list, axis=0)
    return jets_raw, labels_raw

def get_jet_type(labels_raw):
    return np.argmax(labels_raw, axis=1)

def clean(X, jet_type):
    mask = np.all(np.isfinite(X), axis=1)
    return X[mask], jet_type[mask]

def split_bkg_sig(X, jet_type):
    is_bkg = (jet_type == GLUON) | (jet_type == QUARK)
    is_sig = (jet_type == W)     | (jet_type == Z) | (jet_type == TOP)
    return X[is_bkg], X[is_sig], jet_type[is_bkg], jet_type[is_sig]

def preprocess(X_bkg, X_sig, test_size=0.15, val_size=0.15, seed=42):
    X_train, X_temp = train_test_split(X_bkg, test_size=test_size + val_size, random_state=seed)
    X_val, X_test_bkg = train_test_split(X_temp, test_size=0.5, random_state=seed)

    scaler = StandardScaler()
    X_train    = scaler.fit_transform(X_train)
    X_val      = scaler.transform(X_val)
    X_test_bkg = scaler.transform(X_test_bkg)
    X_test_sig = scaler.transform(X_sig[:len(X_test_bkg)])

    return X_train, X_val, X_test_bkg, X_test_sig, scaler
