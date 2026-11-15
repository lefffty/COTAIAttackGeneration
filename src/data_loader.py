import os

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "class",
    "difficulty",
]

CATEGORICAL = ["protocol_type", "service", "flag"]


def load_raw(data_dir):
    """Загружает KDDTrain+.txt и KDDTest+.txt как DataFrame с фиксированной схемой COLUMNS."""
    train = pd.read_csv(
        os.path.join(data_dir, "KDDTrain+.txt"), header=None, names=COLUMNS
    )
    test = pd.read_csv(
        os.path.join(data_dir, "KDDTest+.txt"), header=None, names=COLUMNS
    )
    return train, test


def _labels(df):
    """Возвращает бинарные метки: 0 = normal, 1 = атака."""
    return (df["class"] != "normal").astype(int).values


def preprocess_data(train, test):
    """
    Готовит признаки для модели.

    Возвращает кортеж (X_train, y_train, X_test, y_test, scaler,
    onehot_groups, feature_names): one-hot для CATEGORICAL, MinMaxScaler
    для числовых (fit на train), onehot_groups — границы блоков для
    проекции после атаки.
    """
    y_train = _labels(train)
    y_test = _labels(test)

    feats_train = train.drop(columns=["class", "difficulty"])
    feats_test = test.drop(columns=["class", "difficulty"])

    full = pd.concat([feats_train, feats_test], axis=0)
    full = pd.get_dummies(full, columns=CATEGORICAL, dtype=float)

    n_tr = len(feats_train)
    X_train = full.iloc[:n_tr].values
    X_test = full.iloc[n_tr:].values

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    onehot_groups = []
    for col in CATEGORICAL:
        cols = [i for i, c in enumerate(full.columns) if c.startswith(col + "_")]
        if cols:
            onehot_groups.append((min(cols), max(cols) + 1))

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        scaler,
        onehot_groups,
        full.columns.tolist(),
    )
