import numpy as np
import tensorflow as tf
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import TensorFlowV2Classifier


def build_art_classifier(model, input_dim):
    """Оборачивает Keras-модель в TensorFlowV2Classifier для использования в ART."""
    return TensorFlowV2Classifier(
        model=model,
        nb_classes=2,
        input_shape=(input_dim,),
        loss_object=tf.keras.losses.SparseCategoricalCrossentropy(),
        clip_values=(0.0, 1.0),
    )


def predict_labels(model, X):
    """Возвращает метки классов (argmax по выходу softmax) для массива X."""
    return np.argmax(model.predict(X, verbose=0), axis=1)


def project_onehot_back(X_adv, onehot_groups):
    """Схлопывает каждый one-hot блок обратно в валидный вектор: argmax → 1.0, остальные → 0.0."""
    X_proj = X_adv.copy()
    for s, e in onehot_groups:
        block = X_proj[:, s:e]
        idx = block.argmax(axis=1)
        new_block = np.zeros_like(block)
        new_block[np.arange(len(block)), idx] = 1.0
        X_proj[:, s:e] = new_block
    return X_proj


def round_binary_features(X_adv, binary_indices):
    """Округляет указанные бинарные признаки до {0, 1} после атаки."""
    if binary_indices is None or len(binary_indices) == 0:
        return X_adv
    X_out = X_adv.copy()
    X_out[:, binary_indices] = np.round(X_out[:, binary_indices])
    return X_out


def detect_binary_features(X_train, onehot_groups):
    """Возвращает индексы бинарных признаков ({0,1} на train), исключая one-hot блоки."""
    binary_mask = np.all((X_train == 0.0) | (X_train == 1.0), axis=0)
    for s, e in onehot_groups:
        binary_mask[s:e] = False
    return np.where(binary_mask)[0]


def run_fgsm(
    model,
    X_test,
    eps,
    onehot_groups=None,
    binary_indices=None,
    project=True,
    round_binary=True,
):
    """Запускает FGSM-атаку ART и при необходимости восстанавливает one-hot и бинарные признаки."""
    classifier = build_art_classifier(model, X_test.shape[1])
    attack = FastGradientMethod(estimator=classifier, eps=eps, norm=np.inf)
    X_adv = attack.generate(x=X_test)

    if project and onehot_groups:
        X_adv = project_onehot_back(X_adv, onehot_groups)

    if round_binary and binary_indices is not None:
        X_adv = round_binary_features(X_adv, binary_indices)

    return X_adv
