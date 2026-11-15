import hashlib
import json
import logging
import os
import sys
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
warnings.filterwarnings("ignore", message="PyTorch not found.*", category=UserWarning)
warnings.filterwarnings(
    "ignore", message="pkg_resources is deprecated.*", category=UserWarning
)

import random
import traceback

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger

import config
from src.attack import detect_binary_features, predict_labels, run_fgsm
from src.data_loader import load_raw, preprocess_data
from src.env_info import collect_env_info
from src.json_writer import build_run_block, write_results
from src.metrics import compute_metrics
from src.model import build_model, train_or_load
from src.run_logger import Tee, setup_logger
from src.validity import check_validity


def set_seeds(seed):
    """
    Фиксирует random seed для random, numpy и tensorflow (для воспроизводимости прогона).
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def model_path_for_seed(seed):
    """
    Возвращает путь к .keras-файлу модели для данного seed (или общий MODEL_PATH, если RETRAIN_PER_RUN=False).
    """
    if not config.RETRAIN_PER_RUN:
        return config.MODEL_PATH
    fp = hashlib.sha256(
        json.dumps(
            {
                "epochs": config.EPOCHS,
                "batch_size": config.BATCH_SIZE,
                "arch": config.MODEL_INFO["architecture"],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:8]
    return os.path.join(config.MODELS_DIR, f"nslkdd_mlp_seed{seed}_{fp}.keras")


def get_or_train_model(
    X_train, y_train, X_test, y_test, model_path, run_id, logger: logging.Logger
):
    """
    Загружает модель с диска по model_path или обучает новую на train с валидацией на test
    и сохраняет в model_path. В процессе обучения пишет CSV (CSVLogger) и TXT (Tee) в logs/.

    Возвращает (model, log_paths), где log_paths — dict с путями к training_csv / training_txt
    (пустой, если соответствующее логирование отключено в config).
    """
    if os.path.exists(model_path):
        logger.info(f"Load model from {model_path}")
        return tf.keras.models.load_model(model_path), {}

    logger.info(f"Train new model -> {model_path}")

    m = build_model(X_train.shape[1])

    callbacks = []
    log_paths = {}

    if config.SAVE_TRAINING_CSV:
        csv_path = os.path.join(config.LOGS_DIR, f"{run_id}_training.csv")
        callbacks.append(CSVLogger(csv_path, separator=",", append=False))
        log_paths["training_csv"] = csv_path

    txt_path = None
    if config.SAVE_TRAINING_TXT:
        txt_path = os.path.join(config.LOGS_DIR, f"{run_id}_training.txt")
        log_paths["training_txt"] = txt_path

    old_stdout = sys.stdout
    tee = None
    if txt_path:
        tee = Tee(sys.stdout, open(txt_path, "w", encoding="utf-8"))
        sys.stdout = tee
    try:
        m.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            verbose=2,
            callbacks=callbacks,
        )
    finally:
        if tee is not None:
            sys.stdout = old_stdout
            tee.close()

    m.save(model_path)
    logger.info("Model saved: %s", model_path)
    return m, log_paths


def runner(random_seeds: list[int]):
    """
    Прогоняет полный эксперимент по списку seed'ов: обучение/загрузка модели, FGSM,
    расчёт метрик, сохранение артефактов и запись итогового JSON в RESULTS_PATH.
    """
    logger = setup_logger("cotai.main", os.path.join(config.LOGS_DIR, "main.log"))
    logger.info(f"Starting experiment {config.EXPERIMENT_ID}")
    logger.info("Loading NSL-KDD ...")

    train, test = load_raw(config.DATA_DIR)
    X_train, y_train, X_test, y_test, scaler, onehot_groups, columns = preprocess_data(
        train, test
    )
    logger.info(f"X_train={X_train.shape}  X_test={X_test.shape}")

    binary_indices = detect_binary_features(X_train, onehot_groups)
    logger.info(f"Binary non-onehot features: {len(binary_indices)}")

    runs = []
    for i, seed in enumerate(random_seeds, start=1):
        set_seeds(seed)
        run_id = f"run_{i:03d}"

        run_log_path = os.path.join(config.LOGS_DIR, f"{run_id}.log")
        run_logger = setup_logger(f"cotai.{run_id}", run_log_path)
        run_tee = Tee(sys.__stdout__, open(run_log_path, "a", encoding="utf-8"))

        old_stdout = sys.stdout
        sys.stdout = run_tee
        run_logger.info("=== %s (seed=%d) ===", run_id, seed)

        try:
            set_seeds(seed)
            model_path = model_path_for_seed(seed)
            if config.RETRAIN_PER_RUN:
                model, train_logs = get_or_train_model(
                    X_train, y_train, X_test, y_test, model_path, run_id, run_logger
                )
            else:
                model = train_or_load(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    config.MODEL_PATH,
                    config.EPOCHS,
                    config.BATCH_SIZE,
                )
                model_path = config.MODEL_PATH
                train_logs = {}

            y_before = predict_labels(model, X_test)

            X_adv = run_fgsm(
                model,
                X_test,
                config.FGM_EPS,
                onehot_groups=onehot_groups,
                binary_indices=binary_indices,
                project=config.PROJECT_ONEHOT,
                round_binary=config.ROUND_BINARY,
            )
            y_after = predict_labels(model, X_adv)

            valid_mask = check_validity(X_adv, onehot_groups)
            metrics = compute_metrics(
                y_test,
                y_before,
                y_after,
                X_test,
                X_adv,
                valid_mask,
            )

            run_logger.info(
                "metrics: acc_before=%.4f acc_after=%.4f ASR=%.4f "
                "success=%d/%d valid=%d/%d",
                metrics["accuracy_before_attack"],
                metrics["accuracy_after_attack"],
                metrics["attack_success_rate"],
                metrics["n_objects_successfully_attacked"],
                metrics["n_correctly_classified_before"],
                metrics["n_objects_valid_modified"],
                metrics["n_objects_modified"],
            )

            artifact_paths = {
                "model_path": model_path,
                "run_log": run_log_path,
                **train_logs,
            }
            if config.SAVE_PREDICTIONS:
                artifact_paths.update(
                    {
                        "model_path": model_path,
                        "predictions_before": os.path.join(
                            config.RESULTS_DIR, f"{run_id}_y_before.npy"
                        ),
                        "predictions_after": os.path.join(
                            config.RESULTS_DIR, f"{run_id}_y_after.npy"
                        ),
                        "adversarial_examples": os.path.join(
                            config.RESULTS_DIR, f"{run_id}_X_adv.npy"
                        ),
                    }
                )
                np.save(artifact_paths["predictions_before"], y_before)
                np.save(artifact_paths["predictions_after"], y_after)
                np.save(artifact_paths["adversarial_examples"], X_adv)

            block = build_run_block(
                run_id=run_id,
                seed=seed,
                status="success",
                attack_params={
                    "evaluation_method_id": config.EVALUATION_METHOD_ID,
                    "library": config.ATTACK_INFO["library"],
                    "method": config.ATTACK_INFO["method"],
                    "params": config.ATTACK_INFO["params"],
                    "onehot_projection": config.PROJECT_ONEHOT,
                    "binary_rounding": config.ROUND_BINARY,
                },
                metrics=metrics,
            )
            block.update(artifact_paths)
            runs.append(block)

        except Exception as e:
            runs.append(
                build_run_block(
                    run_id=run_id,
                    seed=seed,
                    status="error",
                    attack_params={},
                    metrics={},
                    error=f"{type(e).__name__}: {e}",
                )
            )
            run_logger.error("ERROR: %s", e)
            traceback.print_exc()
        finally:
            sys.stdout = old_stdout
            run_tee.close()

    payload = {
        "experiment": {
            "experiment_id": config.EXPERIMENT_ID,
            "description": config.EXPERIMENT_DESCRIPTION,
            "dataset": config.DATASET_INFO,
            "model": config.MODEL_INFO,
            "attack": config.ATTACK_INFO,
            "environment": {
                "application_environment_id": config.APPLICATION_ENVIRONMENT_ID,
                "life_cycle_stage_id": config.LIFE_CYCLE_STAGE_ID,
                "life_cycle_stage_description": config.LIFE_CYCLE_STAGE_DESCRIPTION,
                "object_domain_id": config.OBJECT_DOMAIN_ID,
                **collect_env_info(),
            },
            "preprocessing": {
                "numeric": "MinMaxScaler to [0,1] (fit on train)",
                "categorical": "one-hot encoding (protocol_type, service, flag)",
                "post_attack": (
                    (
                        "argmax-projection of one-hot blocks"
                        if config.PROJECT_ONEHOT
                        else "raw FGSM output"
                    )
                    + "; binary non-onehot features rounded to {0,1}"
                    if config.ROUND_BINARY
                    else ""
                ),
            },
            "retrain_per_run": config.RETRAIN_PER_RUN,
            "n_test_objects": int(len(X_test)),
            "n_binary_features": int(len(binary_indices)),
            "binary_feature_indices": binary_indices.tolist(),
            "feature_columns": columns,
        },
        "runs": runs,
    }

    write_results(config.RESULTS_PATH, payload)
    logger.info(f"\nResults written to {config.RESULTS_PATH}")


def main():
    runner(config.RANDOM_SEEDS)


if __name__ == "__main__":
    main()
