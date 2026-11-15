import numpy as np


def compute_metrics(y_true, y_pred_before, y_pred_after, X_orig, X_adv, valid_mask):
    """
    Считает метрики эксперимента до и после атаки.

    Возвращает dict:
        accuracy_before_attack         = correct_before / n_total
        accuracy_after_attack          = correct_after  / n_total
        attack_success_rate (ASR)      = n_success / n_correct_before
                                         (0.0, если n_correct_before == 0)
        n_objects_total                = len(y_true)
        n_objects_modified             = Σ(||X_adv - X_orig||_∞ > 1e-9)
        n_objects_valid_modified       = Σ(modified ∧ valid)
        n_objects_successfully_attacked = Σ(correct_before ∧ ¬correct_after
                                            ∧ modified ∧ valid)
        n_correctly_classified_before  = Σ(correct_before)
    """
    n_total = len(y_true)

    modified_mask = np.any(np.abs(X_orig - X_adv) > 1e-9, axis=1)
    n_modified = int(modified_mask.sum())

    n_valid = int((modified_mask & valid_mask).sum())

    correct_before = y_pred_before == y_true
    correct_after = y_pred_after == y_true
    n_correct_before = int(correct_before.sum())

    success_mask = correct_before & (~correct_after) & modified_mask & valid_mask
    n_success = int(success_mask.sum())

    accuracy_before = float(correct_before.mean())
    accuracy_after = float(correct_after.mean())
    asr = float(n_success / n_correct_before) if n_correct_before > 0 else 0.0

    return {
        "accuracy_before_attack": accuracy_before,
        "accuracy_after_attack": accuracy_after,
        "attack_success_rate": asr,
        "n_objects_total": n_total,
        "n_objects_modified": n_modified,
        "n_objects_valid_modified": n_valid,
        "n_objects_successfully_attacked": n_success,
        "n_correctly_classified_before": n_correct_before,
    }
