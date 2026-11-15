import numpy as np


def check_validity(X_adv, onehot_groups, tol=0.15):
    """
    Проверяет допустимость состязательных примеров.

    Возвращает булев массив: True — запись допустима, False — нереалистична.
    Критерии:
      1. все значения в [0, 1] (с допуском 1e-6);
      2. для каждого one-hot блока: |Σ - 1| < tol и max > 0.5
         (ровно один доминирующий признак).
    """
    n = len(X_adv)
    in_bounds = np.all((X_adv >= -1e-6) & (X_adv <= 1.0 + 1e-6), axis=1)

    valid_cat = np.ones(n, dtype=bool)
    for s, e in onehot_groups:
        block = X_adv[:, s:e]
        block_sum = block.sum(axis=1)
        block_max = block.max(axis=1)
        valid_cat &= (np.abs(block_sum - 1.0) < tol) & (block_max > 0.5)

    return in_bounds & valid_cat
