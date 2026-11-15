import json
import os

import config


def make_evidence(run_id, metric_id, value):
    """Формирует одну запись измерения: одна метрика — одно числовое значение (has_metric_value)."""
    return {
        "experiment_id": config.EXPERIMENT_ID,
        "run_id": run_id,
        "evidence_id": f"ev_{run_id}_{metric_id}",
        "ai_system_id": config.AI_SYSTEM_ID,
        "evaluation_method_id": config.EVALUATION_METHOD_ID,
        "application_environment_id": config.APPLICATION_ENVIRONMENT_ID,
        "life_cycle_stage_id": config.LIFE_CYCLE_STAGE_ID,
        "object_domain_id": config.OBJECT_DOMAIN_ID,
        "metric_id": metric_id,
        "has_metric_value": value,
    }


def build_run_block(
    run_id, seed, status, attack_params, metrics: dict, error: str | None = None
):
    """Собирает блок прогона: параметры атаки, seed, статус и записи измерений по каждой метрике."""
    measurements = [make_evidence(run_id, k, v) for k, v in metrics.items()]
    block = {
        "run_id": run_id,
        "status": status,
        "random_seed": seed,
        "attack": attack_params,
        "measurements": measurements,
    }
    if error:
        block["error"] = error
    return block


def write_results(path, payload):
    """Сохраняет payload в JSON-файл с отступами и UTF-8, создавая родительский каталог при необходимости."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
