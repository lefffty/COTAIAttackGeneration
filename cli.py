import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import config


def _print_header(title: str):
    line = "=" * 128
    print(f"\n{line}\n{title}\n{line}")


def _print_config_summary():
    items = [
        ("experiment_id", config.EXPERIMENT_ID),
        ("data_dir", config.DATA_DIR),
        ("results_path", config.RESULTS_PATH),
        ("models_dir", config.MODELS_DIR),
        ("logs_dir", config.LOGS_DIR),
        ("retrain_per_run", config.RETRAIN_PER_RUN),
        ("project_onehot", config.PROJECT_ONEHOT),
        ("round_binary", config.ROUND_BINARY),
        ("save_predictions", config.SAVE_PREDICTIONS),
        ("save_training_csv", getattr(config, "SAVE_TRAINING_CSV", True)),
        ("save_training_txt", getattr(config, "SAVE_TRAINING_TXT", True)),
        ("n_runs / seeds", f"{len(config.RANDOM_SEEDS)} / {config.RANDOM_SEEDS}"),
        ("fgm_eps", config.FGM_EPS),
        ("epochs", config.EPOCHS),
        ("batch_size", config.BATCH_SIZE),
        ("model", config.MODEL_INFO["name"]),
        ("dataset", config.DATASET_INFO["name"]),
    ]
    width = max(len(k) for k, _ in items)
    for k, v in items:
        print(f"  {k:<{width}} : {v}")


def _apply_run_overrides(args):
    if args.seeds is not None:
        config.RANDOM_SEEDS = list(args.seeds)
    if args.eps is not None:
        config.FGM_EPS = args.eps
    if args.epochs is not None:
        config.EPOCHS = args.epochs
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.retrain is not None:
        config.RETRAIN_PER_RUN = args.retrain
    if args.project_onehot is not None:
        config.PROJECT_ONEHOT = args.project_onehot
    if args.round_binary is not None:
        config.ROUND_BINARY = args.round_binary
    if args.predictions is not None:
        config.SAVE_PREDICTIONS = args.predictions
    if args.training_logs is not None:
        config.SAVE_TRAINING_CSV = args.training_logs
        config.SAVE_TRAINING_TXT = args.training_logs
    if args.output is not None:
        config.RESULTS_PATH = os.path.abspath(args.output)
    if args.data_dir is not None:
        config.DATA_DIR = os.path.abspath(args.data_dir)
    if args.experiment_id is not None:
        config.EXPERIMENT_ID = args.experiment_id


def cmd_run(args):
    _apply_run_overrides(args)

    _print_header(f"COTAI experiment: {config.EXPERIMENT_ID}")
    _print_config_summary()

    if args.dry_run:
        print("\n[dry-run] Конфигурация валидна, запуск пропущен.")
        return 0

    for fname in (config.DATASET_INFO["train_file"], config.DATASET_INFO["test_file"]):
        fpath = os.path.join(config.DATA_DIR, fname)
        if not os.path.exists(fpath):
            print(f"\n[ERROR] Не найден файл датасета: {fpath}", file=sys.stderr)
            return 2

    import run_experiments

    try:
        run_experiments.main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Прервано пользователем.", file=sys.stderr)
        return 130
    return 0


def cmd_show_config(args):
    _print_header("Effective COTAI configuration")
    _print_config_summary()
    return 0


def _validate_minimal(payload: dict) -> list[str]:
    errors = []

    required_evidence = [
        "experiment_id",
        "run_id",
        "evidence_id",
        "ai_system_id",
        "evaluation_method_id",
        "application_environment_id",
        "life_cycle_stage_id",
        "object_domain_id",
        "metric_id",
        "has_metric_value",
    ]

    if "experiment" not in payload:
        errors.append("missing top-level 'experiment'")
    if "runs" not in payload or not isinstance(payload["runs"], list):
        errors.append("missing or non-list 'runs'")
        return errors

    seen_ids = set()
    for i, run in enumerate(payload["runs"]):
        for key in ("run_id", "status", "measurements"):
            if key not in run:
                errors.append(f"runs[{i}] missing '{key}'")
        if run.get("status") not in ("success", "error", "no_result"):
            errors.append(f"runs[{i}].status invalid: {run.get('status')!r}")
        for j, m in enumerate(run.get("measurements", [])):
            for key in required_evidence:
                if key not in m:
                    errors.append(f"runs[{i}].measurements[{j}] missing '{key}'")
            ev = m.get("evidence_id")
            if ev in seen_ids:
                errors.append(f"duplicate evidence_id: {ev}")
            seen_ids.add(ev)
            v = m.get("has_metric_value")
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                errors.append(
                    f"runs[{i}].measurements[{j}] has_metric_value not numeric: {v!r}"
                )
    return errors


def cmd_validate(args):
    results_path = os.path.abspath(args.results or config.RESULTS_PATH)
    schema_path = os.path.abspath(args.schema)

    if not os.path.exists(results_path):
        print(f"[ERROR] Файл результатов не найден: {results_path}", file=sys.stderr)
        return 2
    if not os.path.exists(schema_path):
        print(
            f"[WARN] Схема не найдена: {schema_path}; используется минимальная проверка."
        )

    with open(results_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    _print_header(f"Validating {results_path}")

    errors: list[str] = []
    try:
        import jsonschema

        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        validator = jsonschema.Draft202012Validator(schema)
        errors = [
            f"{'/'.join(str(p) for p in e.path)}: {e.message}"
            for e in sorted(validator.iter_errors(payload), key=lambda e: e.path)
        ]
    except ImportError:
        print("[WARN] jsonschema не установлен — минимальная проверка.")
        errors = _validate_minimal(payload)

    if not errors:
        print("  OK: JSON валиден по схеме.")
        n_runs = len(payload.get("runs", []))
        n_ev = sum(len(r.get("measurements", [])) for r in payload.get("runs", []))
        print(f"  runs={n_runs}  evidence_records={n_ev}")
        return 0

    print(f"  Найдено {len(errors)} ошибок:")
    for e in errors:
        print(f"   - {e}")
    return 1


def cmd_summarize(args):
    results_path = os.path.abspath(args.results or config.RESULTS_PATH)
    if not os.path.exists(results_path):
        print(f"[ERROR] Файл результатов не найден: {results_path}", file=sys.stderr)
        return 2

    with open(results_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    _print_header(f"Summary: {payload['experiment']['experiment_id']}")

    rows = []
    for run in payload.get("runs", []):
        rid = run.get("run_id")
        status = run.get("status")
        seed = run.get("random_seed")
        metrics = {
            m["metric_id"]: m["has_metric_value"] for m in run.get("measurements", [])
        }
        rows.append((rid, status, seed, metrics))

    cols = [
        ("run_id", "run"),
        ("seed", "seed"),
        ("status", "status"),
        ("accuracy_before_attack", "acc_before"),
        ("accuracy_after_attack", "acc_after"),
        ("attack_success_rate", "ASR"),
        ("n_objects_total", "n_total"),
        ("n_objects_modified", "n_modified"),
        ("n_objects_valid_modified", "n_valid"),
        ("n_objects_successfully_attacked", "n_success"),
    ]
    widths = [max(len(h), 8) for _, h in cols]
    header = " | ".join(h.ljust(w) for (_, h), w in zip(cols, widths))
    print(header)
    print("-" * len(header))

    for rid, status, seed, m in rows:
        values = []
        for (key, _), w in zip(cols, widths):
            if key == "run_id":
                v = rid
            elif key == "seed":
                v = str(seed)
            elif key == "status":
                v = status
            else:
                val = m.get(key, "")
                v = f"{val:.4f}" if isinstance(val, float) else str(val)
            values.append(str(v).ljust(w))
        print(" | ".join(values))

    success_rows = [m for _, s, _, m in rows if s == "success"]
    if len(success_rows) > 1:
        print("\nMean ± std (success runs):")
        for key in (
            "accuracy_before_attack",
            "accuracy_after_attack",
            "attack_success_rate",
        ):
            vals = [m[key] for m in success_rows if key in m]
            if vals:
                import statistics as st

                print(f"  {key:<24}: {st.mean(vals):.4f} ± {st.stdev(vals):.4f}")
    return 0


def cmd_clean(args):
    _print_header("Cleaning artifacts")
    targets = []
    if args.logs:
        targets.append(("logs", config.LOGS_DIR))
    if args.results:
        targets.append(("results", config.RESULTS_DIR))
    if args.models:
        targets.append(("models", config.MODELS_DIR))

    if not targets:
        print("[INFO] Ничего не выбрано; используйте --logs/--results/--models/all.")
        return 0

    for name, path in targets:
        if not os.path.isdir(path):
            print(f"  [skip] {name}: не существует")
            continue
        files = list(Path(path).glob("*"))
        print(f"  {name:<8}: {path}  ({len(files)} файлов)")
        if args.yes or _confirm(f"Удалить содержимое {name}? [y/N] "):
            for f in files:
                if f.is_file():
                    f.unlink()
                else:
                    shutil.rmtree(f)
            print(f"  -> очищено: {name}")
    return 0


def _confirm(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() in ("y", "yes", "да")
    except (EOFError, KeyboardInterrupt):
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cotai",
        description="COTAI attack-generation experiments CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser(
        "run",
        help="Запустить эксперимент",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_run.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Список random seed'ов (например: 42 43 44)",
    )
    p_run.add_argument("--eps", type=float, help="Параметр eps для FGSM")
    p_run.add_argument("--epochs", type=int, help="Число эпох обучения")
    p_run.add_argument("--batch-size", type=int, help="Размер батча")
    p_run.add_argument(
        "--retrain",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Переобучать модель на каждом seed",
    )
    p_run.add_argument(
        "--project-onehot",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Схлопывать one-hot обратно после FGSM",
    )
    p_run.add_argument(
        "--round-binary",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Округлять бинарные признаки до {0,1}",
    )
    p_run.add_argument(
        "--predictions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Сохранять y_before/y_after/X_adv в .npy",
    )
    p_run.add_argument(
        "--training-logs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Сохранять CSV/TXT-логи обучения",
    )
    p_run.add_argument("--output", type=str, help="Путь к JSON с результатами")
    p_run.add_argument("--data-dir", type=str, help="Каталог с файлами NSL-KDD")
    p_run.add_argument("--experiment-id", type=str, help="Переопределить experiment_id")
    p_run.add_argument(
        "--dry-run",
        action="store_true",
        help="Показать конфигурацию и выйти без запуска",
    )

    sub.add_parser("show-config", help="Печать текущей конфигурации")

    p_val = sub.add_parser("validate", help="Проверить JSON по схеме")
    p_val.add_argument(
        "--results", type=str, help="Путь к JSON (по умолчанию config.RESULTS_PATH)"
    )
    p_val.add_argument(
        "--schema",
        type=str,
        default=os.path.join(
            config.PROJECT_ROOT, "schema", "cotai_evidence.schema.json"
        ),
        help="Путь к JSON-схеме",
    )

    p_sum = sub.add_parser("summarize", help="Сводная таблица метрик")
    p_sum.add_argument(
        "--results", type=str, help="Путь к JSON (по умолчанию config.RESULTS_PATH)"
    )

    p_clean = sub.add_parser("clean", help="Очистка артефактов")
    p_clean.add_argument("--logs", action="store_true", help="Удалить logs/")
    p_clean.add_argument("--results", action="store_true", help="Удалить results/")
    p_clean.add_argument("--models", action="store_true", help="Удалить models/")
    p_clean.add_argument(
        "--all", action="store_true", help="Удалить всё вышеперечисленное"
    )
    p_clean.add_argument("--yes", action="store_true", help="Без подтверждения")

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "clean" and getattr(args, "all", False):
        args.logs = args.results = args.models = True

    dispatch = {
        "run": cmd_run,
        "show-config": cmd_show_config,
        "validate": cmd_validate,
        "summarize": cmd_summarize,
        "clean": cmd_clean,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
