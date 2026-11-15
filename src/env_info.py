import platform
import sys


def collect_env_info():
    """Собирает версии Python, ОС и ключевых библиотек для записи в блок environment JSON."""
    info = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "libraries": {},
    }
    for mod in ("numpy", "pandas", "sklearn", "tensorflow", "art"):
        try:
            m = __import__(mod)
            info["libraries"][mod] = getattr(m, "__version__", "unknown")
        except Exception as e:
            info["libraries"][mod] = f"unavailable ({e})"
    return info
