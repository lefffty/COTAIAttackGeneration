import logging
import os
import sys


class Tee:
    """Дублирует поток вывода (stdout/stderr) одновременно в несколько приёмников, например в консоль и в файл."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self._streams:
            stream.flush()

    def close(self):
        """Закрывает все потоки, кроме sys.__stdout__ и sys.__stderr__."""
        for stream in self._streams:
            if stream is sys.__stdout__ or stream is sys.__stderr__:
                continue
            try:
                stream.close()
            except Exception:
                pass


def setup_logger(name: str, log_path: str) -> logging.Logger:
    """Создаёт логгер с записью в файл log_path и дублированием в stdout; старые хендлеры удаляются."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger
