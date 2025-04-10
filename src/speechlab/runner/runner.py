import sys
import atexit
import signal
import threading
from typing import Callable

from loguru import logger

from .worker import Worker

LOGURU_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | <cyan>[{extra[name]}]</cyan> "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)


class Runner:
    def __init__(self) -> None:
        logger.configure(extra={"name": "N/A"})
        logger.remove()
        logger.add(sys.stdout, format=LOGURU_FORMAT, enqueue=True, level="DEBUG")

        self._log = logger.bind(name="Runner")
        self._workers: dict[str, Worker] = {}
        self._lock = threading.RLock()

        atexit.register(self.cleanup)
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)

        self._log.info("Starting")

    def start(
        self,
        name: str,
        target_func: Callable,
        *args,
        **kwargs,
    ) -> bool:
        with self._lock:
            if name in self._workers and self._workers[name].is_alive():
                self._log.warning(f"Worker {name} is already running")
                return False
            worker = Worker(name, target_func, *args, **kwargs)
            if worker.start():
                self._workers[name] = worker
                return True

            return False

    def stop(self, name: str, timeout: int = 5) -> bool:
        with self._lock:
            if name not in self._workers:
                self._log.info(f"Worker {name} not found")
                return True

            worker = self._workers[name]
            if worker.stop(timeout):
                del self._workers[name]
                return True

            return False

    def wait(self) -> None:
        self._log.info("Main process is running. Press Ctrl+C to stop")
        while True:
            signal.pause()

    def stop_all(self) -> None:
        with self._lock:
            for name in list(self._workers.keys()):
                self.stop(name)

    def cleanup(self) -> None:
        self._log.info("Cleaning up workers...")
        self.stop_all()

    def signal_handler(self, sig, frame) -> None:
        self._log.info(f"Received signal {sig}, shutting down...")
        self.cleanup()
        sys.exit(0)

    def list_workers(self) -> list:
        with self._lock:
            if not self._workers:
                self._log.info("No active workers")
                return []
            result = []
            for name, worker in self._workers.items():
                is_alive = worker.is_alive()
                status = "Running" if is_alive else "Stopped"
                pid = worker.pid if is_alive else "N/A"
                self._log.info(f"Worker {name} (PID: {pid}): {status}")
                result.append({"name": name, "pid": pid, "status": status})
            return result

    def is_worker_running(self, name: str) -> bool:
        with self._lock:
            return name in self._workers and self._workers[name].is_alive()

    def get_worker_info(self, name: str) -> dict[str, any] | None:
        with self._lock:
            if name not in self._workers:
                return None
            worker = self._workers[name]
            return {
                "name": name,
                "pid": worker.pid if worker.is_alive() else None,
                "is_alive": worker.is_alive(),
                "exitcode": worker.exitcode,
            }
