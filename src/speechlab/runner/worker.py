from typing import Callable
from multiprocessing import Process

from loguru import logger


class Worker(Process):
    def __init__(
        self,
        name: str,
        target_func: Callable,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            target=self._target,
            args=args,
            kwargs=kwargs,
            daemon=False,  # Do not make a daemon to control termination
        )
        self._func = target_func
        self._log = logger.bind(name=name)

    def _target(
        self,
        *args,
        **kwargs,
    ) -> None:
        try:
            self._func(*args, **kwargs)
            self._log.info(f"Worker {self.name} completed successfully")
        except Exception as e:
            self._log.exception(f"Error in worker {self.name}: {e}")
        finally:
            logger.complete()

    def start(self) -> bool:
        try:
            super().start()
            self._log.info(f"Worker {self.name} (PID: {self.pid}) started successfully")
            return True
        except Exception as e:
            self._log.exception(f"Failed to start worker {self.name}: {e}")
            return False

    def stop(self, timeout: int) -> bool:
        if not self.is_alive():
            self._log.debug(f"Worker {self.name} is not running")
            return True

        try:
            self._log.debug(f"Terminating worker {self.name} (PID: {self.pid})")
            self.terminate()
            self.join(timeout)
            if self.is_alive():
                self._log.warning(f"Worker {self.name} did not terminate gracefully, killing")
                self.kill()
                self.join(1)
        except Exception as e:
            self._log.exception(f"Error stopping worker {self.name}: {e}")

        is_stopped = not self.is_alive()
        if is_stopped:
            self._log.debug(f"Stopped worker {self.name}")
        else:
            self._log.error(f"Failed to stop worker {self.name}")

        return is_stopped
