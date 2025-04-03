import signal
from abc import ABC, abstractmethod

import zmq
from loguru import logger

from .message import BaseMessage


class BaseZMQServer(ABC):
    def __init__(self, address: str, name: str = "ZMQServer") -> None:
        self._stopped = False

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.ROUTER)
        self._socket.bind(address)

        self.log = logger.bind(name=name)
        self.log.info(f"Server started on {address}")

    def _on_stop(self, signum, frame) -> None:
        self.log.info(f"Received signal {signum}, stopping server")
        self._stopped = True

    def send(self, client_id: bytes, msg: BaseMessage, data: bytes | None = None) -> bool:
        try:
            if data is None:
                parts = [client_id, msg.pack()]
            else:
                parts = [client_id, msg.pack(), data]
            self._socket.send_multipart(parts)
        except zmq.ZMQError:
            self.log.exception(
                f"Failed to send message to client '{client_id.decode('utf-8', errors='replace')}', ZMQ error"
            )
            return False
        except Exception:
            self.log.exception(
                f"Failed to send message to client '{client_id.decode('utf-8', errors='replace')}'"
            )
            return False

        return True

    def recv2(self, msg_type: type) -> tuple[bytes, BaseMessage] | None:
        try:
            # [client_id, msg_bin]
            parts = self._socket.recv_multipart()
            if len(parts) != 2:
                self.log.warning(
                    f"Received malformed message, actual parts len: {len(parts)}, expected 2"
                )
                return None
            client_id = parts[0]
            msg = msg_type.unpack(parts[1])
            return client_id, msg
        except zmq.ZMQError:
            self.log.exception("Failed to recv message, ZMQ error")
            return None
        except Exception:
            self.log.exception("Failed to recv message")
            return None

    def recv3(self, msg_type: type) -> tuple[bytes, BaseMessage, bytes | None] | None:
        try:
            # [client_id, msg_bin, data | None]
            parts = self._socket.recv_multipart()
            if len(parts) < 2 or len(parts) > 3:
                self.log.warning(
                    f"Received malformed message, actual parts len: {len(parts)}, expected between 2 and 3"
                )
                return None
            client_id = parts[0]
            msg = msg_type.unpack(parts[1])
            if len(parts) == 3:
                data = parts[2]
            else:
                data = None
            return client_id, msg, data
        except zmq.ZMQError:
            self.log.exception("Failed to recv message, ZMQ error")
            return None
        except Exception:
            self.log.exception("Failed to recv message")
            return None

    def run(self) -> None:
        signal.signal(signal.SIGTERM, self._on_stop)
        signal.signal(signal.SIGINT, self._on_stop)

        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)

        try:
            self.log.info("Starting server loop")
            while not self._stopped:
                events = dict(poller.poll(timeout=1000))
                if self._socket in events and events[self._socket] == zmq.POLLIN:
                    self.process()
        except Exception:
            self.log.exception(f"Server error")
        finally:
            self.close()

    @abstractmethod
    def process(self) -> None: ...

    def close(self) -> None:
        self.log.debug("Closing server")
        try:
            if self._socket is not None:
                self._socket.close()
                self._socket = None
            if self._context is not None:
                self._context.term()
                self._context = None
        except Exception:
            self.log.exception(f"Error during server shutdown")
