import zmq
import zmq.asyncio
from loguru import logger

from .message import BaseMessage


class BaseZMQAsyncClient:
    def __init__(self, address: str, client_id: str = None, name: str = "ZMQClient") -> None:
        self._client_id = client_id or f"client-{id(self)}"

        # ZMQ setup
        self._context = zmq.asyncio.Context()
        self._socket = self._context.socket(zmq.DEALER)
        self._socket.setsockopt_string(zmq.IDENTITY, self._client_id)
        self._socket.connect(address)

        self.log = logger.bind(name=name)
        self.log.info(f"Connected to server at {address}")

    async def send(self, msg: BaseMessage, data: bytes | None = None) -> bool:
        try:
            if data is None:
                parts = [msg.pack()]
            else:
                parts = [msg.pack(), data]
            await self._socket.send_multipart(parts)
        except zmq.ZMQError:
            self.log.exception("Failed to send message, ZMQ error")
            return False
        except Exception:
            self.log.exception("Failed to send message")
            return False

        return True

    async def recv(self, msg_type: type) -> tuple[BaseMessage, bytes | None] | None:
        parts = None
        try:
            parts = await self._socket.recv_multipart()
            cnt = len(parts)
            if cnt != 1 and cnt != 2:
                self.log.warning(
                    f"Received malformed message, actual parts len: {cnt}, expected: 1 or 2"
                )
                return None
            msg = msg_type.unpack(parts[0])
            data = None if cnt == 1 else parts[1]
            return msg, data
        except zmq.ZMQError:
            self.log.exception("Failed to recv message, ZMQ error")
            return None
        except Exception:
            self.log.exception("Failed to recv message")
            return None

    def close(self) -> None:
        self.log.info("Closing client connection")
        try:
            if self._socket is not None:
                self._socket.close()
                self._socket = None
            if self._context is not None:
                self._context.term()
                self._context = None
        except Exception:
            self.log.error("Error during client shutdown")
