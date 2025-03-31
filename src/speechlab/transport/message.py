from typing import Self

from pydantic import BaseModel
from ormsgpack import OPT_SERIALIZE_PYDANTIC, packb, unpackb


class BaseMessage(BaseModel):
    def pack(self) -> bytes:
        return packb(self, option=OPT_SERIALIZE_PYDANTIC)

    @classmethod
    def unpack(cls, data: bytes) -> Self:
        return cls.model_validate(unpackb(data))
