from abc import abstractmethod

import numpy as np

from ...transport.model import BaseModel


class SileroModel(BaseModel):
    """
    The data type np.int16 (S16LE)
    """

    @abstractmethod
    def __call__(self, data: np.ndarray) -> list[float]:
        """
        The data type np.int16 (S16LE)
        """
        ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def clear(self) -> None: ...
