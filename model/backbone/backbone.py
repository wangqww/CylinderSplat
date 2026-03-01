from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from jaxtyping import Float
from torch import Tensor, nn


T = TypeVar("T")


class Backbone(nn.Module, ABC, Generic[T]):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(
        self,
        context,
    ) -> Float[Tensor, "batch view d_out height width"]:
        pass

    @property
    @abstractmethod
    def d_out(self) -> int:
        pass
