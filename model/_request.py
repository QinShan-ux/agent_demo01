from typing import TypeVar, Generic, Optional

from pydantic import BaseModel

T = TypeVar('T')

class RequestBody(BaseModel,Generic[T]):
    """"""
    data: Optional[T] = None 