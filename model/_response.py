from typing import TypeVar, Generic, Optional

from pydantic import BaseModel

T = TypeVar('T')

class ResponseBody(BaseModel,Generic[T]):
    """"""
    data: Optional[T] = None