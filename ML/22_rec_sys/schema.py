from pydantic import BaseModel
import datetime


class UserGet(BaseModel):
    age: int
    city: str
    country: str
    exp_group: int
    gender: int
    user_id: int
    os: str
    source: str

    class Config:
        orm_mode = True


class PostGet(BaseModel):
    post_id: int
    text: str
    topic: str

    class Config:
        orm_mode = True


class FeedGet(BaseModel):
    timestamp: datetime.datetime
    user_id: int
    post_id: int
    action: str
    target: int

    class Config:
        orm_mode = True