from fastapi import FastAPI, HTTPException, Depends
from typing import List

from sqlalchemy.orm import Session
from sqlalchemy import func

from database import SessionLocal
from table_user import User
from table_post import Post
from table_feed import Feed
from schema import UserGet, PostGet, FeedGet


app = FastAPI()


def get_db():
    with SessionLocal() as db:
        return db


@app.get("/user/{id}", response_model=UserGet)
def get_user(id: int, db: Session = Depends(get_db)):
    result = db.query(User).filter(User.id == id).first()
    if not result:
        raise HTTPException(404, "user not found")
    return result


@app.get("/post/{id}", response_model=PostGet)
def get_post(id: int, db: Session = Depends(get_db)):
    result = db.query(Post).filter(Post.id == id).first()
    if not result:
        raise HTTPException(404, "post not found")
    return result


@app.get("/user/{id}/feed", response_model=List[FeedGet])
def get_user(id: int, limit: int = 10, db: Session = Depends(get_db)):
    result = (
        db.query(Feed)
        .filter(Feed.user_id == id)
        .order_by(Feed.time.desc())
        .limit(limit)
        .all()
    )
    if not result:
        return []
    return [
        FeedGet(
            user_id=item.user_id,
            post_id=item.post_id,
            action=item.action,
            time=item.time,
        )
        for item in result
    ]


@app.get("/post/{id}/feed", response_model=List[FeedGet])
def get_post(id: int, limit: int = 10, db: Session = Depends(get_db)):
    result = (
        db.query(Feed)
        .filter(Feed.post_id == id)
        .order_by(Feed.time.desc())
        .limit(limit)
        .all()
    )
    if not result:
        return []
    return [
        FeedGet(
            user_id=item.user_id,
            post_id=item.post_id,
            action=item.action,
            time=item.time,
        )
        for item in result
    ]


@app.get("/post/recommendations/", response_model=List[PostGet])
def get_post_recommendations(limit: int = 10, db: Session = Depends(get_db)):
    result = (
        db.query(Post, func.count(Post.id).label("like_count"))
        .filter(Feed.action == "like")
        .join(Feed, Feed.post_id == Post.id)
        .group_by(Post.id)
        .order_by(func.count(Post.id).desc())
        .limit(limit)
        .all()
    )
    if not result:
        return []
    return [
        PostGet(id=item.Post.id, text=item.Post.text, topic=item.Post.topic)
        for item in result
    ]
