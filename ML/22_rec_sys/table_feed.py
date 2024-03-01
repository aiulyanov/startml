from database import Base, engine, SessionLocal
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from table_post import Post
from table_user import User


class Feed(Base):
    __tablename__ = "feed_data"
    __table_args__ = {"schema": "public"}
    user_id = Column(Integer, ForeignKey(User.user_id), primary_key=True)
    # user = relationship("User")
    post_id = Column(Integer, ForeignKey(Post.post_id), primary_key=True)
    # post = relationship("Post")
    action = Column(String)
    timestamp = Column(DateTime)
    target = Column(Integer)


if __name__ == "__main__":
    Base.metadata.create_all(engine)

    session = SessionLocal()
    for obj in session.query(Feed).order_by(Feed.timestamp.desc()).limit(10).all():
        print(type(obj.user_id), type(obj.post_id), type(obj.action), type(obj.timestamp))
