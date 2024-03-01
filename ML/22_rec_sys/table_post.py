from database import Base, engine, SessionLocal
from sqlalchemy import Column, Integer, String


class Post(Base):
    __tablename__ = "post_text_df"
    __table_args__ = {"schema": "public"}
    post_id = Column(Integer, primary_key=True, name="post_id")
    text = Column(String)
    topic = Column(String)


if __name__ == "__main__":
    Base.metadata.create_all(engine)

    session = SessionLocal()
    result = [
        obj.post_id
        for obj in session.query(Post)
        .filter(Post.topic == "business")
        .order_by(Post.post_id.desc())
        .limit(10)
        .all()
    ]
    print(result)
