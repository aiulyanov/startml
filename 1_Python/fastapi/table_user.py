from database import Base, engine, SessionLocal
from sqlalchemy import Column, Integer, String, func


class User(Base):
    __tablename__ = "user"
    __table_args__ = {"schema": "public"}
    id = Column(Integer, primary_key=True)
    gender = Column(Integer)
    age = Column(Integer)
    country = Column(String)
    city = Column(String)
    exp_group = Column(Integer)
    os = Column(String)
    source = Column(String)


if __name__ == "__main__":
    Base.metadata.create_all(engine)

    session = SessionLocal()
    result = (
        session.query(User.country, User.os, func.count(User.country).label("count"))
        .filter(User.exp_group == 3)
        .group_by(User.country, User.os)
        .having(func.count(User.country) > 100)
        .order_by(func.count(User.country).desc())
    )

    print([x for x in result])
