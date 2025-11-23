# db.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db_models import Base

# SQLite file "aura.db" in current directory
DATABASE_URL = "sqlite:///aura.db"

engine = create_engine(
    DATABASE_URL,
    echo=False,              # set True to see SQL in logs
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

def init_db():
    Base.metadata.create_all(bind=engine)
