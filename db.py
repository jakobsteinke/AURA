from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db_models import Base

DATABASE_URL = "sqlite:///aura.db"

engine = create_engine(
    DATABASE_URL,
    echo=False,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

def init_db():
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    print("Creating database tables in aura.db ...")
    init_db()
    print("Done.")
