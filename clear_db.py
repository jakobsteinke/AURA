from db import SessionLocal, engine
from db_models import Base

def clear_all_data():
    """
    Deletes ALL DATA from every table in aura.db,
    but keeps the tables/schema.
    """
    session = SessionLocal()
    try:
        Base.metadata.drop_all(bind=engine)

        Base.metadata.create_all(bind=engine)

        print("All data erased. Tables recreated empty.")
    finally:
        session.close()

if __name__ == "__main__":
    clear_all_data()
