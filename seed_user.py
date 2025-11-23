from db import SessionLocal
from db_models import User

def create_user(
    email: str,
    residence_location: str | None = None,
    timezone: str = "Europe/Berlin",
):
    session = SessionLocal()
    try:
        user = User(
            email=email,
            timezone=timezone,
            residence_location=residence_location,
        )
        session.add(user)
        session.commit()

        session.refresh(user)
        print("Created user with id:", user.user_id)
        return user.user_id
    finally:
        session.close()

if __name__ == "__main__":
    create_user("janina@example.com", "Munich, Germany")
