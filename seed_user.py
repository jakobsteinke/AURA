# seed_user.py
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

        # ensure we get the generated user_id
        session.refresh(user)
        print("Created user with id:", user.user_id)
        return user.user_id
    finally:
        session.close()

if __name__ == "__main__":
    # adjust residence_location as needed
    create_user("jakob@example.com", "Munich, Germany")
