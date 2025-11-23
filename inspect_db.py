# inspect_db.py
from db import SessionLocal
from db_models import User, DailyMetrics

def show_users():
    session = SessionLocal()
    try:
        users = session.query(User).all()
        for u in users:
            print(
                "user_id:", u.user_id,
                "| email:", u.email,
                "| residence_location:", u.residence_location,
                "| timezone:", u.timezone,
            )
    finally:
        session.close()

def show_metrics(user_id: int):
    session = SessionLocal()
    try:
        metrics = session.query(DailyMetrics).filter_by(user_id=user_id).all()
        for m in metrics:
            print(
                "day:", m.day,
                "| sleep_hours:", m.last_nights_sleep_duration_hours,
                "| resting_hr_bpm:", m.resting_hr_bpm,
                "| total_screen_minutes:", m.total_screen_minutes,
                "| steps:", m.steps,
                "| long_sessions_over_20_min:", m.long_sessions_over_20_min,
            )
    finally:
        session.close()

if __name__ == "__main__":
    print("=== Users ===")
    show_users()
    print("=== Metrics for user 1 ===")
    show_metrics(1)
