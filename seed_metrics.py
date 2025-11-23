# seed_metrics.py
import datetime
from db import SessionLocal
from db_models import DailyMetrics

def add_today_metrics(
    user_id: int,
    sleep_hours: float,
    resting_hr: int,
    screen_minutes: int,
    steps: int,
    long_sessions: int,
):
    session = SessionLocal()
    try:
        today = datetime.date.today()

        # if you want to overwrite existing metrics for today, delete or update first
        metrics = DailyMetrics(
            user_id=user_id,
            day=today,
            last_nights_sleep_duration_hours=sleep_hours,
            resting_hr_bpm=resting_hr,
            total_screen_minutes=screen_minutes,
            steps=steps,
            long_sessions_over_20_min=long_sessions,
        )

        session.add(metrics)
        session.commit()
        print(f"Inserted metrics for user {user_id} on {today}")
    finally:
        session.close()

if __name__ == "__main__":
    # use the id printed from seed_user.py
    add_today_metrics(
        user_id=1,
        sleep_hours=6.5,
        resting_hr=65,
        screen_minutes=220,
        steps=8000,
        long_sessions=2,
    )
