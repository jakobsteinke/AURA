from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean,
    Date, DateTime, ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.sqlite import JSON
import datetime

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String, unique=True, nullable=False)
    timezone = Column(String, nullable=True)

    residence_location = Column(String, nullable=True)

    avg_sleep_7d_hours = Column(Float, default=0.0)
    avg_resting_hr_7d = Column(Float, default=0.0)
    avg_steps_7d = Column(Float, default=0.0)
    avg_screen_minutes_7d = Column(Float, default=0.0)
    avg_long_sessions_7d = Column(Float, default=0.0)

    current_week_start_date = Column(Date, nullable=True)
    weekly_sleep_score = Column(Integer, default=0)
    weekly_activity_score = Column(Integer, default=0)
    weekly_screen_hygiene_score = Column(Integer, default=0)
    weekly_overall_score = Column(Integer, default=0)
    weekly_bonus_awarded = Column(Boolean, default=False)

    healthy_sleep_streak_days = Column(Integer, default=0)
    movement_streak_days = Column(Integer, default=0)
    screen_hygiene_streak_days = Column(Integer, default=0)
    longest_healthy_sleep_streak_days = Column(Integer, default=0)
    longest_movement_streak_days = Column(Integer, default=0)

    total_aura_points = Column(Integer, default=0)
    current_level = Column(String, default="Beginner")
    points_this_week = Column(Integer, default=0)
    last_bonus_awarded_at = Column(DateTime, nullable=True)

    flag_potential_burnout = Column(Boolean, default=False)
    flag_high_screen_dependence = Column(Boolean, default=False)
    flag_low_activity = Column(Boolean, default=False)

    daily_metrics = relationship("DailyMetrics", back_populates="user")
    aura_outputs = relationship("AuraAgentOutput", back_populates="user")


class DailyMetrics(Base):
    __tablename__ = "daily_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    day = Column(Date, nullable=False)

    last_nights_sleep_duration_hours = Column(Float, nullable=True)
    resting_hr_bpm = Column(Integer, nullable=True)
    total_screen_minutes = Column(Integer, nullable=True)
    steps = Column(Integer, nullable=True)
    long_sessions_over_20_min = Column(Integer, nullable=True)

    user = relationship("User", back_populates="daily_metrics")

    __table_args__ = (
        UniqueConstraint("user_id", "day", name="uq_user_day"),
    )


class AuraAgentOutput(Base):
    __tablename__ = "aura_agent_outputs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    output = Column(JSON, nullable=False)

    context = Column(JSON, nullable=True)

    user = relationship("User", back_populates="aura_outputs")
