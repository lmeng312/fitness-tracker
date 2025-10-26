"""
Database models for AI Health Tracker
SQLAlchemy ORM models for User, Workout, and Nutrition data
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Date, ForeignKey, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class User(Base):
    """User model for storing profile and Strava authentication"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    strava_user_id = Column(Integer, unique=True, nullable=True, index=True)
    access_token = Column(String(255), nullable=True)
    refresh_token = Column(String(255), nullable=True)
    token_expires_at = Column(Integer, nullable=True)  # Unix timestamp
    weight_lbs = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    workouts = relationship("Workout", back_populates="user", cascade="all, delete-orphan")
    nutrition_logs = relationship("Nutrition", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, strava_user_id={self.strava_user_id})>"


class Workout(Base):
    """Workout/activity model for storing fitness data from Strava or CSV uploads"""
    __tablename__ = "workouts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    strava_activity_id = Column(String(50), unique=True, nullable=True, index=True)  # Unique constraint for deduplication
    date = Column(Date, nullable=False, index=True)
    type = Column(String(50), nullable=False)  # Run, Ride, Swim, etc.
    duration_min = Column(Float, nullable=True)
    distance_mi = Column(Float, nullable=True)
    pace_min_mi = Column(Float, nullable=True)  # Minutes per mile
    calories_burned = Column(Integer, nullable=True)
    elevation_gain_ft = Column(Float, nullable=True)
    heart_rate_avg = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="workouts")
    
    def __repr__(self):
        return f"<Workout(id={self.id}, date={self.date}, type={self.type}, strava_id={self.strava_activity_id})>"


class Nutrition(Base):
    """Nutrition log model for storing daily nutrition data"""
    __tablename__ = "nutrition_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    method = Column(String(20), nullable=False)  # 'ai_estimate', 'manual', 'csv'
    source_description = Column(Text, nullable=True)  # Description of meal/food for AI estimates
    calories = Column(Integer, nullable=True)
    protein_g = Column(Float, nullable=True)
    carbs_g = Column(Float, nullable=True)
    fat_g = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="nutrition_logs")
    
    def __repr__(self):
        return f"<Nutrition(id={self.id}, date={self.date}, method={self.method}, calories={self.calories})>"

