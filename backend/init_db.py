"""
Database Initialization Script
Creates tables and optionally migrates existing JSON data to PostgreSQL
"""

import os
import json
from datetime import datetime
from pathlib import Path
from database import init_db, get_db
from models import User, Workout, Nutrition

def migrate_json_data():
    """Migrate existing JSON data files to PostgreSQL"""
    
    with get_db() as db:
        # Create default user (single-user mode)
        user = db.query(User).filter(User.id == 1).first()
        if not user:
            user = User(id=1, weight_lbs=150.0)  # Default weight, can be updated in settings
            db.add(user)
            db.commit()
            print("✅ Created default user (ID=1)")
        
        # Migrate Strava workout data
        strava_file = Path(__file__).parent / "my_strava_data.json"
        if strava_file.exists():
            print(f"📂 Found Strava data file: {strava_file}")
            with open(strava_file, "r") as f:
                strava_data = json.load(f)
            
            workouts_data = strava_data.get("workouts", [])
            imported_count = 0
            
            for workout_dict in workouts_data:
                # Check if workout already exists (by date and type)
                existing = db.query(Workout).filter(
                    Workout.user_id == 1,
                    Workout.date == datetime.fromisoformat(workout_dict["date"]).date(),
                    Workout.type == workout_dict.get("type", "Run")
                ).first()
                
                if not existing:
                    workout = Workout(
                        user_id=1,
                        date=datetime.fromisoformat(workout_dict["date"]).date(),
                        type=workout_dict.get("type", "Run"),
                        duration_min=workout_dict.get("duration_min"),
                        distance_mi=workout_dict.get("distance_mi"),
                        pace_min_mi=workout_dict.get("pace_min_mi"),
                        calories_burned=workout_dict.get("calories_burned"),
                        elevation_gain_ft=workout_dict.get("elevation_gain_ft"),
                        heart_rate_avg=workout_dict.get("heart_rate_avg")
                    )
                    db.add(workout)
                    imported_count += 1
            
            db.commit()
            print(f"✅ Imported {imported_count} workouts from Strava JSON")
        else:
            print("⚠️  No Strava data file found")
        
        # Migrate nutrition log data
        nutrition_file = Path(__file__).parent / "my_nutrition_logs.json"
        if nutrition_file.exists():
            print(f"📂 Found nutrition data file: {nutrition_file}")
            with open(nutrition_file, "r") as f:
                nutrition_data = json.load(f)
            
            entries = nutrition_data.get("entries", [])
            imported_count = 0
            
            for entry_dict in entries:
                # Check if entry already exists (by date)
                existing = db.query(Nutrition).filter(
                    Nutrition.user_id == 1,
                    Nutrition.date == datetime.fromisoformat(entry_dict["date"]).date()
                ).first()
                
                if not existing:
                    nutrition = Nutrition(
                        user_id=1,
                        date=datetime.fromisoformat(entry_dict["date"]).date(),
                        method=entry_dict.get("method", "manual"),
                        source_description=entry_dict.get("source_description"),
                        calories=entry_dict.get("calories"),
                        protein_g=entry_dict.get("protein_g"),
                        carbs_g=entry_dict.get("carbs_g"),
                        fat_g=entry_dict.get("fat_g")
                    )
                    db.add(nutrition)
                    imported_count += 1
            
            db.commit()
            print(f"✅ Imported {imported_count} nutrition entries from JSON")
        else:
            print("⚠️  No nutrition data file found")


def main():
    """Main initialization function"""
    print("🚀 Initializing database...")
    
    # Create all tables
    init_db()
    
    # Migrate existing data
    print("\n📦 Migrating existing JSON data...")
    migrate_json_data()
    
    print("\n✅ Database initialization complete!")
    print("💡 You can now start the server with: python health_main.py")


if __name__ == "__main__":
    main()

