# Deployment Guide - AI Health Tracker with Strava Integration

This guide walks you through deploying your fitness tracker to Render.com with automatic Strava workout imports.

## Overview

The app will:
- ✅ Run on Render.com (free tier)
- ✅ Use PostgreSQL for data storage
- ✅ Automatically import workouts from Strava via webhooks
- ✅ Keep all existing features (nutrition logging, AI insights, charts)

## Prerequisites

1. **GitHub Account** - Your code needs to be in a GitHub repository
2. **Render Account** - Sign up at https://render.com (free)
3. **Strava Account** - You already have this!

## Step 1: Prepare Local Environment

### 1.1 Install New Dependencies

```bash
cd backend
source venv/bin/activate  # Or activate your virtual environment
pip install -r requirements.txt
```

This installs PostgreSQL support, SQLAlchemy, and other new dependencies.

### 1.2 Set Up Local Environment Variables

Copy the example environment file:

```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env` with your local settings:

```bash
# For local testing with SQLite (easiest):
DATABASE_URL=sqlite:///./fitness_tracker.db
BASE_URL=http://localhost:8001

# Strava (get from Step 2):
STRAVA_CLIENT_ID=your_client_id
STRAVA_CLIENT_SECRET=your_client_secret
STRAVA_WEBHOOK_VERIFY_TOKEN=my_random_secret_token

# OpenAI (you already have this):
OPENAI_API_KEY=your_openai_key
```

### 1.3 Initialize Database & Migrate Data

Run the initialization script to create tables and migrate your existing JSON data:

```bash
python backend/init_db.py
```

You should see:
```
✅ Database tables created successfully
📦 Migrating existing JSON data...
✅ Imported X workouts from Strava JSON
✅ Imported X nutrition entries from JSON
✅ Database initialization complete!
```

### 1.4 Test Locally

Start the server:

```bash
cd backend
python health_main.py
```

Visit http://localhost:8001/dashboard.html and verify:
- ✅ Your workouts appear
- ✅ Nutrition data loads
- ✅ AI insights work
- ✅ Charts render correctly

## Step 2: Register Strava API Application

1. Go to https://www.strava.com/settings/api
2. Click "Create New App"
3. Fill in the form:
   - **Application Name**: AI Health Tracker (or your choice)
   - **Category**: Training
   - **Club**: Leave empty
   - **Website**: http://localhost:8001 (update after deployment)
   - **Authorization Callback Domain**: localhost (update after deployment)
   - **Application Description**: Personal fitness tracker with AI insights
4. Click "Create"
5. Note your **Client ID** and **Client Secret**
6. Add these to your `backend/.env` file (local testing)

## Step 3: Push to GitHub

If you haven't already, push your code to GitHub:

```bash
git add .
git commit -m "Add Strava integration and PostgreSQL support"
git push origin main
```

## Step 4: Deploy to Render

### 4.1 Create New Web Service

1. Log in to https://render.com
2. Click "New +" → "Blueprint"
3. Connect your GitHub repository
4. Render will detect `render.yaml` and show:
   - **fitness-tracker-db** (PostgreSQL database)
   - **fitness-tracker-backend** (Web service)
5. Click "Apply"

### 4.2 Configure Environment Variables

While the services are deploying, go to the **fitness-tracker-backend** service:

1. Go to "Environment" tab
2. Add these environment variables:

```
BASE_URL=https://fitness-tracker-backend.onrender.com  (or your actual URL)
STRAVA_CLIENT_ID=<from Step 2>
STRAVA_CLIENT_SECRET=<from Step 2>
STRAVA_WEBHOOK_VERIFY_TOKEN=<any random string>
OPENAI_API_KEY=<your OpenAI key>
```

3. Click "Save Changes"
4. The service will redeploy automatically

### 4.3 Get Your Production URL

After deployment completes, note your app's URL:
```
https://fitness-tracker-backend.onrender.com
```

## Step 5: Update Strava App for Production

1. Go back to https://www.strava.com/settings/api
2. Edit your application
3. Update:
   - **Website**: https://fitness-tracker-backend.onrender.com
   - **Authorization Callback Domain**: fitness-tracker-backend.onrender.com
4. Click "Update"

## Step 6: Initialize Production Database

### 6.1 Run Database Initialization

From Render dashboard:

1. Go to **fitness-tracker-backend** service
2. Click "Shell" tab
3. Run:

```bash
cd backend
python init_db.py
```

This will:
- Create all database tables
- Be ready for Strava data import

### 6.2 Connect Strava (First Time)

1. Visit your app: https://fitness-tracker-backend.onrender.com/settings.html
2. Click "🔗 Connect to Strava"
3. Authorize the app on Strava
4. You'll be redirected back - your last 30 days of workouts will sync automatically!

## Step 7: Set Up Strava Webhooks (Automatic Sync)

This enables real-time workout imports whenever you complete an activity on Strava.

### 7.1 Register Webhook Subscription

From Render Shell:

```bash
cd backend
python setup_webhook.py https://fitness-tracker-backend.onrender.com/webhooks/strava
```

You should see:
```
✅ Webhook subscription created successfully!
   Subscription ID: 12345
   Callback URL: https://fitness-tracker-backend.onrender.com/webhooks/strava
```

### 7.2 Test It!

1. Complete a workout on Strava (or edit an existing one)
2. Within 1-2 minutes, visit your dashboard
3. The workout should appear automatically! 🎉

## Step 8: Verify Everything Works

### Checklist

- ✅ Dashboard loads and shows your workouts
- ✅ Nutrition logging works (AI estimator and manual entry)
- ✅ Settings page shows "Connected to Strava"
- ✅ AI Insights page works
- ✅ Charts display correctly
- ✅ New Strava workouts appear automatically

## Troubleshooting

### "Database connection failed"

- Check that DATABASE_URL is set correctly in Render environment variables
- Verify the PostgreSQL service is running

### "Strava not connected"

- Check STRAVA_CLIENT_ID and STRAVA_CLIENT_SECRET are set
- Verify the callback domain in Strava settings matches your Render URL
- Make sure BASE_URL is set correctly

### "Webhook verification failed"

- Ensure STRAVA_WEBHOOK_VERIFY_TOKEN matches between .env and webhook setup
- Check Render logs for webhook requests

### "No workouts showing"

- Try clicking "Sync Now" in Settings
- Check Render logs for errors during sync
- Verify your Strava account has activities in the last 30 days

### View Logs

In Render dashboard:
1. Go to your service
2. Click "Logs" tab
3. Look for errors or warnings

## Maintenance

### Manual Sync

If webhooks stop working or you want to import older data:

1. Go to Settings
2. Click "🔄 Sync Now"
3. This will import last 30 days of activities

### Re-run Database Migration

If you need to re-import your JSON data:

```bash
python backend/init_db.py
```

### Update Strava Tokens

Tokens refresh automatically! The app handles this in the background.

### Check Webhook Status

```bash
cd backend
python -c "from strava_client import StravaClient; print(StravaClient.list_webhook_subscriptions())"
```

## Local Development Tips

### Use SQLite for Fast Iteration

```bash
# In backend/.env:
DATABASE_URL=sqlite:///./fitness_tracker.db
```

### Test Webhooks Locally with ngrok

1. Install ngrok: https://ngrok.com/download
2. Start your local server: `python health_main.py`
3. In another terminal: `ngrok http 8001`
4. Use the ngrok URL for webhook setup

### Reset Local Database

```bash
rm backend/fitness_tracker.db
python backend/init_db.py
```

## Architecture Notes

### Single-User Mode

The app currently operates in single-user mode:
- All data is associated with `user_id=1`
- Perfect for personal use
- Can be extended to multi-user in the future

### Data Flow

1. **Strava → Webhook → Database**
   - New activity on Strava → Webhook fires → Saved to workouts table

2. **Dashboard Load**
   - Frontend calls `/api/workouts?days=30`
   - Backend queries database
   - Returns workout data in same format as before

3. **Nutrition Logging**
   - Frontend sends data → `/api/nutrition/save`
   - Backend saves to nutrition_logs table
   - Dashboard queries `/api/nutrition/logs`

### Database Schema

```sql
-- Users (one user in single-user mode)
users: id, strava_user_id, access_token, refresh_token, token_expires_at, weight_lbs

-- Workouts (from Strava or CSV uploads)
workouts: id, user_id, strava_activity_id, date, type, duration_min, distance_mi, pace_min_mi, calories_burned, elevation_gain_ft, heart_rate_avg

-- Nutrition logs
nutrition_logs: id, user_id, date, method (ai_estimate/manual/csv), source_description, calories, protein_g, carbs_g, fat_g
```

## Next Steps

- 🎯 Complete a workout on Strava and watch it auto-import
- 📊 Log your nutrition for accurate macro tracking
- 🤖 Ask AI questions about your training
- ⚙️ Adjust your weight in Settings to update nutrition goals

## Support

If you encounter issues:

1. Check Render logs for errors
2. Verify all environment variables are set
3. Test locally first (SQLite + localhost)
4. Check Strava API status: https://status.strava.com

Enjoy your integrated fitness tracker! 🏃‍♂️💪

