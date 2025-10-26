# Strava Integration - Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
cd backend
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Initialize Database
```bash
python backend/init_db.py
```

This will:
- Create database tables
- Migrate your existing JSON data (workouts & nutrition)
- Set up everything for Strava integration

### Step 3: Test Locally
```bash
python backend/health_main.py
```

Visit http://localhost:8001/dashboard.html - your data should load from the database!

---

## 📝 Environment Setup

Create `backend/.env` with:

```bash
# Database (SQLite for local testing)
DATABASE_URL=sqlite:///./fitness_tracker.db
BASE_URL=http://localhost:8001

# Strava (optional for local testing)
STRAVA_CLIENT_ID=your_client_id_here
STRAVA_CLIENT_SECRET=your_client_secret_here
STRAVA_WEBHOOK_VERIFY_TOKEN=any_random_string

# OpenAI (you already have this)
OPENAI_API_KEY=your_openai_key_here
```

---

## 🌐 Deploy to Render.com

### Prerequisites
- GitHub account with your code pushed
- Render.com account (free tier)
- Strava API app created at https://www.strava.com/settings/api

### Quick Deploy

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Add Strava integration"
   git push origin main
   ```

2. **Create on Render**:
   - Go to https://render.com
   - New+ → Blueprint
   - Connect your GitHub repo
   - Click "Apply" (creates database + web service)

3. **Set Environment Variables** (in Render dashboard):
   ```
   BASE_URL=https://your-app-name.onrender.com
   STRAVA_CLIENT_ID=<from Strava settings>
   STRAVA_CLIENT_SECRET=<from Strava settings>
   STRAVA_WEBHOOK_VERIFY_TOKEN=<any random string>
   OPENAI_API_KEY=<your OpenAI key>
   ```

4. **Update Strava App**:
   - Go to https://www.strava.com/settings/api
   - Update your app:
     - Website: `https://your-app-name.onrender.com`
     - Callback Domain: `your-app-name.onrender.com`

5. **Initialize Production Database** (Render Shell):
   ```bash
   cd backend
   python init_db.py
   ```

6. **Connect Strava**:
   - Visit your app's `/settings.html`
   - Click "Connect to Strava"
   - Authorize the app
   - ✅ Your workouts will sync automatically!

7. **Enable Webhooks** (Render Shell):
   ```bash
   cd backend
   python setup_webhook.py https://your-app-name.onrender.com/webhooks/strava
   ```

---

## ✅ What You Get

- **Automatic Workout Import**: New Strava activities appear in your dashboard within 1-2 minutes
- **Historical Sync**: Initial connection imports last 30 days of activities
- **Manual Sync**: Click "Sync Now" in settings anytime
- **Data Persistence**: PostgreSQL database (1GB free on Render)
- **All Existing Features**: Nutrition logging, AI insights, charts - everything still works!

---

## 🔧 Common Commands

### Local Development
```bash
# Start server
cd backend && python health_main.py

# Reset local database
rm fitness_tracker.db && python init_db.py

# Check Strava connection
curl http://localhost:8001/auth/strava/status
```

### Production (Render Shell)
```bash
# Initialize/reset database
cd backend && python init_db.py

# Check webhook subscriptions
cd backend && python -c "from strava_client import StravaClient; print(StravaClient.list_webhook_subscriptions())"

# Re-register webhook
python setup_webhook.py https://your-app-name.onrender.com/webhooks/strava
```

---

## 📊 API Endpoints Added

### Strava OAuth
- `GET /auth/strava/login` - Start OAuth flow
- `GET /auth/strava/callback` - OAuth callback
- `GET /auth/strava/status` - Check connection
- `POST /auth/strava/disconnect` - Disconnect account
- `POST /auth/strava/sync` - Manual sync

### Webhooks
- `GET /webhooks/strava` - Webhook verification
- `POST /webhooks/strava` - Webhook events

### Data
- `GET /api/workouts?days=30` - Get workouts from database
- `GET /api/nutrition/logs?days=7` - Get nutrition logs from database
- `POST /api/nutrition/save` - Save nutrition entry to database

---

## 🐛 Troubleshooting

### "No workouts showing"
- Click "Sync Now" in Settings
- Check Render logs for errors
- Verify Strava connection status

### "Strava connection failed"
- Check STRAVA_CLIENT_ID and STRAVA_CLIENT_SECRET are correct
- Verify callback domain matches in Strava settings
- Ensure BASE_URL is set correctly

### "Database error"
- Run `python backend/init_db.py` to initialize/reset
- Check DATABASE_URL environment variable

### View Logs (Render)
- Go to your service → Logs tab
- Look for 🔴 errors or ⚠️ warnings

---

## 📚 Full Documentation

See `DEPLOYMENT_GUIDE.md` for comprehensive deployment instructions and architecture details.

---

## 🎯 Next Steps After Deployment

1. ✅ Complete a workout on Strava → Watch it auto-import
2. 📊 Log your nutrition for today
3. 🤖 Ask AI about your training patterns
4. ⚙️ Update your weight in Settings to adjust nutrition goals

**Enjoy your integrated fitness tracker!** 🏃‍♂️💪

