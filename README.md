# AI Health Tracker

A **production-ready multi-agent AI system** for comprehensive health and fitness analysis. This repo demonstrates essential AI engineering patterns that students can study, modify, and adapt for their own health tracking applications.

## What You'll Learn

- 🤖 **Multi-Agent Orchestration**: 4 specialized agents running in parallel using LangGraph
- 🏃 **Real-Time Data Integration**: Strava OAuth with automatic activity sync and webhooks
- 🧠 **AI-Powered Analysis**: Nutrition estimation, readiness scoring, and performance trends
- 📊 **Correlation Detection**: Statistical analysis linking recovery metrics to performance
- 📈 **Production Observability**: Comprehensive tracing with Arize for debugging and evaluation
- 🛠️ **Composable Architecture**: Easily adapt from "health tracker" to your own agent system

**Perfect for:** Students learning to build, evaluate, and deploy agentic AI systems for health and fitness applications.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Health Data                        │
│                    (workouts, recovery, nutrition)               │
└──────────────────────┬─────────────────────────────────────────┘
                       │
            ┌──────────▼──────────┐
            │   FastAPI Endpoint  │
            │   + Session Tracking │
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │   LangGraph Workflow│
            │   (Parallel Execution)│
            └──────────┬──────────┘
                       │
    ┌──────────────────┼──────────────────┐
    │                  │                  │
┌───▼─────┐       ┌────▼──────┐     ┌────▼─────┐
│Activity │       │ Recovery  │     │Nutrition │
│ Agent   │       │  Agent    │     │ Agent    │
└───┬─────┘       └────┬──────┘     └────┬─────┘
    │                  │                  │
    │ Tools:           │ Tools:           │ Tools:
    │ • analyze_perf   │ • calc_readiness │ • estimate_nutrition
    │ • trend_analysis │ • hrv_analysis   │ • macro_analysis
    │                  │                  │
    └──────────────────┼──────────────────┘
                        │
                 ┌──────▼─────┐
                 │  Insights  │
                 │   Agent    │
                 │(Synthesis) │
                 └──────┬─────┘
                        │
            ┌───────────▼───────────┐
            │   Health Insights     │
            │   + Recommendations   │
            └───────────────────────┘

All agents, tools, and LLM calls → Arize Observability Platform
```

## Learning Paths

### 🎓 Beginner Path
1. **Setup & Run** (15 min)
   - Clone repo, configure `.env` with OpenAI key
   - Start server: `cd backend && python health_main.py`
   - Test API: `curl http://localhost:8001/health`

2. **Observe & Understand** (30 min)
   - Make health analysis requests
   - View traces in Arize dashboard
   - Understand agent execution flow and tool calls

3. **Experiment with Prompts** (30 min)
   - Modify agent prompts in `backend/health_main.py`
   - Change tool descriptions
   - See how it affects health insights

### 🚀 Intermediate Path
1. **Connect Real Data** (20 min)
   - Set up Strava OAuth integration
   - Sync your actual workout data
   - Compare AI insights with your real performance

2. **Add Custom Tools** (45 min)
   - Add a new health tool (e.g., `injury_risk_assessment`)
   - Integrate it into an agent
   - Test and trace the new tool calls

3. **Upload Your Data** (30 min)
   - Export data from your fitness apps
   - Upload CSV/Excel files
   - Analyze your personal trends

### 💪 Advanced Path
1. **Change the Domain** (2-3 hours)
   - Use Cursor AI to help transform the system
   - Example: Change from "health tracker" to "nutrition coach"
   - Modify state, agents, and tools for your use case

2. **Add a New Agent** (2 hours)
   - Create a 5th agent (e.g., "injury prevention")
   - Update the LangGraph workflow
   - Test parallel vs sequential execution

3. **Implement Evaluations** (2 hours)
   - Create evaluation criteria for health insights
   - Set up automated evals in Arize
   - Measure accuracy of readiness scores

## Common Use Cases (Built by Students)

Students have successfully adapted this codebase for:

- **🏥 Medical Data Analyzer**
  - Agents: Symptom Tracker, Medication Monitor, Vital Signs Analyzer
  - Replaces fitness tools with medical data processing
  - Used by patients to track health trends

- **🥗 Nutrition Coach**
  - Agents: Meal Planner, Macro Tracker, Recipe Generator
  - RAG over nutrition databases instead of fitness data
  - Generates personalized meal plans

- **🧘 Wellness Monitor**
  - Agents: Sleep Analyzer, Stress Tracker, Mood Correlator
  - Web search for wellness research + RAG over personal logs
  - Generates wellness recommendations

- **🏋️ Strength Training Assistant**
  - Agents: Form Checker, Progression Tracker, Recovery Monitor
  - Tools for exercise analysis and load progression
  - Creates personalized training programs

- **🏃 Running Performance Coach**
  - Agents: Pace Analyzer, Training Load Monitor, Race Predictor
  - RAG over running science literature
  - Reviews training plans for optimization

**💡 Your Turn**: Use Cursor AI to help you adapt this system for your domain!

## Quickstart

1) Requirements
- Python 3.10+ (Docker optional)

2) Configure environment
- Copy `backend/.env.example` to `backend/.env`.
- Set one LLM key: `OPENAI_API_KEY=...` or `OPENROUTER_API_KEY=...`.
- Optional: `ARIZE_SPACE_ID` and `ARIZE_API_KEY` for tracing.

3) Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

4) Run
```bash
# Start the health tracker server
cd backend
python health_main.py
# Server runs on http://localhost:8001
```

5) Open
- Frontend: http://localhost:8001
- Dashboard: http://localhost:8001/dashboard.html
- AI Insights: http://localhost:8001/ai-insights.html
- Upload Data: http://localhost:8001/upload.html
- Settings: http://localhost:8001/settings.html
- API Docs: http://localhost:8001/docs

## Project Structure
- `backend/`: FastAPI app (`health_main.py`), LangGraph agents, tracing hooks.
- `frontend/`: Static HTML pages served by backend.
- `backend/data/`: Sample data and templates.
- `backend/`: Database models, Strava integration, analysis scripts.
- Root: `render.yaml`, `README.md`.

## Development Commands
- Backend (dev): `python health_main.py`
- Test API: `curl http://localhost:8001/health`
- Analyze data: `python backend/analyze_my_data.py`
- Convert Strava: `python backend/convert_strava.py`

## API

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/analyze-health` | POST | Full multi-agent health analysis |
| `/api/nutrition/estimate` | POST | AI nutrition estimation |
| `/api/workouts` | GET | Get recent workouts |
| `/api/nutrition/logs` | GET | Get nutrition entries |
| `/auth/strava/login` | GET | Connect Strava account |
| `/auth/strava/sync` | POST | Sync Strava activities |

### Example: Health Analysis Request

```bash
curl -X POST http://localhost:8001/api/analyze-health \
  -H "Content-Type: application/json" \
  -d '{
    "workouts": [
      {
        "date": "2024-10-15",
        "activity_type": "running",
        "duration_min": 30,
        "distance_km": 5.0,
        "avg_pace_min_per_km": 5.5,
        "avg_hr_bpm": 150
      }
    ],
    "recovery": [
      {
        "date": "2024-10-15",
        "sleep_hours": 7.5,
        "hrv_ms": 62,
        "rhr_bpm": 48
      }
    ],
    "nutrition": [
      {
        "date": "2024-10-15",
        "calories": 2200,
        "protein_g": 120,
        "carbs_g": 280,
        "fat_g": 70,
        "method": "manual"
      }
    ]
  }'
```

### Example: AI Nutrition Estimation

```bash
curl -X POST http://localhost:8001/api/nutrition/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "meal_description": "oatmeal with banana for breakfast, chicken salad for lunch, salmon with rice for dinner"
  }'
```

**Response:**
```json
{
  "calories": 2150,
  "protein_g": 135,
  "carbs_g": 270,
  "fat_g": 55,
  "confidence_score": 0.85,
  "method": "ai_estimate"
}
```

## Key Features

### 1. Multi-Agent Health Analysis
Four specialized agents work in parallel:

- **Activity Agent**: Analyzes workout data and performance trends
- **Recovery Agent**: Assesses sleep, HRV, and calculates readiness scores  
- **Nutrition Agent**: Evaluates diet patterns and energy balance
- **Insights Agent**: Synthesizes all analyses and detects correlations

### 2. Strava Integration
- **OAuth Connection**: Secure login with Strava
- **Automatic Sync**: Import activities automatically
- **Webhook Support**: Real-time updates when you log activities
- **Data Processing**: Converts Strava data to analysis format

### 3. AI Nutrition Estimator
Describe meals naturally:
```
"Oatmeal with banana for breakfast, chicken wrap for lunch"
```

AI estimates:
- Calories
- Protein, Carbs, Fat
- Confidence score

### 4. Readiness Score (1-10)
Combines multiple factors:
- Sleep duration and quality
- Heart Rate Variability (HRV)
- Resting Heart Rate (RHR)
- Recent training load
- Days since rest

### 5. Performance Trend Analysis
Tracks improvement over time:
- Running pace improvements
- Cycling power trends
- Personal records
- Seasonal patterns

### 6. Correlation Detection
Identifies what drives performance:
- "Sleep duration strongly correlates with pace (r=-0.67)"
- "High HRV days = faster runs"
- "Carb intake impact on endurance"

## Data Sources

### Strava Activities (Primary)
- OAuth integration for automatic sync
- Webhook support for real-time updates
- Supports running, cycling, swimming, and more

### CSV/Excel Upload
- Manual data entry
- Bulk import from other fitness apps
- Template provided for easy formatting

### Manual Nutrition Logging
- AI-powered meal descriptions
- Manual macro entry
- Confidence scoring

### Future Integrations
- Apple Watch HealthKit
- Garmin Connect
- Oura Ring
- Whoop

## Frontend Pages

### Dashboard (`dashboard.html`)
- Performance trends with interactive charts
- Today's readiness score
- Recent activity feed
- Weekly summary

### AI Insights (`ai-insights.html`)
- Multi-agent analysis results
- Correlation findings
- Personalized recommendations
- Interactive Q&A

### Upload Data (`upload.html`)
- CSV/Excel file upload
- Data validation and preview
- Template download
- Import progress tracking

### Settings (`settings.html`)
- Strava OAuth connection
- API key configuration
- Data export options
- Account management

## Tools & Agents

### Activity Agent Tools
- `analyze_performance_trends()` - Statistical trend analysis over time
  - Detects improvement/decline
  - Calculates best/worst performances
  - Provides percentage changes

### Recovery Agent Tools
- `calculate_readiness_score()` - Computes 1-10 readiness score
  - Factors: sleep, HRV, RHR, training load, recovery time
  - Returns recommendation (rest, light, moderate, high intensity)

### Nutrition Agent Tools
- `estimate_nutrition_from_description()` - AI-powered nutrition estimation
  - Natural language input
  - Returns calories and macros
  - Confidence scoring

### Insights Agent Tools
- `detect_correlations()` - Statistical correlation analysis
  - Sleep ↔ Performance
  - HRV ↔ Performance
  - Nutrition ↔ Performance
  - Pearson correlation coefficients

## Database Models

### User Model
```python
class User(Base):
    id = Column(Integer, primary_key=True)
    strava_user_id = Column(Integer, unique=True)
    access_token = Column(String(255))
    refresh_token = Column(String(255))
    weight_lbs = Column(Float)
```

### Workout Model
```python
class Workout(Base):
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    strava_activity_id = Column(String(50), unique=True)
    date = Column(Date)
    type = Column(String(50))  # Run, Ride, Swim, etc.
    duration_min = Column(Float)
    distance_mi = Column(Float)
    pace_min_mi = Column(Float)
    calories_burned = Column(Integer)
```

### Nutrition Model
```python
class Nutrition(Base):
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    date = Column(Date)
    method = Column(String(20))  # ai_estimate, manual
    calories = Column(Integer)
    protein_g = Column(Float)
    carbs_g = Column(Float)
    fat_g = Column(Float)
```

## Observability & Tracing

### Arize Integration
- Initialize tracing ONCE at module level (not per request!)
- Use `using_attributes()` for span metadata
- Use `using_prompt_template()` for prompt tracking
- Set span attributes: `agent_type`, `agent_node`, custom fields

### Best Practices
- Wrap agent invocations in observability contexts
- Track tool calls explicitly in state
- Add metadata for session/user tracking
- Use tags for filtering traces by agent/task type

## Environment Variables

### Required (choose one)
- `OPENAI_API_KEY` - OpenAI API key
- `OPENROUTER_API_KEY` + `OPENROUTER_MODEL` - OpenRouter config

### Optional Features
- `ARIZE_SPACE_ID` + `ARIZE_API_KEY` - Observability tracing
- `STRAVA_CLIENT_ID` + `STRAVA_CLIENT_SECRET` - Strava integration
- `TEST_MODE=1` - Use fake LLM for unit tests

## Testing Guidelines

### Manual Testing
```bash
# Health check
curl http://localhost:8001/health

# Nutrition estimation
curl -X POST http://localhost:8001/api/nutrition/estimate \
  -H "Content-Type: application/json" \
  -d '{"meal_description": "2 eggs, toast with avocado, coffee"}'

# Full health analysis
curl -X POST http://localhost:8001/api/analyze-health \
  -H "Content-Type: application/json" \
  -d '{"workouts": [...], "recovery": [...], "nutrition": [...]}'
```

### What to Test
- Multi-agent execution consistency
- Tool fallback behavior (with/without API keys)
- Strava integration (OAuth flow)
- Response time (should be ~6-7 seconds)
- State accumulation across agents

## Common Issues & Solutions

### Duplicate Traces
- Ensure tracing initialized at module level only
- Check LangChainInstrumentor().instrument() called once

### Inconsistent Agent Execution
- Remove any MemorySaver/checkpointer usage
- Verify graph edges are correct
- Check that state is fresh per request

### Strava Not Working
- Verify STRAVA_CLIENT_ID and STRAVA_CLIENT_SECRET set
- Check redirect URI matches Strava app settings
- Ensure webhook URL is publicly accessible

### Slow Responses
- Check API timeout settings (default 30s)
- Verify parallel execution working
- Consider faster LLM model (gpt-3.5-turbo)

## Deploy on Render
- This repo includes `render.yaml`. Connect your GitHub repo in Render and deploy as a Web Service.
- Render will run: `pip install -r backend/requirements.txt` and `python backend/health_main.py`.
- Set `OPENAI_API_KEY` (or `OPENROUTER_API_KEY`) and optional Arize vars in the Render dashboard.

## Next Steps

1. **🎯 Start Simple**: Get it running, make some requests, view traces
2. **🔍 Explore Code**: Read through `backend/health_main.py` to understand patterns
3. **🛠️ Modify Prompts**: Change agent behaviors to see what happens
4. **🚀 Connect Data**: Try Strava integration or upload your own data
5. **💡 Build Your Own**: Use Cursor to transform it into your health agent system

## Troubleshooting

- **401/empty results**: Verify `OPENAI_API_KEY` or `OPENROUTER_API_KEY` in `backend/.env`
- **No traces**: Ensure Arize credentials are set and reachable
- **Port conflicts**: Stop existing services on 8001 or change ports
- **Strava not working**: Check client credentials and redirect URI
- **Slow responses**: Web search APIs may timeout; LLM fallback will handle it

## Learning Resources
- LangGraph: https://langchain-ai.github.io/langgraph/
- OpenInference: https://github.com/Arize-ai/openinference
- Strava API: https://developers.strava.com/
- FastAPI: https://fastapi.tiangolo.com/

## When Helping Users
- Assume they're learning multi-agent patterns
- Explain WHY patterns are used (not just HOW)
- Reference production best practices
- Suggest trade-offs for different approaches
- Help adapt this system to their use case