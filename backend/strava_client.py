"""
Strava API Client
Handles OAuth, activity fetching, token refresh, and webhook management
"""

import os
import requests
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from models import User, Workout

STRAVA_CLIENT_ID = os.getenv("STRAVA_CLIENT_ID")
STRAVA_CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET")
STRAVA_WEBHOOK_VERIFY_TOKEN = os.getenv("STRAVA_WEBHOOK_VERIFY_TOKEN", "default_verify_token")

STRAVA_AUTH_URL = "https://www.strava.com/oauth/authorize"
STRAVA_TOKEN_URL = "https://www.strava.com/oauth/token"
STRAVA_API_BASE = "https://www.strava.com/api/v3"


class StravaClient:
    """Client for interacting with Strava API"""
    
    @staticmethod
    def get_authorization_url(redirect_uri: str, state: str = "") -> str:
        """
        Generate Strava OAuth authorization URL
        
        Args:
            redirect_uri: Callback URL after authorization
            state: Optional state parameter for CSRF protection
            
        Returns:
            Authorization URL to redirect user to
        """
        params = {
            "client_id": STRAVA_CLIENT_ID,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": "activity:read_all",  # Read all activity data
            "state": state
        }
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        return f"{STRAVA_AUTH_URL}?{query_string}"
    
    @staticmethod
    def exchange_code_for_tokens(code: str) -> Dict[str, Any]:
        """
        Exchange authorization code for access and refresh tokens
        
        Args:
            code: Authorization code from OAuth callback
            
        Returns:
            Dict with access_token, refresh_token, expires_at, athlete info
        """
        response = requests.post(
            STRAVA_TOKEN_URL,
            data={
                "client_id": STRAVA_CLIENT_ID,
                "client_secret": STRAVA_CLIENT_SECRET,
                "code": code,
                "grant_type": "authorization_code"
            }
        )
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def refresh_access_token(refresh_token: str) -> Dict[str, Any]:
        """
        Refresh access token using refresh token
        
        Args:
            refresh_token: User's refresh token
            
        Returns:
            Dict with new access_token, refresh_token, expires_at
        """
        response = requests.post(
            STRAVA_TOKEN_URL,
            data={
                "client_id": STRAVA_CLIENT_ID,
                "client_secret": STRAVA_CLIENT_SECRET,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token"
            }
        )
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def ensure_fresh_token(user: User, db: Session) -> str:
        """
        Ensure user has a valid access token, refresh if needed
        
        Args:
            user: User object from database
            db: Database session
            
        Returns:
            Valid access token
        """
        # Check if token is expired (with 5 min buffer)
        now = int(datetime.utcnow().timestamp())
        if user.token_expires_at and user.token_expires_at < (now + 300):
            # Token expired or expiring soon, refresh it
            token_data = StravaClient.refresh_access_token(user.refresh_token)
            user.access_token = token_data["access_token"]
            user.refresh_token = token_data["refresh_token"]
            user.token_expires_at = token_data["expires_at"]
            db.commit()
            print(f"🔄 Refreshed access token for user {user.id}")
        
        return user.access_token
    
    @staticmethod
    def get_athlete_activities(
        access_token: str,
        after: Optional[int] = None,
        per_page: int = 30,
        page: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Fetch athlete activities from Strava
        
        Args:
            access_token: Valid Strava access token
            after: Unix timestamp to fetch activities after
            per_page: Number of activities per page (max 200)
            page: Page number
            
        Returns:
            List of activity summaries
        """
        headers = {"Authorization": f"Bearer {access_token}"}
        params = {"per_page": per_page, "page": page}
        if after:
            params["after"] = after
        
        response = requests.get(
            f"{STRAVA_API_BASE}/athlete/activities",
            headers=headers,
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def get_activity_details(access_token: str, activity_id: int) -> Dict[str, Any]:
        """
        Fetch detailed activity data from Strava
        
        Args:
            access_token: Valid Strava access token
            activity_id: Strava activity ID
            
        Returns:
            Detailed activity data
        """
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get(
            f"{STRAVA_API_BASE}/activities/{activity_id}",
            headers=headers
        )
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def activity_to_workout(activity: Dict[str, Any], user_id: int) -> Workout:
        """
        Convert Strava activity data to Workout model
        
        Args:
            activity: Strava activity dict
            user_id: User ID to associate workout with
            
        Returns:
            Workout model instance
        """
        # Parse date from ISO format
        start_date = datetime.fromisoformat(activity["start_date"].replace("Z", "+00:00"))
        
        # Calculate pace if distance and duration available
        distance_mi = activity.get("distance", 0) / 1609.34  # meters to miles
        duration_min = activity.get("moving_time", 0) / 60  # seconds to minutes
        pace_min_mi = duration_min / distance_mi if distance_mi > 0 else None
        
        # Convert elevation to feet
        elevation_ft = activity.get("total_elevation_gain", 0) * 3.28084  # meters to feet
        
        return Workout(
            user_id=user_id,
            strava_activity_id=str(activity["id"]),
            date=start_date.date(),
            type=activity.get("type", "Run"),
            duration_min=duration_min if duration_min > 0 else None,
            distance_mi=distance_mi if distance_mi > 0 else None,
            pace_min_mi=pace_min_mi,
            calories_burned=activity.get("calories"),
            elevation_gain_ft=elevation_ft if elevation_ft > 0 else None,
            heart_rate_avg=activity.get("average_heartrate")
        )
    
    @staticmethod
    def sync_recent_activities(user: User, db: Session, days: int = 30) -> int:
        """
        Sync recent activities from Strava
        
        Args:
            user: User object
            db: Database session
            days: Number of days to sync (default 30)
            
        Returns:
            Number of new activities imported
        """
        # Get fresh token
        access_token = StravaClient.ensure_fresh_token(user, db)
        
        # Calculate after timestamp (30 days ago)
        after_timestamp = int((datetime.utcnow() - timedelta(days=days)).timestamp())
        
        # Fetch activities
        activities = StravaClient.get_athlete_activities(
            access_token,
            after=after_timestamp,
            per_page=100
        )
        
        new_count = 0
        for activity in activities:
            # Check if activity already exists
            existing = db.query(Workout).filter(
                Workout.strava_activity_id == str(activity["id"])
            ).first()
            
            if not existing:
                # Create new workout
                workout = StravaClient.activity_to_workout(activity, user.id)
                db.add(workout)
                new_count += 1
        
        db.commit()
        print(f"✅ Synced {new_count} new activities from Strava")
        return new_count
    
    @staticmethod
    def subscribe_to_webhooks(callback_url: str) -> Dict[str, Any]:
        """
        Subscribe to Strava webhooks
        
        Args:
            callback_url: Full URL for webhook endpoint
            
        Returns:
            Subscription data including subscription_id
        """
        response = requests.post(
            f"{STRAVA_API_BASE}/push_subscriptions",
            data={
                "client_id": STRAVA_CLIENT_ID,
                "client_secret": STRAVA_CLIENT_SECRET,
                "callback_url": callback_url,
                "verify_token": STRAVA_WEBHOOK_VERIFY_TOKEN
            }
        )
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def list_webhook_subscriptions() -> List[Dict[str, Any]]:
        """List all active webhook subscriptions"""
        response = requests.get(
            f"{STRAVA_API_BASE}/push_subscriptions",
            params={
                "client_id": STRAVA_CLIENT_ID,
                "client_secret": STRAVA_CLIENT_SECRET
            }
        )
        response.raise_for_status()
        return response.json()
    
    @staticmethod
    def delete_webhook_subscription(subscription_id: int):
        """Delete a webhook subscription"""
        response = requests.delete(
            f"{STRAVA_API_BASE}/push_subscriptions/{subscription_id}",
            params={
                "client_id": STRAVA_CLIENT_ID,
                "client_secret": STRAVA_CLIENT_SECRET
            }
        )
        response.raise_for_status()
        print(f"✅ Deleted webhook subscription {subscription_id}")

