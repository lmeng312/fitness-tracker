"""
Strava Webhook Setup Script
Run this once after deployment to register webhook subscription
"""

import sys
from strava_client import StravaClient

def main():
    if len(sys.argv) < 2:
        print("Usage: python setup_webhook.py <callback_url>")
        print("Example: python setup_webhook.py https://your-app.onrender.com/webhooks/strava")
        sys.exit(1)
    
    callback_url = sys.argv[1]
    
    print(f"🔗 Subscribing to Strava webhooks with callback: {callback_url}")
    
    try:
        # Check existing subscriptions
        existing = StravaClient.list_webhook_subscriptions()
        if existing:
            print(f"⚠️  Found {len(existing)} existing subscription(s):")
            for sub in existing:
                print(f"  - ID: {sub['id']}, Callback: {sub['callback_url']}")
            
            response = input("Delete existing subscriptions? (y/n): ")
            if response.lower() == "y":
                for sub in existing:
                    StravaClient.delete_webhook_subscription(sub["id"])
        
        # Create new subscription
        result = StravaClient.subscribe_to_webhooks(callback_url)
        print(f"✅ Webhook subscription created successfully!")
        print(f"   Subscription ID: {result['id']}")
        print(f"   Callback URL: {callback_url}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

