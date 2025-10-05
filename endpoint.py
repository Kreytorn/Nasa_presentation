from dotenv import load_dotenv
import os
import ngrok
import time

load_dotenv()

listener = ngrok.forward(
	# The port your app is running on.
  8080,
  authtoken=os.getenv("NGROK_AUTHTOKEN"),
  domain=os.getenv("NGROK_DOMAIN"),
	# Secure your endpoint with a traffic policy.
	# This could also be a path to a traffic policy file.
  traffic_policy='{"on_http_request": [{"actions": [{"type": "oauth","config": {"provider": "google"}}]}]}'
)

# Output ngrok url to console
print(f"Ingress established at {listener.url()}")

# Keep the listener alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Closing listener")