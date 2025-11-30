from fastapi.testclient import TestClient
from api import app
import sys

client = TestClient(app)

def test_websocket_run():
    print("Connecting to WebSocket...")
    try:
        with client.websocket_connect("/ws/run/titanic?agent_type=mle&max_steps=5") as websocket:
            print("Connected.")
            while True:
                try:
                    data = websocket.receive_text()
                    print(f"Received: {data}")
                    if "Job completed" in data or "Job failed" in data:
                        break
                except Exception as e:
                    print(f"Connection closed or error: {e}")
                    break
    except Exception as e:
        print(f"Failed to connect: {e}")

if __name__ == "__main__":
    test_websocket_run()
