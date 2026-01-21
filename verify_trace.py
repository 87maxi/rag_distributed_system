import requests
import time
import threading
import sys

session_id = None

def connect_sse():
    global session_id
    try:
        print("Connecting to SSE...")
        with requests.get("http://localhost:8002/sse", stream=True) as r:
            for line in r.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    print(f"SSE Event: {decoded_line}")
                    if "session_id=" in decoded_line:
                        import re
                        match = re.search(r'session_id=([a-zA-Z0-9]+)', decoded_line)
                        if match:
                            session_id = match.group(1)
                            print(f"Captured Session ID: {session_id}")
    except Exception as e:
        print(f"SSE Error: {e}")

def send_message():
    print("Waiting for session ID...")
    for _ in range(10):
        if session_id:
            break
        time.sleep(1)
    
    if not session_id:
        print("Failed to get session ID.")
        return

    print(f"Sending POST request with session_id={session_id}...")
    url = f"http://localhost:8002/messages?session_id={session_id}"
    
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "search_code",
            "arguments": {
                "query": "phoenix tracing"
            }
        },
        "id": 1
    }
    
    try:
        res = requests.post(url, json=payload)
        print(f"POST Status: {res.status_code}")
        print(f"POST Response: {res.text}")
    except Exception as e:
        print(f"POST Error: {e}")

if __name__ == "__main__":
    # Start SSE in a thread
    t = threading.Thread(target=connect_sse, daemon=True)
    t.start()
    
    send_message()
    time.sleep(5)
    print("Done.")
