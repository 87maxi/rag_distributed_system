
import subprocess
import json
import sys

def send_request(proc, request):
    line = json.dumps(request) + "\n"
    proc.stdin.write(line.encode())
    proc.stdin.flush()

def read_response(proc):
    line = proc.stdout.readline().decode()
    if not line:
        return None
    return json.loads(line)

def test_tools():
    print("Starting MCP server via Docker...")
    proc = subprocess.Popen(
        ["docker", "exec", "-i", "rag-mcp", "python3", "-u", "/app/mcp-bridge/stdio_server.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # 1. Initialize
    print("Sending initialize...")
    send_request(proc, {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0"}
        }
    })
    print("Init response:", read_response(proc))

    # 2. List tools
    print("Sending list_tools...")
    send_request(proc, {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {}
    })
    print("List response:", read_response(proc))

    # 3. Call tool
    print("Sending call_tool (get_project_structure)...")
    send_request(proc, {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "get_project_structure",
            "arguments": {"root_path": "."}
        }
    })
    print("Call response:", read_response(proc))

    proc.terminate()

if __name__ == "__main__":
    test_tools()
