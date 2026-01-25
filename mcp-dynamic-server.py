#!/usr/bin/env python3
# ~/.local/bin/mcp_sse_listener.py
import json
import sys

import requests

print(
    json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {
                        "connect_sse": {
                            "name": "connect_sse",
                            "description": "Conectar al stream SSE",
                            "inputSchema": {"type": "object", "properties": {}},
                        }
                    }
                },
                "serverInfo": {"name": "SSE Listener", "version": "1.0.0"},
            },
        }
    ),
    flush=True,
)

BASE = "http://localhost:8002"

for line in sys.stdin:
    msg = json.loads(line.strip())
    try:
        if msg.get("method") == "tools/call" and msg["params"]["name"] == "connect_sse":
            # 1. Obtener endpoint único
            resp = requests.post(f"{BASE}/sse", timeout=5)
            sse_url = None

            for resp_line in resp.iter_lines():
                if resp_line:
                    line_str = resp_line.decode("utf-8")
                    if line_str.startswith("data:"):
                        endpoint = line_str[5:].strip()
                        sse_url = f"{BASE}{endpoint}"
                        break

            if sse_url:
                # 2. Conectar y leer primeros mensajes
                response = requests.get(sse_url, stream=True, timeout=30)
                messages = []

                for line_sse in response.iter_lines():
                    if line_sse and len(messages) < 10:  # Primeros 10 mensajes
                        line_str = line_sse.decode("utf-8")
                        if line_str.startswith("data:"):
                            messages.append(line_str[5:].strip())

                result = f"Conectado a: {sse_url}\n\nMensajes recibidos:\n" + "\n".join(
                    messages
                )
            else:
                result = "No se pudo obtener URL SSE"

            print(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "result": {"content": [{"type": "text", "text": result}]},
                        "id": msg.get("id"),
                    }
                ),
                flush=True,
            )

    except Exception as e:
        print(
            json.dumps(
                {"jsonrpc": "2.0", "error": {"message": str(e)}, "id": msg.get("id")}
            ),
            flush=True,
        )
