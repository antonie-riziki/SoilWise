# echofarm/mcp/config.py
import os

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")
# expected endpoints used by client:
#  - GET  {MCP_SERVER_URL}/tools          -> lists tools
#  - POST {MCP_SERVER_URL}/invoke         -> invoke a tool with {"tool": name, "input": {...}}

# timeout (seconds) for requests
REQUEST_TIMEOUT = float(os.getenv("MCP_REQUEST_TIMEOUT", "10"))
