# echofarm/mcp/client.py
import requests
from .config import MCP_SERVER_URL, REQUEST_TIMEOUT

class MCPClientError(Exception):
    pass

class MCPClient:
    def __init__(self, base_url: str = None, timeout: float = REQUEST_TIMEOUT):
        self.base_url = base_url.rstrip("/") if base_url else MCP_SERVER_URL.rstrip("/")
        self.timeout = timeout

    def list_tools(self):
        url = f"{self.base_url}/tools"
        resp = requests.get(url, timeout=self.timeout)
        if resp.status_code != 200:
            raise MCPClientError(f"failed to list tools: {resp.status_code} {resp.text}")
        return resp.json()

    def invoke_tool(self, tool_name: str, input_payload: dict):
        """
        Invoke a tool/exposed function on the MCP server.

        Request:
            POST {base_url}/invoke
            {
              "tool": "<tool_name>",
              "input": {...}
            }

        Response: should be JSON with the tool output.
        """
        url = f"{self.base_url}/invoke"
        body = {"tool": tool_name, "input": input_payload}
        resp = requests.post(url, json=body, timeout=self.timeout)
        if resp.status_code != 200:
            raise MCPClientError(f"invoke failed: {resp.status_code} {resp.text}")
        return resp.json()
