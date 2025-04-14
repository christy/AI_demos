# demo_mcp_streamlit/utils/mcp_client.py
import json
import sys
import asyncio

class MCPClient:
    def __init__(self):
        pass

    async def call_tool(self, tool_name: str, input_data: dict):
        request = {"tool_call": {"tool_name": tool_name, "input": input_data}}
        json.dump(request, sys.stdout)
        sys.stdout.write('\n')
        sys.stdout.flush()

        response_line = await asyncio.to_thread(sys.stdin.readline)
        if response_line:
            try:
                response = json.loads(response_line)
                if "tool_code_result" in response:
                    return response["tool_code_result"].get("content")
                elif "error" in response:
                    return {"error": response["error"]}
                else:
                    return {"error": "Unexpected response format from MCP server."}
            except json.JSONDecodeError:
                return {"error": f"Could not decode JSON response: {response_line}"}
        else:
            return {"error": "No response from MCP server."}

    async def close(self):
        pass # No explicit closing needed for stdio