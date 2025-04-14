# main.py
import streamlit as st
import asyncio
from typing import Callable
from ui import sidebar, chat_ui
from utils import agent, mcp_client
from mcp.server import fastmcp  # Import from the new location
from config import PROMPT_TEMPLATE_FILE, PROMPT_COMBINE_FILE

async def main():
    st.set_page_config(layout="wide")
    st.title("Form 990 Analysis Tool")

    client, selected_model = sidebar.sidebar()

    async with mcp_client.MCPClient(fastmcp.server_params()) as mcp_client_instance: # Use the imported module
        mcp_tools = await mcp_client_instance.get_available_tools()
        tools = {
            tool.name: {
                "name": tool.name,
                "callable": await mcp_client_instance.call_tool(tool.name),
                "schema": {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    },
                },
            }
            for tool in mcp_tools
        }
        st.session_state.tools = tools

        await chat_ui.chat_ui(client, tools, selected_model)

if __name__ == "__main__":
    asyncio.run(main())