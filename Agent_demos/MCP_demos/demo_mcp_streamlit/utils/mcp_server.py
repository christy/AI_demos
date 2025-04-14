####
# 1. Open new bash window: source ~/Documents/py312/bin/activate
# 2. cd Documents/github_christy_scratch/agent_demos/demo_mcp_streamlit

# Run MCP server with stdout redirected to stdin of the MCP client (Streamlit).
# - print() and sys.stdout.write() in mcp_server.py will be directed to the stdin of Streamlit
# - sys.stdin.readline() in mcp_client.py will read from the stdout of the MCP server process
# 3. uv run --active utils/mcp_server.py | streamlit run main.py
# This didn't work :(

# Run MCP server with Claude Desktop as MCP client
# 4. vi ~/Library/"Application Support"/Claude/claude_desktop_config.json
# 5. Restart Claude Desktop, verify new tools exist (hammer icon)
# 6. Enter the prompt from data/claude_desktop_test_prompt.txt



# 3. Run this code (MCP server) with command: uv run --active utils/mcp_server.py
# 4. Edit /Users/christy/Library/Application Support/Claude/claude_desktop_config.json
# 5.Run MCP client from Claude Desktop
####
# demo_mcp_streamlit/utils/mcp_server.py
import os, time
import tiktoken
import asyncio
from typing import Optional, List
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
from anthropic import Anthropic
from together import Together
from google import genai
from google.genai import types

# api keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize FastMCP server
# Change logging level: # Set to "DEBUG", "INFO", "WARNING", etc.
mcp = FastMCP("document_analyzer", loggingLevel="WARNING")

# Constants
CLAUDE_MODEL = "claude-3-5-haiku-20241022"
DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-R1"
GEMINI_MODEL = "gemini-2.0-flash-thinking-exp-01-21"
OUTPUT_DIR = "output"
CLAUDE_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "claude-3-5-haiku-20241022_results.md")
DEEPSEEK_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "deepseek-ai_results.md")
GEMINI_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "gemini-2.5-pro-exp-03-25_results.md")
CLAUDE_FINAL_REPORT_FILE = os.path.join(OUTPUT_DIR, "claude-3-5-haiku-20241022_final_report.md")
DEEPSEEK_FINAL_REPORT_FILE = os.path.join(OUTPUT_DIR, "deepseek-ai_final_report.md")
GEMINI_FINAL_REPORT_FILE = os.path.join(OUTPUT_DIR, "gemini-2.5-pro-exp-03-25_results.md")

# Define Data Models for Tools
class GeneratePromptInput(BaseModel):
    prompt_file: str
    docs: List[str]

class GeneratePromptOutput(BaseModel):
    prompt: Optional[str]

class CountTokensInput(BaseModel):
    text: str

class CountTokensOutput(BaseModel):
    token_count: int

class LLMCompletionOutput(BaseModel):
    type: str
    content: str
    error: Optional[str]
    duration: Optional[float]

class AnalyzeDocumentsInput(BaseModel):
    prompt: str
    model: str 

class AnalyzeDocumentsOutput(BaseModel):
    model: str
    result: Optional[LLMCompletionOutput]
    error: Optional[str]
    duration: Optional[float]

# Helper Functions (moved to server)
def read_file(file_path: str) -> str:
    try:
        with open(file_path, "r") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file {file_path}: {str(e)}"

def count_tokens(text: str) -> int:
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        return f"Error counting tokens: {str(e)}"

def save_results(output_file: str, model: str, content: str) -> None:
    """Save the analysis results to a markdown file.

    Args:
        output_file (str): The name of the output file.
        model (str): The name of the model used for analysis.
        content (str): The content to save.
    """
    try:
        # Ensure the output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Save file
        with open(output_file, "w") as f:
            f.write(f"# {model} Analysis Results\n\n")
            f.write(content)
    except Exception as e:
        st.error(f"Error saving results to {output_file}: {str(e)}")

def get_output_file(model: str) -> str:
    if model == CLAUDE_MODEL:
        return CLAUDE_OUTPUT_FILE
    elif model == DEEPSEEK_MODEL:
        return DEEPSEEK_OUTPUT_FILE
    elif model == GEMINI_MODEL:
        return GEMINI_OUTPUT_FILE
    return ""

def get_final_report_file(model_type: str) -> str:
    if model_type == "claude":
        return CLAUDE_FINAL_REPORT_FILE
    elif model_type == "deepseek":
        return DEEPSEEK_FINAL_REPORT_FILE
    elif model_type == "gemini":
        return GEMINI_FINAL_REPORT_FILE
    return ""

# Define Tools
@mcp.tool(name="get_prompt", description="Generates an analysis prompt from a template.")
async def get_prompt(input: GeneratePromptInput) -> GeneratePromptOutput:
    """
    Generates an analysis prompt from a specified template file.

    Args:
        input: GeneratePromptInput containing prompt_file and a list of documents.

    Returns:
        GeneratePromptOutput containing the generated prompt.
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        template_path = os.path.join(current_dir, "..", "data", input.prompt_file)
        prompt_template = read_file(template_path)
        if not prompt_template:
            return GeneratePromptOutput(prompt=None)

        doc_dict = {f'doc{i+1}': doc for i, doc in enumerate(input.docs)}
        for i in range(len(input.docs) + 1, 5):
            doc_dict[f'doc{i}'] = ""

        assembled_prompt = prompt_template.format(**doc_dict)
        token_count = count_tokens(assembled_prompt)
        print(f"Number of tokens in the prompt: {token_count}") # Log to server console
        return GeneratePromptOutput(prompt=assembled_prompt)
    except Exception as e:
        return GeneratePromptOutput(prompt=f"Error generating prompt: {str(e)}")

# @mcp.tool(name="analyze_documents", description="Runs analysis on documents using a specified LLM and saves the result.")
# async def analyze_documents(input: AnalyzeDocumentsInput) -> AnalyzeDocumentsOutput:
#     """
#     Runs analysis using the specified model and saves the result.

#     Args:
#         input: AnalyzeDocumentsInput containing the analysis prompt and the model to use.

#     Returns:
#         AnalyzeDocumentsOutput containing the result of the LLM completion.
#     """
#     prompt = input.prompt
#     selected_model = input.model

#     # DEBUG todo delete leter
#     print()
#     print(f"TOOL CALL analyze_documents model: {selected_model}, prompt: {prompt[:100]}")
#     start_time = time.time()
#     try:
#         anthropic_key = ANTHROPIC_API_KEY
#         together_key = TOGETHER_API_KEY
#         google_key = GOOGLE_API_KEY

#         content = ""
#         if selected_model == CLAUDE_MODEL:
#             # DEBUG: todo delete
#             print("Started Claude API call...")
#             client = Anthropic(api_key=anthropic_key)
#             response = await asyncio.to_thread(
#                 client.messages.create,
#                 model=selected_model,
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0.5,
#                 max_tokens=8192
#             )
#             content = response.content[0].text
#         elif selected_model == DEEPSEEK_MODEL:
#             # DEBUG: todo delete
#             print("Started Deepseek API call...")
#             client = Together(api_key=together_key)
#             response = await asyncio.to_thread(
#                 client.chat.completions.create,
#                 model=selected_model,
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0.5,
#                 max_tokens=8192
#             )
#             content = response.choices[0].message.content
#         else: 
#             selected_model == GEMINI_MODEL
#             # DEBUG: todo delete
#             print("Started Gemini API call...")
#             client = genai.Client(google_key)
#             response = await client.aio.models.generate_content(
#                     model=selected_model, 
#                     contents=[prompt],
#                     config=types.GenerateContentConfig(
#                         max_output_tokens=8192,
#                         temperature=0.5
#                         )
#                     )
#             # Extract text from first candidate response.
#             content = ""
#             if response.candidates:
#                 first_candidate = response.candidates[0]
#                 if first_candidate.content.parts:
#                     first_part = first_candidate.content.parts[0]
#                     if hasattr(first_part, 'text') and first_part.text:
#                         content = first_part.text
#             else:
#                 print("Gemini no response from the first candidate.")

#         duration = time.time() - start_time
#         llm_result = LLMCompletionOutput(type=selected_model.split("-")[0], content=content, error=None, duration=duration)

#         # Save results to output/
#         output_file = os.path.join(OUTPUT_DIR, f"{selected_model.replace('/', '_')}_results.md")
#         save_results(output_file, selected_model, content)
#         return AnalyzeDocumentsOutput(model=selected_model.split("-")[0], result=llm_result, error=None, duration=duration)
#     except Exception as e:
#         return AnalyzeDocumentsOutput(model=selected_model.split("-")[0], result=None, error=str(e), duration=time.time() - start_time)

# DEBUG testing
@mcp.tool(name="analyze_documents", description="Runs analysis on documents using a specified LLM and saves the result.")
async def analyze_documents(input: AnalyzeDocumentsInput) -> AnalyzeDocumentsOutput:
    # DEBUG todo delete
    print("\n--- SERVER TOOL CALL: analyze_documents ---")  # <--- Add this line: Start logging
    print(f"Input received: model={input.model}, prompt={input.prompt[:100]}...") # Add input logging
    
    selected_model = input.model
    start_time = time.time()
    # TEMPORARILY BYPASSING LLM CALL
    content = f"This is a TEST RESULT from analyze_documents tool for model: {selected_model}" # Modified test content
    duration = time.time() - start_time
    llm_result = LLMCompletionOutput(type=selected_model.split("-")[0], content=content, error=None, duration=duration)

    # DEBUG todo delete
    print("--- analyze_documents SERVER TOOL CALL COMPLETED ---") # <--- Add this line: End logging
    
    return AnalyzeDocumentsOutput(model=selected_model.split("-")[0], result=llm_result, error=None, duration=duration)


    # # Assemble the final report prompt
    # final_report_prompt = get_prompt(GeneratePromptInput(prompt_file="prompt_combine.txt", docs=report_docs)).prompt
    # final_report_content = None
    # final_report_error = None

    # if final_report_prompt:
    #     # Invoke the summarizing LLM to generate the final report (using DeepSeek for simplicity)
    #     final_report_result = await run_llm_analysis(final_report_prompt, DEEPSEEK_MODEL)
    #     final_report_content = final_report_result.content
    #     final_report_error = final_report_result.error
    #     if final_report_content:
    #         save_results(DEEPSEEK_FINAL_REPORT_FILE, DEEPSEEK_MODEL.split("/")[1], final_report_content)
    #     elif final_report_error:
    #         print(f"Error generating final report: {final_report_error}")
    # else:
    #     final_report_error = "Could not generate final report prompt."

    # return AnalyzeDocumentsOutput(results=results, final_report_content=final_report_content, final_report_error=final_report_error)

# This works with Claude Desktop
if __name__ == "__main__":
    # Initialize and run the server. Registers tools decorated with @mcp.tool().
    mcp.run(transport='stdio')