# activities.py
import asyncio
import os
import time
import tiktoken
from typing import Any, Dict, Optional, List
import markdown
from weasyprint import HTML

from temporalio import activity
from anthropic import AsyncAnthropic, Anthropic  # Import both async and sync Anthropic
from together import Together
from google import genai
from google.genai import types

# DEBUG INFO: todo delete
import importlib.metadata
import temporalio, anthropic
print(f"temporalio: {temporalio.__version__}")
print(f"anthropic: {anthropic.__version__}")
print(f"together: {importlib.metadata.version('together')}")
print(f"google-generativeai: {importlib.metadata.version('google-generativeai')}")

# api keys (Consider secure way to manage API keys in production)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# TODO: move this to a shared config file
# Constants
CLAUDE_MODEL = "claude-3-5-haiku-20241022"
DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-R1"
GEMINI_MODEL = "gemini-2.0-flash-thinking-exp-01-21"
# Make output directory absolute path relative to current file
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
# Echo output directory to logging
activity.logger.info(f"Output directory: {OUTPUT_DIR}")

# Activities should handle their own exceptions. 
# Re-raising exceptions in activities will signal to Temporal that the 
# activity failed, and Temporal will handle retries based on the RetryPolicy 
# configured on the activity stub in the workflow.

class AccountantActivities:
    """Activities for the Forensic Accounting Workflow, performing specific tasks."""

    # Initializes the LLM clients in the activity class constructor. 
    # This is done once per worker process when the worker starts up, 
    # so clients are reused for multiple activity executions.
    def __init__(self) -> None:
        """Initializes Activities, setting up clients for LLM interactions."""
        activity.logger.info(f"Initializing AccountantActivities with output directory: {OUTPUT_DIR}")
        # Initialize clients here if needed for all activities or per activity method
        self.anthropic_client_async = AsyncAnthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
        self.anthropic_client_sync = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None # For final report in example
        self.together_client = Together(api_key=TOGETHER_API_KEY) if TOGETHER_API_KEY else None
        self.google_client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None

    # Decorator that marks each method as a Temporal activity.
    @activity.defn
    async def read_file_activity(self, file_path: str) -> str:
        """Activity to read the content of a file.

        Args:
            file_path: The path to the file to read.

        Returns:
            The content of the file as a string.
        """
        try:
            with open(file_path, "r") as f:
                return f.read()
        except Exception as e:
            # Activities use activity.logger for logging, which is separate from workflow logging. 
            # Activity logs are also valuable for debugging and monitoring.
            activity.logger.exception(f"Error reading file {file_path}")
            raise e  # Re-raise exception for Temporal to handle

    @activity.defn
    async def generate_prompt_activity(self, prompt_file: str, docs: List[str]) -> Optional[str]:
        """Activity to generate an accounting prompt from a template file and document contents.

        Args:
            prompt_file: The name of the prompt template file.
            docs: A list of document contents to be included in the prompt.

        Returns:
            The generated prompt string, or None if prompt generation fails.
        """
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__)) # Important: activities context
            template_path = os.path.join(current_dir, "data", prompt_file) # Assuming 'data' dir is relative to activities.py
            prompt_template = await asyncio.to_thread(self.read_file, template_path) # Use activity's read_file or below

            if not prompt_template:
                return None

            doc_dict = {f'doc{i+1}': doc for i, doc in enumerate(docs)}
            for i in range(len(docs) + 1, 5):
                doc_dict[f'doc{i}'] = ""

            assembled_prompt = prompt_template.format(**doc_dict)
            token_count = await self.count_tokens_activity(assembled_prompt)
            activity.logger.info(f"Generated prompt with {token_count} tokens.")
            return assembled_prompt
        except Exception as e:
            activity.logger.exception("Error generating prompt")
            raise e

    def read_file(self, file_path: str) -> str: # Non-activity helper function
        """Helper function to read file content (used by activity internally)."""
        try:
            with open(file_path, "r") as f:
                return f.read()
        except Exception as e:
            activity.logger.error(f"Error in helper read_file: {e}")
            raise

    @activity.defn
    async def count_tokens_activity(self, text: str) -> int:
        """Activity to count the number of tokens in a given text.

        Args:
            text: The text string to count tokens for.

        Returns:
            The number of tokens in the text.
        """
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception as e:
            activity.logger.exception("Error counting tokens")
            raise e

    @activity.defn
    async def save_results_activity(self, output_file: str, model: str, content: str) -> None:
        """Activity to save analysis results to a markdown file.

        Args:
            output_file: The full path to the output file.
            model: The name of the model used for analysis.
            content: The analysis result content to save.
        """
        try:
            activity.logger.info(f"Creating output directory at: {OUTPUT_DIR}")
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            activity.logger.info(f"Output directory exists or was created successfully")
            with open(output_file, "w") as f:
                f.write(f"# {model.capitalize()} Analysis Results\n\n")
                f.write(content)
            activity.logger.info(f"Results saved to {output_file}")
        except Exception as e:
            activity.logger.exception(f"Error saving results to {output_file}")
            raise e

    # Checks the model parameter and calls the appropriate LLM client (Anthropic, Together, Gemini).
    # Handles model-specific API calls and response parsing.
    # Includes error handling and returns a dictionary containing the analysis content, duration, error (if any), and model_name.
    # Uses the initialized clients from the __init__ method.
    @activity.defn
    async def analyze_with_model_activity(self, prompt: str, model: str, model_name: str) -> Dict[str, Any]:
        """Activity to analyze a prompt using a specified LLM model.

        Args:
            prompt: The prompt string to send to the LLM.
            model: The model identifier string (e.g., "claude-3-5-haiku-20241022").
            model_name: A friendly name for the model (e.g., "claude") for logging and output.

        Returns:
            A dictionary containing the analysis result, duration, error (if any), and model name.
        """
        start_time = time.time()
        try:
            activity.logger.info(f"Analyzing with model: {model} (name: {model_name})")
            if model == CLAUDE_MODEL:
                if self.anthropic_client_async:
                    response = await self.anthropic_client_async.messages.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.5,
                        max_tokens=8192
                    )
                    content = response.content[0].text
                    return {"content": content, "duration": time.time() - start_time, "error": None, "model_name": model_name}
                else:
                    error_msg = "Anthropic client not initialized (ANTHROPIC_API_KEY missing)"
                    activity.logger.error(error_msg)
                    raise Exception(error_msg)

            elif model == DEEPSEEK_MODEL:
                if self.together_client:
                    response = await asyncio.to_thread(
                        self.together_client.chat.completions.create,
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.5,
                        max_tokens=8192
                    )
                    content = response.choices[0].message.content
                    return {"content": content, "duration": time.time() - start_time, "error": None, "model_name": model_name}
                else:
                    error_msg = "Together client not initialized (TOGETHER_API_KEY missing)"
                    activity.logger.error(error_msg)
                    raise Exception(error_msg)

            elif model == GEMINI_MODEL:
                if self.google_client:
                    response = await self.google_client.aio.models.generate_content(
                        model=model,
                        contents=[prompt],
                        config=types.GenerateContentConfig(
                            max_output_tokens=8192,
                            temperature=0.5
                        )
                    )
                    content = ""
                    if response.candidates:
                        first_candidate = response.candidates[0]
                        if first_candidate.content.parts:
                            first_part = first_candidate.content.parts[0]
                            if hasattr(first_part, 'text') and first_part.text:
                                content = first_part.text
                    return {"content": content, "duration": time.time() - start_time, "error": None, "model_name": model_name}
                else:
                    raise Exception("Google client not initialized (API key missing?)")
            else:
                raise ValueError(f"Unsupported model: {model}")

        except Exception as e:
            # Activities use activity.logger for logging, which is separate from workflow logging. 
            # Activity logs are also valuable for debugging and monitoring.
            activity.logger.exception(f"Error analyzing with model {model_name}")
            return {"error": f"Error calling analyze_documents for {model_name}: {e}", "duration": time.time() - start_time, "content": "", "model_name": model_name}

    @activity.defn
    async def convert_markdown_to_pdf_activity(self, markdown_file_path: str) -> Dict[str, str]:
        """Activity to convert a Markdown file to HTML and then to PDF.
        
        Args:
            markdown_file_path: The path to the Markdown file to convert.
            
        Returns:
            A dictionary containing the paths to the generated HTML and PDF files.
        """
        activity.logger.info(f"Converting {markdown_file_path} to HTML and PDF")
        
        try:
            # Make sure the output directory exists
            os.makedirs(os.path.dirname(markdown_file_path), exist_ok=True)
            
            # Read the markdown file
            with open(markdown_file_path, 'r') as f:
                markdown_content = f.read()
            
            # Convert markdown to HTML
            html_content = markdown.markdown(
                markdown_content,
                extensions=['tables', 'fenced_code', 'codehilite']
            )
            
            # Add some basic styling
            styled_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>{os.path.basename(markdown_file_path)}</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        line-height: 1.6;
                        margin: 40px;
                        max-width: 900px;
                    }}
                    h1, h2, h3 {{
                        color: #333;
                    }}
                    table {{
                        border-collapse: collapse;
                        width: 100%;
                        margin: 20px 0;
                    }}
                    th, td {{
                        border: 1px solid #ddd;
                        padding: 8px;
                    }}
                    th {{
                        background-color: #f2f2f2;
                    }}
                    code {{
                        background-color: #f5f5f5;
                        padding: 2px 5px;
                        border-radius: 3px;
                    }}
                    pre {{
                        background-color: #f5f5f5;
                        padding: 10px;
                        border-radius: 5px;
                        overflow-x: auto;
                    }}
                </style>
            </head>
            <body>
                {html_content}
            </body>
            </html>
            """
            
            # Save HTML file
            html_file_path = markdown_file_path.replace('.md', '.html')
            with open(html_file_path, 'w') as f:
                f.write(styled_html)
            
            # Convert HTML to PDF
            pdf_file_path = markdown_file_path.replace('.md', '.pdf')
            HTML(string=styled_html).write_pdf(pdf_file_path)
            
            activity.logger.info(f"Successfully converted {markdown_file_path} to HTML and PDF")
            return {
                "html_path": html_file_path,
                "pdf_path": pdf_file_path
            }
            
        except Exception as e:
            activity.logger.error(f"Error converting {markdown_file_path} to HTML and PDF: {e}")
            raise