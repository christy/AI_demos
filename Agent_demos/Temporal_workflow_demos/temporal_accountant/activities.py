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
print("Versions installed: ")
print(f"temporalio: {temporalio.__version__}")
print(f"anthropic: {anthropic.__version__}")
print(f"together: {importlib.metadata.version('together')}")
print(f"google-generativeai: {importlib.metadata.version('google-generativeai')}")

# api keys (Consider secure way to manage API keys in production)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# # TODO: move this to a shared config file
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Ensure the output directory exists at startup
# Echo output directory to logging
activity.logger.info(f"Output directory: {OUTPUT_DIR}")

# Model names dictionary
MODEL_NAMES = {
    "claude": "claude-3-5-haiku-20241022",
    "deepseek": "deepseek-ai/DeepSeek-R1",
    "gemini": "gemini-2.0-flash-thinking-exp-01-21",
}
REVERSE_MODEL_NAMES = {v: k for k, v in MODEL_NAMES.items()}  # Map full model name to short key

# Output files dictionaries (using consistent short model names as keys)
OUTPUT_FILES = {
    "claude": os.path.join(OUTPUT_DIR, "claude_results.md"),
    "deepseek": os.path.join(OUTPUT_DIR, "deepseek_results.md"),
    "gemini": os.path.join(OUTPUT_DIR, "gemini_results.md"),
}
FINAL_REPORT_FILES = {
    "claude": os.path.join(OUTPUT_DIR, "claude_final_report.md"),
    "deepseek": os.path.join(OUTPUT_DIR, "deepseek_final_report.md"),
    "gemini": os.path.join(OUTPUT_DIR, "gemini_final_report.md"),
}
FINAL_REPORT_HTML_FILES = {
    "claude": os.path.join(OUTPUT_DIR, "claude_final_report.html"),
    "deepseek": os.path.join(OUTPUT_DIR, "deepseek_final_report.html"),
    "gemini": os.path.join(OUTPUT_DIR, "gemini_final_report.html"),
}
FINAL_REPORT_PDF_FILES = {
    "claude": os.path.join(OUTPUT_DIR, "claude_final_report.pdf"),
    "deepseek": os.path.join(OUTPUT_DIR, "deepseek_final_report.pdf"),
    "gemini": os.path.join(OUTPUT_DIR, "gemini_final_report.pdf"),
}


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

    # Decorator that marks a method as a Temporal activity.
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
            current_dir = os.path.dirname(os.path.abspath(__file__))
            template_path = os.path.join(current_dir, "data", prompt_file)
            prompt_template = await self.read_file_activity(template_path)
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
            if model == MODEL_NAMES["claude"]:
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

            elif model == MODEL_NAMES["deepseek"]:
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

            elif model == MODEL_NAMES["gemini"]:
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
    async def save_results_activity(self, model_key: str, content: str, report_type: str = "analysis") -> None:
        """Activity to save the analysis or final report as Markdown.
        Args:
            model_key: The short model key (e.g., 'claude').
            content: The report content.
            report_type: 'analysis' or 'final'.
        """
        try:
            if report_type == "final":
                filepath = FINAL_REPORT_FILES[model_key]
            else:
                filepath = OUTPUT_FILES[model_key]
            with open(filepath, "w") as f:
                f.write(content)
        except Exception as e:
            import logging
            logging.error(f"Error saving results to {filepath}: {str(e)}")


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

    @activity.defn
    async def save_final_report_activity(self, model_key: str, content_md: str) -> None:
        """Activity to save the final report as Markdown, HTML, and PDF."""
        try:
            # Save Markdown
            md_path = FINAL_REPORT_FILES[model_key]
            with open(md_path, "w") as f:
                f.write(content_md)
            # Convert to HTML
            html_content = markdown.markdown(content_md)
            html_path = FINAL_REPORT_HTML_FILES[model_key]
            with open(html_path, "w") as f:
                f.write(html_content)
            # Convert HTML to PDF
            pdf_path = FINAL_REPORT_PDF_FILES[model_key]
            HTML(string=html_content).write_pdf(pdf_path)
        except Exception as e:
            import logging
            logging.error(f"Error saving final report in all formats for {model_key}: {str(e)}")

    # @activity.defn
    # async def save_final_report_all_formats(self, model_key: str, content_md: str) -> None:
    #     """Save the final report as Markdown, HTML, and PDF for the given model key."""
    #     try:
    #         # 1. Save Markdown
    #         await self.save_results_activity(model_key, content_md, report_type="final")
    #         # 2. Convert to HTML
    #         html_content = markdown.markdown(content_md)
    #         html_path = FINAL_REPORT_HTML_FILES[model_key]
    #         with open(html_path, "w") as f:
    #             await f.write(html_content)
    #         # 3. Convert HTML to PDF
    #         pdf_path = FINAL_REPORT_PDF_FILES[model_key]
    #         await HTML(string=html_content).write_pdf(pdf_path)
    #     except Exception as e:
    #         activity.logger.error(f"Error saving final report in all formats for {model_key}: {str(e)}")
    #         raise e