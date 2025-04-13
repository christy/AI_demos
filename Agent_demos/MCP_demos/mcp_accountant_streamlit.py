import streamlit as st
import os, time, tiktoken, asyncio
# Anthropic async API doc:  
# https://github.com/anthropics/anthropic-sdk-python/blob/8b244157a7d03766bec645b0e1dc213c6d462165/README.md?plain=1#L382
from anthropic import AsyncAnthropic
from together import Together
from google import genai
from google.genai import types
from typing import Any, Optional, List, Dict

# api keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Constants
CLAUDE_MODEL = "claude-3-5-haiku-20241022"
DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-R1"
# GEMINI_MODEL = "gemini-2.5-pro-exp-03-25" #Returned nothing!
GEMINI_MODEL = "gemini-2.0-flash-thinking-exp-01-21" # works!
OUTPUT_DIR = "output"
CLAUDE_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "claude-3-5-haiku-20241022_results.md")
DEEPSEEK_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "deepseek-ai_results.md")
GEMINI_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "gemini-2.5-pro-exp-03-25_results.md")
CLAUDE_FINAL_REPORT_FILE = os.path.join(OUTPUT_DIR, "claude-3-5-haiku-20241022_final_report.md")
DEEPSEEK_FINAL_REPORT_FILE = os.path.join(OUTPUT_DIR, "deepseek-ai_final_report.md")
GEMINI_FINAL_REPORT_FILE = os.path.join(OUTPUT_DIR, "gemini-2.5-pro-exp-03-25_final_report.md")

## HELPER FUNCTIONS

def read_file(file_path: str) -> str:
    try:
        with open(file_path, "r") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file {file_path}: {str(e)}"

def get_form_990_texts_from_input(num_docs: int, default_texts: Optional[List[str]] = None) -> List[str]:
    """Get Form 990 text content directly from user input via Streamlit text areas.

    Args:
        num_docs (int): The number of Form 990 documents to get.
        default_texts (Optional[List[str]]): Optional list of default texts to pre-fill.

    Returns:
        List[str]: A list of the Form 990 text contents provided by the user.
    """
    try:
        st.header("Enter 990 Text Content")
        st.write(f"Please paste the raw text content from {num_docs} Form 990s from the same organization")

        st.markdown("""
            <style>
            .stTextArea textarea {
                transition: height 0.3s ease-in-out;
            }
            .stTextArea textarea:focus {
                height: 800px !important;
            }
            </style>
        """, unsafe_allow_html=True)

        if 'num_docs' not in st.session_state or st.session_state.num_docs != num_docs:
            st.session_state.num_docs = num_docs
            st.session_state.texts = [None] * num_docs

        for i in range(num_docs):
            with st.expander(f"Form 990 Text Content for Year {i+1}", expanded=True):
                text = st.text_area(
                    label=f"Form 990 content for year {i+1}",
                    key=f"text_{i}",
                    height=400,
                    placeholder="Paste Form 990 content here...",
                    value=default_texts[i] if default_texts and i < len(default_texts) else ""
                )
                if text.strip():
                    st.session_state.texts[i] = text

                if st.button("Upload Content", key=f"upload_{i}"):
                    if text.strip():
                        st.session_state.texts[i] = text
                        st.success(f"✓ Form 990 content for Year {i+1} uploaded successfully!")
                    else:
                        st.error("Please paste content before uploading")

        valid_texts = [t for t in st.session_state.texts if t is not None]
        if valid_texts:
            st.info(f"{len(valid_texts)} of {num_docs} documents uploaded")
        return valid_texts if valid_texts else []
    except Exception as e:
        st.error(f"Error getting Form 990 texts: {str(e)}")
        return []

def generate_prompt_from_template(prompt_file: str, docs: List[str]) -> Optional[str]:
    """Generate analysis prompt from a specified template file.

    Args:
        prompt_file (str): The name of the template file to use.
        docs (List[str]): A list of document contents.

    Returns:
        Optional[str]: The generated prompt, or None if an error occurs.
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        template_path = os.path.join(current_dir, "data", prompt_file)  # Use the prompt_file parameter
        prompt_template = read_file(template_path)
        if not prompt_template:
            return None

        doc_dict = {f'doc{i+1}': doc for i, doc in enumerate(docs)}
        for i in range(len(docs) + 1, 5):
            doc_dict[f'doc{i}'] = ""

        assembled_prompt = prompt_template.format(**doc_dict)
        token_count = count_tokens(assembled_prompt)
        st.info(f"Number of tokens in the prompt: {token_count}")
        st.expander("To debug: view full prompt").text_area(label='', value=assembled_prompt, height=400, disabled=True)
        return assembled_prompt
    except Exception as e:
        st.error(f"Error generating prompt: {str(e)}")
        return None
    
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
        # Ensure the output directory exists, if not create it.
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Save file, either new or replaced.
        with open(output_file, "w") as f:
            f.write(f"# {model} Analysis Results\n\n")
            f.write(content)
    except Exception as e:
        st.error(f"Error saving results to {output_file}: {str(e)}")

def display_report(report_content: str, report_type: str, is_error: bool = False) -> None:
    """Display a report in Streamlit.

    Args:
        report_content (str): The content of the report.
        report_type (str): The type of report (e.g., "Final Report").
        is_error (bool): Whether the report indicates an error.
    """
    if report_content and not is_error:
        st.write(f"{report_type}:")
        st.write(report_content)
        if report_type == "Final Report":
            st.info(f"The final report has been saved.")
    elif is_error:
        st.error(f"{report_type} Generation Error: {report_content if report_content else 'Unknown error'}")


## ASYNC FUNCTIONS

async def run_analysis(prompt: str):
    """Runs analysis using three different models asynchronously."""

    # Add new button in UI
    st.header("Asynchronous Analysis Results")
    tasks = []
    results = []
    report_docs = []

    # Initialize which models to use in analysis step
    models = {
        "claude": "claude-3-5-haiku-20241022",
        "deepseek": "deepseek-ai/DeepSeek-R1",
        # "gemini": "gemini-2.0-flash-thinking-exp-01-21",
    }
    # Initialize clients outside the loop
    # anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
    anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
    together_client = Together(api_key=TOGETHER_API_KEY) if TOGETHER_API_KEY else None
    # google_client = genai.GenerativeModel(GEMINI_MODEL, google_api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None
    google_client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None 
    client_map = {
        "claude": anthropic_client,
        "deepseek": together_client,
        "gemini": google_client,
    }

    # Add all LLM calls to tasks list
    for name, model in models.items():
        client = client_map.get(name)
        if client: # Only create task if client is initialized
            task = asyncio.create_task(
                analyze_with_model(client, prompt, model, name)
            )
            tasks.append(task)
        else:
            st.warning(f"Skipping {name} analysis because API key is missing.")
    st.info(f"{len(tasks)} Tasks created")
    st.info("Waiting for tasks to run asynchronously in parallel...")

    # Parallel asynchronous run Tasks of different LLM inference calls
    if tasks:
        new_results = await asyncio.gather(*tasks)
        results.extend(new_results)

        # Add new results to report_docs
        for result in new_results:
            if result and not result.get("error") and 'content' in result:
                report_docs.append(result['content'])

    st.info(f"Finished! Got {len(results)} LLM results.")

    # Save each individual LLM response in its own output file
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Assemble all LLM responses together for the final report prompt
    for result in results:
        if result:
            name = result.get("model_name", "Unknown") # Try to get model name
            st.subheader(f"{name.capitalize()} Analysis")
            if result.get("error"):
                st.error(f"Error: {result['error']}")
            else:
                duration = result.get("duration", "N/A")
                st.info(f"{name.capitalize()} took {duration:.2f} seconds.")
                content = result.get("content", "")
                # Save the result to a file
                filename = f"{name.replace('/', '_')}_results.md"
                filepath = os.path.join(output_dir, filename)
                try:
                    with open(filepath, "w") as f:
                        f.write(f"# {name.capitalize()} Analysis Result\n\n")
                        f.write(content)
                    st.success(f"✓ {name.capitalize()} analysis result saved to `{filepath}`")
                except Exception as e:
                    st.error(f"Error saving {name.capitalize()} result: {e}")

                # Create a debug collapsed button
                with st.expander(f"To inspect {name.capitalize()} result"):
                    st.text_area(label='', value=content, height=400, disabled=True)

    return report_docs # <--- Return concatted different LLM analysis reports

async def analyze_with_model(client: Any, prompt: str, model: str, model_name: str):
    """Calls the MCP server to run analysis with a specific model."""
    start_time = time.time()
    try:
        # Check client type and adjust call accordingly
        if isinstance(client, AsyncAnthropic):
            response = await client.messages.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=8192
            )
            content = response.content[0].text
            return {"content": content, "duration": time.time() - start_time, "error": None, "model_name": model_name}

        elif isinstance(client, Together):
            # Together AI SDK method is create, even in asynchronous contexts.
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=8192
            )
            content = response.choices[0].message.content
            return {"content": content, "duration": time.time() - start_time, "error": None, "model_name": model_name}

        else:  #Google
            # response = await client.generate_content_async( # Corrected to *_async
            #     contents=[prompt],
            #     generation_config=types.GenerationConfig(
            #         max_output_tokens=8192,
            #         temperature=0.5
            #     )
            # )
            response = await client.aio.models.generate_content(
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


    except Exception as e:
        return {"error": f"Error calling analyze_documents for {model}: {e}", "duration": time.time() - start_time, "content": "", "model_name": model_name}

async def run_final_report(prompt_file, report_docs: List[str], selected_model: str): # <-- Add selected_model
    """Generates the final report using the selected model."""
    final_report_prompt = generate_prompt_from_template(prompt_file, report_docs)
    if not final_report_prompt:
        st.error("Could not generate final report prompt.")
        return {"error": "Could not generate final report prompt.", "content": ""}

    if selected_model == CLAUDE_MODEL:
        client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
        output_file = CLAUDE_FINAL_REPORT_FILE
    elif selected_model == DEEPSEEK_MODEL:
        client = Together(api_key=TOGETHER_API_KEY) if TOGETHER_API_KEY else None
        output_file = DEEPSEEK_FINAL_REPORT_FILE
    elif selected_model == GEMINI_MODEL:
        # client = genai.GenerativeModel(GEMINI_MODEL, google_api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None
        client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None
        output_file = GEMINI_FINAL_REPORT_FILE
    else:
        st.error(f"Unsupported model selected for final report: {selected_model}")
        return {"error": f"Unsupported model selected for final report: {selected_model}", "content": ""}

    if not client:
        st.error(f"API key missing for selected model: {selected_model}")
        return {"error": f"API key missing for selected model: {selected_model}", "content": ""}

    final_report_result = await analyze_with_model(client, final_report_prompt, selected_model, selected_model) # Re-use analyze_with_model
    if not final_report_result.get("error"):
        save_results(output_file, selected_model, final_report_result["content"])
    return final_report_result

if __name__ == "__main__":

    st.title("Form 990 Analysis Tool")

    # Sidebar for Model Selection
    st.sidebar.header("Model Configuration")
    available_models = ["claude-3-5-haiku-20241022", "deepseek-ai/DeepSeek-R1", "gemini-2.0-flash-thinking-exp-01-21"]
    # TODO change back to claude later
    # default_model_index = available_models.index("deepseek-ai/DeepSeek-R1")
    default_model_index = available_models.index("gemini-2.0-flash-thinking-exp-01-21")
    selected_model = st.sidebar.selectbox(
        "Select LLM Model for Final Report:",
        available_models,
        index=default_model_index,
        key="selected_model"
    )

    # Get the number of documents to analyze from the user
    num_docs = st.number_input(
        "How many IRS 990 documents would you like to analyze?",
        min_value=1, max_value=4, value=1
    )

    # Get default doc texts.
    default_texts = [read_file(f"data/test_i990_{year}_pdf.txt")
                     for year in range(2020, 2024)]
    # Get doc texts from user if any, override defaults.
    docs = get_form_990_texts_from_input(num_docs, default_texts)
    if not docs:
        st.warning(f"Please paste content for {num_docs} Form 990(s) to continue")
        st.stop()

    # Use Streamlit's session state to manage user input alert messages.
    if 'first_run' not in st.session_state or not st.session_state.first_run:
        if not docs:
            st.warning(f"Please paste content for {num_docs} Form 990(s) to continue")
    st.session_state.first_run = False

    # User pushes Compose Prompt when done editing docs.
    # For debugging purposes, the prompt is shown before launching workflow.
    if st.button("Compose the Prompt"):
        prompt = generate_prompt_from_template("prompt_template.txt", docs)
        if prompt:
            st.session_state.prompt = prompt

    # Button to run the analysis (only shown after the prompt is composed) 
    if 'prompt' in st.session_state and st.session_state.prompt:
        if st.button("Run Analysis (Async Parallel 3 Models)"):
            report_docs = asyncio.run(run_analysis(st.session_state.prompt)) # Capture report_docs

        # DEBUGGING WITH CLAUDE DESKTOP
        # st.info("Prompt composed. You can now use this prompt in Claude Desktop to call the 'analyze_documents' tool.")
        # st.text_area("Prompt for Claude Desktop (copy and paste into Claude)", value=st.session_state.prompt, height=200, disabled=True)

            # Generate and display final report after async analysis is done
            if report_docs:
                # Assemble the final report prompt
                final_report_result = asyncio.run(run_final_report("prompt_combine.txt", report_docs, selected_model))
                display_report(final_report_result.get("content", ""),
                               "Final Report",
                               final_report_result.get("error") is not None)
            else:
                st.error("No analysis results to generate final report.")
