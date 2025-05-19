# streamlit_mcp_client_server.py
# Run this with: streamlit run original_mcp_accountant_streamlit.py
# If you have uv installed, run this with:
# uv run --active original_mcp_accountant_streamlit.py

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
# GEMINI_MODEL = "gemini-2.0-flash-thinking-exp-01-21" # works!
GEMINI_MODEL = "gemini-2.5-flash-preview-04-17"
OUTPUT_DIR = "demo_mcp_streamlit/output"
# Update file names to match actual files in output directory
CLAUDE_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "claude_results.md")
DEEPSEEK_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "deepseek_results.md")
GEMINI_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "gemini_results.md")
# Update final report file names to match consistent naming convention
CLAUDE_FINAL_REPORT_FILE = os.path.join(OUTPUT_DIR, "claude_final_report.md")
DEEPSEEK_FINAL_REPORT_FILE = os.path.join(OUTPUT_DIR, "deepseek_final_report.md")
GEMINI_FINAL_REPORT_FILE = os.path.join(OUTPUT_DIR, "gemini_final_report.md")

## HELPER FUNCTIONS

def load_prompt_resources(prompt_dir="demo_mcp_streamlit/data") -> Dict[str, str]:
    """Loads prompt templates from files from the specified directory and returns them as a dictionary."""
    prompt_resources = {}
    current_dir = os.path.dirname(os.path.abspath(__file__))

    template_files = {
        "analysis_prompt": "prompt_template.txt",  # Resource name: filename
        "combine_prompt": "prompt_combine.txt",   # Resource name: filename
    }

    for resource_name, filename in template_files.items():
        template_path = os.path.join(current_dir, prompt_dir, filename) # Use prompt_dir here
        template_content = read_file(template_path)
        if template_content:
            prompt_resources[resource_name] = template_content
        else:
            st.error(f"Failed to load prompt template from {filename} in directory '{prompt_dir}'") # Updated error message
    return prompt_resources

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

def generate_prompt(resource_name: str, docs: List[str]) -> Optional[str]:
    """Generates analysis prompt using a loaded prompt resource."""
    try:
        prompt_template = st.session_state.prompt_resources.get(resource_name) # Get from resources
        if not prompt_template:
            st.error(f"Prompt resource '{resource_name}' not found.")
            return None

        doc_dict = {f'doc{i+1}': doc for i, doc in enumerate(docs)}
        for i in range(len(docs) + 1, 5):
            doc_dict[f'doc{i}'] = ""

        assembled_prompt = prompt_template.format(**doc_dict)
        return assembled_prompt
    except Exception as e:
        st.error(f"Error generating prompt from resource '{resource_name}': {str(e)}")
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

def display_report(report_content: str, report_type: str, is_error: bool = False, is_cached: bool = False) -> None:
    """Display a report in Streamlit.

    Args:
        report_content (str): The content of the report.
        report_type (str): The type of report (e.g., "Final Report").
        is_error (bool): Whether the report indicates an error.
        is_cached (bool): Whether the report was loaded from cache.
    """
    if report_content and not is_error:
        st.subheader(report_type)
        if is_cached:
            st.success(f"{report_type} loaded from cache")
        else:
            st.success(f"{report_type} generated successfully")
        
        with st.expander(f"View {report_type}", expanded=True):
            st.markdown(report_content)
        
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
        "claude": CLAUDE_MODEL,
        "deepseek": DEEPSEEK_MODEL,
        "gemini": GEMINI_MODEL,
    }
    # Initialize clients outside the loop
    # anthropic not async API
    # anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
    # anthropic async API
    anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
    together_client = Together(api_key=TOGETHER_API_KEY) if TOGETHER_API_KEY else None
    google_client = genai.Client() if GOOGLE_API_KEY else None
                        
    client_map = {
        "claude": anthropic_client,
        "deepseek": together_client,
        "gemini": google_client,
    }

    # Map model names to output files
    output_file_map = {
        "claude": CLAUDE_OUTPUT_FILE,
        "deepseek": DEEPSEEK_OUTPUT_FILE,
        "gemini": GEMINI_OUTPUT_FILE,
    }
    
    # Add all LLM calls to tasks list
    for name, model in models.items():
        client = client_map.get(name)
        output_file = output_file_map.get(name)
        
        # Check if output file already exists
        if os.path.exists(output_file):
            st.success(f"Using existing {name} output file: {output_file}")
            content = read_file(output_file)
            if content:
                # Add cached result directly to results
                cached_result = {"content": content, "duration": 0, "error": None, "model_name": name, "cached": True}
                results.append(cached_result)
                # Also add the content to report_docs for final report generation
                report_docs.append(content)
                continue
            else:
                st.warning(f"Existing file {output_file} is empty or couldn't be read. Generating new content.")
        
        if client: # Only create task if client is initialized
            st.info(f"Calling {name} API to generate content...")
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
            is_cached = result.get("cached", False)
            if is_cached:
                st.subheader(f"{name.capitalize()} Analysis (Loaded from Cache)")
            else:
                st.subheader(f"{name.capitalize()} Analysis")
                
            if result.get("error"):
                st.error(f"Error: {result['error']}")
            else:
                duration = result.get("duration", "N/A")
                if is_cached:
                    st.success("✓ Loaded from cache (no API call made)")
                else:
                    st.info(f"{name.capitalize()} took {duration:.2f} seconds.")
                content = result.get("content", "")
                # Save the result to a file using consistent naming
                filename = f"{name}_results.md"  # Simplified naming to match our constants
                filepath = output_file_map.get(name)  # Use the same file path defined in our map
                try:
                    with open(filepath, "w") as f:
                        f.write(f"# {name.capitalize()} Analysis Result\n\n")
                        f.write(content)
                    st.success(f"✓ {name.capitalize()} analysis result saved to `{filepath}`")
                except Exception as e:
                    st.error(f"Error saving {name.capitalize()} result: {e}")

                # Create a debug collapsed button that can be expanded if needed
                with st.expander(f"To inspect {name.capitalize()} result{' (cached)' if is_cached else ''}", expanded=False):
                    st.text_area(label='Analysis Content', value=content, height=400, disabled=True)

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
            # Not async API
            # response = await client.generate_content_async( # Corrected to *_async
            #     contents=[prompt],
            #     generation_config=types.GenerationConfig(
            #         max_output_tokens=8192,
            #         temperature=0.5
            #     )
            # )
            # async API
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
            else:
                print(f"DEBUG: {response.text[:50]}")
            return {"content": content, "duration": time.time() - start_time, "error": None, "model_name": model_name}


    except Exception as e:
        return {"error": f"Error calling analyze_documents for {model}: {e}", "duration": time.time() - start_time, "content": "", "model_name": model_name}

async def run_final_report(prompt_file, report_docs: List[str], selected_model: str): 
    """Generates the final report using the selected model."""
    # Display which model was selected for the final report
    st.info(f"Generating final report using model: {selected_model}")
    
    final_report_prompt = generate_prompt(prompt_file, report_docs)
    if not final_report_prompt:
        st.error("Could not generate final report prompt.")
        return {"error": "Could not generate final report prompt.", "content": ""}
    
    # Display the final report prompt for debugging purposes
    st.subheader("Final Report Prompt (for debugging)")
    token_count = count_tokens(final_report_prompt)
    st.info(f"Final report token count: {token_count}")
    with st.expander("View Final Report Prompt", expanded=False):
        st.markdown(final_report_prompt)

    # Map models to clients and output files
    model_to_client = {
        CLAUDE_MODEL: AsyncAnthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None,
        DEEPSEEK_MODEL: Together(api_key=TOGETHER_API_KEY) if TOGETHER_API_KEY else None,
        GEMINI_MODEL: genai.Client(),
        # GEMINI_MODEL: genai.configure(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None
    }
    
    # Map models to final report files
    model_to_output_file = {
        CLAUDE_MODEL: CLAUDE_FINAL_REPORT_FILE,
        DEEPSEEK_MODEL: DEEPSEEK_FINAL_REPORT_FILE,
        GEMINI_MODEL: GEMINI_FINAL_REPORT_FILE
    }
    
    # Get client and output file for selected model
    client = model_to_client.get(selected_model)
    output_file = model_to_output_file.get(selected_model)
    
    # Check if model is supported
    if not client or not output_file:
        st.error(f"Unsupported model selected for final report: {selected_model}")
        return {"error": f"Unsupported model selected for final report: {selected_model}", "content": ""}

    # This check is now redundant since we already check for client in the previous if statement
    # But keeping a specific error message for missing API keys is helpful
    if not client and selected_model in [CLAUDE_MODEL, DEEPSEEK_MODEL, GEMINI_MODEL]:
        st.error(f"API key missing for selected model: {selected_model}")
        return {"error": f"API key missing for selected model: {selected_model}", "content": ""}
    
    # Check if output file already exists
    if os.path.exists(output_file):
        st.success(f"Using existing output file: {output_file}")
        content = read_file(output_file)
        if content:
            # Don't display the report here, let the caller handle it with display_report
            return {"content": content, "duration": 0, "error": None, "model_name": selected_model, "cached": True}
        else:
            st.warning(f"Existing file {output_file} is empty or couldn't be read. Generating new content.")
    
    # If file doesn't exist or couldn't be read, call the LLM
    st.info(f"Calling {selected_model} API to generate content...")
    final_report_result = await analyze_with_model(client, final_report_prompt, selected_model, selected_model) 
    if not final_report_result.get("error"):
        save_results(output_file, selected_model, final_report_result["content"])
        # Don't display the report here, let the caller handle it with display_report
    return final_report_result

if __name__ == "__main__":

    st.title("Form 990 Analysis Tool")

    # Load prompt resources at the start
    if 'prompt_resources' not in st.session_state:
        st.session_state.prompt_resources = \
            load_prompt_resources(prompt_dir="demo_mcp_streamlit/data")

    # Sidebar for Model Selection
    st.sidebar.header("Model Configuration")
    available_models = [CLAUDE_MODEL, DEEPSEEK_MODEL, GEMINI_MODEL]
    # TODO change back to claude later
    # default_model_index = available_models.index("deepseek-ai/DeepSeek-R1")
    default_model_index = available_models.index(GEMINI_MODEL)
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
    default_texts = [read_file(f"demo_mcp_streamlit/data/test_i990_{year}_pdf.txt")
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
        # Use resource name "analysis_prompt"
        prompt = generate_prompt("analysis_prompt", docs)
        if prompt:
            st.session_state.prompt = prompt
            st.session_state.prompt_token_count = count_tokens(prompt)

    # Always display the initial prompt and token count if present in session_state
    if 'prompt' in st.session_state and st.session_state.prompt:
        st.subheader("Initial Accountant Prompt (for debugging)")
        st.info(f"Initial prompt token count: {st.session_state.get('prompt_token_count', 'N/A')}")
        with st.expander("View Initial Accountant Prompt", expanded=False):
            st.markdown(st.session_state.prompt)


    # Button to run the analysis (only shown after the prompt is composed) 
    if 'prompt' in st.session_state and st.session_state.prompt:

        if st.button("Run Analysis (Async Parallel 3 Models)"):
            # Set session state to track that we're running analysis
            st.session_state.running_analysis = True
            report_docs = asyncio.run(run_analysis(st.session_state.prompt)) # Capture report_docs
            
            # Generate and display final report after async analysis is done
            if report_docs:
                # Show which model was selected for the final report
                st.subheader("Final Report Generation")
                st.info(f"Selected model for final report: {selected_model}")
                
                # Assemble the final report prompt, use resource name "combine_prompt"
                final_report_result = asyncio.run(run_final_report("combine_prompt", report_docs, selected_model))
                if final_report_result.get("error"):
                    st.error(f"Error generating final report: {final_report_result['error']}")
                else:
                    # Display the final report to the user
                    st.subheader("Final Report")
                    if final_report_result.get("cached", False):
                        st.success("Final report loaded from cache")
                    else:
                        st.success("Final report generated successfully")
                    
                    with st.expander("View Final Report", expanded=True):
                        st.markdown(final_report_result.get("content", ""))
                    
                    st.info("The final report has been saved.")