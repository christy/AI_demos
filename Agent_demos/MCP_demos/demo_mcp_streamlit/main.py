# demo_mcp_streamlit/main.py
import os, time
from typing import Optional, List
import streamlit as st
import asyncio
from utils.mcp_client import MCPClient

# Import the helper functions from the server module
from utils.mcp_server import read_file, count_tokens

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
    
async def run_analysis(mcp_client: MCPClient, prompt: str):
    """Runs analysis using three different models asynchronously."""
    st.header("Asynchronous Analysis Results")
    models = {
        # "claude": "claude-3-5-haiku-20241022",
        # "deepseek": "deepseek-ai/DeepSeek-R1",
        "gemini": "gemini-2.0-flash-thinking-exp-01-21",
    }
    tasks = []
    results = []
    report_docs = []

    for name, model in models.items():
        task = asyncio.create_task(
            analyze_with_model(mcp_client, prompt, model, name)
        )
        tasks.append(task)

    st.info("Tasks created")
    st.info("Waiting for tasks...")

    if tasks:
        new_results = await asyncio.gather(*tasks)
        results.extend(new_results)

        # Add new results to report_docs
        for result in new_results:
            if result and not result.get("error") and 'content' in result:
                report_docs.append(result['content'])

    st.info(f"Finished! Got {len(results)} LLM results.")

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

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

async def analyze_with_model(mcp_client: MCPClient, prompt: str, model: str, model_name: str):
    """Calls the MCP server to run analysis with a specific model."""
    start_time = time.time()
    try:
        # DEBUG to do delete
        st.info(f"MAIN TOOL CALL analyze_documents model: {model}, prompt: {prompt[:100]}")
                
        result = await mcp_client.call_tool("analyze_documents", {"prompt": prompt, "model": model})
        end_time = time.time()
        duration = end_time - start_time
        # DEBUG to do delete
        st.info(f"MAIN Done!! {duration} seconds")
        if isinstance(result, dict) and "error" in result:
            return {"error": result["error"], "duration": duration, "content": ""}
        elif isinstance(result, dict) and "result" in result and "content" in result["result"]:
            return {"content": result["result"]["content"], "duration": duration, "error": None}
        else:
            return {"error": "Unexpected response from analysis server.", "duration": duration, "content": ""}
    except Exception as e:
        return {"error": f"Error calling analyze_documents for {model}: {e}", "duration": time.time() - start_time, "content": ""}


async def main():
    st.title("Form 990 Analysis Tool")

    # Initialize MCP Client
    mcp_client = MCPClient()

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
            st.session_state.prompt = prompt

    # Button to run the analysis (only shown after the prompt is composed)
    if 'prompt' in st.session_state and st.session_state.prompt:
        if st.button("Run Analysis (Async Parallel 3 Models)"):
            await run_analysis(mcp_client, st.session_state.prompt)

if __name__ == "__main__":
    asyncio.run(main())