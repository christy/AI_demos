# main.py
# This file is responsible for starting the AccountantWorkflow by submitting it to Temporal.
# This is the User's Application: requests workflow code execution and handles results.
# Client communicates with User's Application and Temporal Worker
# 4. from 3rd bash window, activate env
# 5. from bash, run user's workflow with: python run_workflow.py

import asyncio
import time
import traceback
import os
# Import temporal libraries
from temporalio.client import Client, WorkflowFailureError
# Import user's application logic
from shared import AccountantInput  # data classes
from workflows import AccountantWorkflow  # workflow

# TODO: move this to a shared config file
# Temporal uses task queues to route workflows and activities to worker processes.
ACCOUNTANT_TASK_QUEUE_NAME = "accountant-tasks"

async def main():
    """Runs the Forensic Accounting Workflow."""
    print("Starting main.py - will connect to Temporal and execute workflow")

    # Client connection to the Temporal server
    client = await Client.connect("localhost:7233", namespace="default")
    print("Client connection to Temporal server successful!")

    # Instead of Streamlit UI interactions, we are mocking the input data here
    # In a real integration, you would pass the data from a UI to the workflow
    num_docs = 1  # Suggest 1-4 docs
    
    # Get the current directory where main.py is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory does not exist: {data_dir}")
        print(f"Creating data directory...")
        os.makedirs(data_dir, exist_ok=True)
        
    # Check if prompt templates exist
    prompt_template_path = os.path.join(current_dir, "data", "prompt_template.txt")
    prompt_combine_path = os.path.join(current_dir, "data", "prompt_combine.txt")
    
    # Create prompt templates if they don't exist
    if not os.path.exists(prompt_template_path):
        print(f"WARNING: Prompt template file does not exist: {prompt_template_path}")
        print(f"Creating a sample prompt template...")
        with open(prompt_template_path, 'w') as f:
            f.write("You are a forensic accountant analyzing IRS Form 990 documents.\n")
            f.write("Please analyze the following documents and provide insights:\n\n")
            f.write("{doc1}\n\n{doc2}\n\n{doc3}\n\n{doc4}\n\n")
    else:
        print(f"Found prompt template: {prompt_template_path}")
        
    if not os.path.exists(prompt_combine_path):
        print(f"WARNING: Prompt combine file does not exist: {prompt_combine_path}")
        print(f"Creating a sample prompt combine template...")
        with open(prompt_combine_path, 'w') as f:
            f.write("You are a forensic accountant creating a final report.\n")
            f.write("Please combine the following analysis results into a comprehensive report:\n\n")
            f.write("{doc1}\n\n{doc2}\n\n{doc3}\n\n{doc4}\n\n")
    else:
        print(f"Found prompt combine template: {prompt_combine_path}")
    
    # Create absolute paths to the data files
    default_docs_paths = [
        # Use absolute paths to avoid path resolution issues
        os.path.join(current_dir, "data", "test_i990_2020_pdf.txt"),
        # os.path.join(current_dir, "data", "test_i990_2021_pdf.txt"),
    ]
    
    # Check if input files exist
    for path in default_docs_paths:
        if not os.path.exists(path):
            print(f"WARNING: Input file does not exist: {path}")
            print(f"Creating an empty test file for demonstration...")
            # Create a simple test file if it doesn't exist
            with open(path, 'w') as f:
                f.write("This is a sample IRS Form 990 text for testing purposes.\n")
                f.write("It contains financial information for a nonprofit organization.\n")
        else:
            print(f"Found input file: {path}")
    selected_model = "gemini-2.0-flash-thinking-exp-01-21" # Example model

    # Create typesafe data (defined in shared.py)
    accountant_input = AccountantInput(
        num_docs=num_docs,
        default_docs_paths=default_docs_paths,
        selected_model_for_final_report=selected_model
    )
    # --- End Mock User Input ---
    print("Starting workflow...")

    # Generate a unique workflow ID
    workflow_id = f"accountant-workflow-{accountant_input.num_docs}-docs-{int(time.time())}"
    print(f"Generated workflow ID: {workflow_id}")
    
    try:
        print(f"Starting workflow with task queue: {ACCOUNTANT_TASK_QUEUE_NAME}")
        print(f"Starting workflow with ID: {workflow_id}")
        
        # Execute the workflow synchronously.
        # Workflow to execute: AccountantWorkflow.run.
        # Input data: accountant_input.
        # Assign the workflow to the task queue workers: ACCOUNTANT_TASK_QUEUE_NAME.
        # Wait for workflow to actually run! (This is a blocking call)
        workflow_result = await client.execute_workflow(
            AccountantWorkflow.run,  # Reference the workflow's run method
            accountant_input,  # Pass the input data
            task_queue=ACCOUNTANT_TASK_QUEUE_NAME,
            id=workflow_id,          # Unique workflow ID
        )
        print(f"\nWorkflow completed successfully!{workflow_result}")

        # # Execute the workflow asynchronously.
        # # Specify the workflow to execute: AccountantWorkflow.run.
        # # Pass in input data: accountant_input.
        # # Assigns the workflow to the task queue: ACCOUNTANT_TASK_QUEUE_NAME.
        # # Non-blocking call get the future from handle.
        # handle = await client.start_workflow(
        #     AccountantWorkflow.run,  # Reference the workflow's run method
        #     accountant_input,   # Pass the input data
        #     task_queue=ACCOUNTANT_TASK_QUEUE_NAME,
        #     id=workflow_id,           # Unique workflow ID
        # )
        # result = await handle.result()
        # print(f"\nWorkflow completed successfully! {result}")
        
        # Note: The prompts and token counts are now logged in the worker's logs
        # rather than being returned directly in the workflow result
        print("\nPrompt details are available in the worker logs.")
        print("To see the prompts and token counts, check the worker terminal output.")

    except WorkflowFailureError as e:
        print(f"Workflow failed: {e}")
        print(traceback.format_exc())

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())