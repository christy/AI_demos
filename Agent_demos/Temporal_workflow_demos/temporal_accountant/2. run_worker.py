# run_worker.py
# Client communicates with User's Application and Temporal Worker
# 1. from bash, run server with: 
#    temporal server start-dev --db-filename christy_temporal.db
# 2. from 2nd bash window, activate env
# 3. from bash, run worker with: python run_worker.py
# 4. from 3rd bash window, activate env
# 5. from bash, run worker with: python run_worker.py
# DEMO:
# - copy data folder over
# - remove intermediate Gemini report
# - run workflow, observe Gemini rate limit error
# - observe Temporal retry
# - observe all reports + final report is generated

import asyncio
# Import temporal libraries
from temporalio.client import Client
from temporalio.worker import Worker
from temporalio import workflow
# Import user's application logic
from workflows import AccountantWorkflow
from activities import AccountantActivities

# Add pass throughs here, so sandbox does not reimport libraries every time.
# Standard Python and temporalio import are automatically passed through.
with workflow.unsafe.imports_passed_through():
    from activities import AccountantActivities

# TODO: move this to a shared config file
# Temporal uses task queues to route workflows and activities to worker processes.
ACCOUNTANT_TASK_QUEUE_NAME = "accountant-tasks"

async def main():
    """Starts the Temporal Worker to execute activities and workflows."""
    # Connect to the Temporal server
    client = await Client.connect("localhost:7233", namespace="default")
    print(f"Client connection successful!")
    
    # Initialize the activities class
    activities = AccountantActivities()
    
    # Create a list of activity methods to register with the worker
    activity_methods = [
        activities.read_file_activity,
        activities.generate_prompt_activity,
        activities.analyze_with_model_activity,
        activities.save_results_activity,
        activities.count_tokens_activity,
        activities.save_final_report_activity,
    ]
    
    # Debug: Print all available activity methods
    print("Available activity methods:")
    for method in dir(activities):
        if method.endswith('_activity'):
            print(f"  - {method}")
    
    # Create worker instance
    worker = Worker(
        client,
        task_queue=ACCOUNTANT_TASK_QUEUE_NAME,
        workflows=[AccountantWorkflow],
        activities=activity_methods,
    )
    
    print(f"Worker started and listening on task queue: {ACCOUNTANT_TASK_QUEUE_NAME}")
    print("Ready to process workflow and activity tasks")
    print("Press Ctrl+C to exit")
    
    # Run the worker until it's stopped
    await worker.run()

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())