# run_worker.py
import asyncio

# Import temporal libraries
from temporalio.client import Client
from temporalio.worker import Worker

# Import user's application logic
from workflows import AccountantWorkflow
from activities import AccountantActivities

# Define the task queue name
ACCOUNTANT_TASK_QUEUE_NAME = "accountant-task-queue"

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
        activities.convert_markdown_to_pdf_activity,
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