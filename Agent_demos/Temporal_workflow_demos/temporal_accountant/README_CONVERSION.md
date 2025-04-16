# Converting a Forensic Accounting AI Agent to Temporal

## The Original Problem

For Nonprofits and public companies, IRS forms are publicly available.  But IRS forms are hard for humans to read and analyze.  The original AI code uses 3 different LLMs to generate analysis and insights, as if you had 3 Junior Forensic Accountants working on the same task.  After they are finished, the code combine the best of their results into a final report.

## "Before": The Streamlit + MCP Approach

Our initial solution consisted of:

1.  **Streamlit Frontend (`utils/mcp_client.py`):** Provided the UI for users to paste Form 990 text, configure analysis, trigger the process, and view results. It used `asyncio.gather` to manage parallel requests to the backend.
2.  **MCP Server (`utils/mcp_server.py`):** A Python script acting as a "tool server" using the Multi-Claude Protocol (MCP) library (`FastMCP`). It exposed tools like `analyze_documents` which would contain the logic to call different LLM APIs.
3.  **LLMs used were:** Claude, DeepSeek, and Gemini

**(Include simple diagram of the "Before" architecture here)**

### Challenges Encountered

1. MCP Client/Server mapping to Temporal Server, Workers, Cluster is not a clear mapping.
2. Removed complexity of MCP Client/Server into Streamlit application.
3. Mapping Streamlit Client to Temporal Server, Workers, Cluster is not a clear mapping.  I think it requires creating a new Activity to handle Streamlit.
4. Due to time constraints, I did not implement the Streamlit Client to Temporal Server, Workers, Cluster mapping.


### Key Temporal Concepts Used

* **Workflow (`workflow.py:DocumentAnalysisWorkflow`):** Defines the *orchestration* logic – the sequence of steps involved in the analysis. This code is durable; Temporal ensures it runs to completion, preserving state even if workers crash.
* **Activity (`activities.py`):** Represents a single unit of work, like calling an LLM API (`analyze_document_activity`) or generating a prompt (`generate_prompt_activity`). Activities are designed to be idempotent and retriable.
* **Worker (`run_worker.py`):** A process that hosts the Workflow and Activity code, polls the Temporal Cluster for tasks, and executes them. Workers can be scaled horizontally.
* **Temporal Cluster:** The backend service (self-hosted or Cloud) that manages workflow state, task queues, timers, and ensures durability and retries.

### The New Architecture

1.  **Client (e.g., Streamlit, CLI - `start_workflow.py`):** Initiates the process by starting the `DocumentAnalysisWorkflow` via the Temporal Client SDK. It receives a workflow handle but doesn't need to manage the execution itself.
2.  **Temporal Cluster:** Persists the workflow state and schedules tasks.
3.  **Temporal Worker(s):**
    * Execute the `DocumentAnalysisWorkflow` logic.
    * When the workflow calls `workflow.execute_activity(...)`, the Cluster schedules the activity task.
    * Workers poll for activity tasks (like `analyze_document_activity`), execute them (calling the LLM APIs), and report results back to the Cluster.

### How Temporal Solved the Challenges

* **State Management:** Workflow state is automatically persisted by the Temporal Cluster. No manual state handling is needed in the application code for orchestration progress.
* **Failure Handling:**
    * **Retries:** Temporal automatically retries failed Activities based on configurable `RetryPolicy` (e.g., exponential backoff for API timeouts).
    * **Durability:** If a Worker crashes, the workflow state is safe in the Cluster. Another Worker can pick up and resume execution exactly where it left off.
* **Scalability:** Workers can be scaled independently to handle increased load. The Cluster manages task distribution. Temporal easily handles long-running tasks without blocking the client.
* **Observability:** Temporal Web UI provides detailed visibility into workflow execution history, current state, inputs/outputs of each step, errors, and retry attempts.

## "After": Migrating to Temporal Workflows

To address these limitations, we migrated the core analysis logic to Temporal. Temporal is an open-source, durable execution system designed for orchestrating long-running, reliable applications.

**(Include simple diagram of the "After" architecture here)**

### Code Structure (`After` Version)

* **`shared.py`:** Dataclasses for workflow inputs/outputs.
* **`activities.py`:** Contains functions decorated with `@activity.defn`. These perform the actual work (e.g., LLM calls, save files).
* **`workflow.py`:** Contains the class decorated with `@workflow.defn`. This orchestrates calls to activities. Note the use of `workflow.execute_activity` and standard Python `asyncio.gather` *within* the workflow definition for parallelism.
* **`run_worker.py`:** Script to start a Temporal Worker process that hosts the workflow and activity code.
* **`start_workflow.py`:** Example script to trigger the workflow execution using the Temporal client.

### Trade-offs

* **User's client/server mappings:**  Extra steps required to map Streamlit and MCP client/server to Temporal server/worker/cluster.
* **Infrastructure:** Requires running the Temporal Cluster and Worker processes.
* **Learning Curve:** Involves understanding Temporal concepts and SDKs.
* **Complexity:** Adds an orchestration layer, which might be overkill for extremely simple tasks but provides immense value for complex, long-running, or critical processes.

## Conclusion

Migrating the Form 990 analysis task from a script-based approach to Temporal significantly improved its reliability, scalability, and observability. By leveraging Temporal's durable execution guarantees and built-in retry mechanisms, we eliminated complex custom logic for state management and failure handling, allowing developers to focus on the core business logic within Activities. While introducing new infrastructure components, the benefits gained in operational robustness make Temporal a powerful choice for orchestrating complex, multi-step processes like AI agentic workflows.

## Running the Temporal Version (Example)

1.  **Setup Temporal:** Start a local Temporal development cluster (e.g., using `temporal server start-dev`) or connect to Temporal Cloud.
2.  **Install Dependencies:** `pip install temporalio pydantic requests anthropic together google-generativeai ...`
3.  **Set Environment Variables:** Export `ANTHROPIC_API_KEY`, `TOGETHER_API_KEY`, `GOOGLE_API_KEY`.
4. **Start Server:** temporal server start-dev
5.  **Run Worker:** `python run_worker.py` (Keep this running in one terminal)
6.  **Start Workflow:** `python start_workflow.py` (Run in another terminal to trigger an execution)
7.  **Observe:** Monitor progress via the Worker logs and the Temporal Web UI (usually `http://localhost:8233`).
