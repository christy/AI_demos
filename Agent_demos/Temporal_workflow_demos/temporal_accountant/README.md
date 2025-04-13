# Temporal Workflow for a Forensic Accounting AI Agent

A distributed, fault-tolerant application for performing forensic accounting analysis on IRS Form 990 documents using Temporal and multiple LLM providers.

## Overview

This application uses Temporal to orchestrate a forensic accounting workflow that:

1. Reads IRS Form 990 documents
2. Generates prompts for LLM analysis
3. Runs parallel analyses using different LLM providers (Claude, DeepSeek, Gemini)
4. Combines the results into a final comprehensive report
5. Converts the final report to HTML and PDF formats

The application demonstrates how to build resilient, distributed workflows with Temporal, handling retries, error scenarios, and long-running processes.

## Features

- **Distributed Workflow**: Uses Temporal to manage the workflow execution across multiple activities
- **Multiple LLM Integration**: Supports Claude, DeepSeek, and Gemini models
- **Parallel Processing**: Runs LLM analyses in parallel for faster results
- **Document Conversion**: Automatically converts Markdown reports to HTML and PDF
- **Token Counting**: Tracks token usage for prompt optimization
- **Fault Tolerance**: Handles retries and error scenarios gracefully

## Prerequisites

- Python 3.10+
- Temporal server running locally
- API keys for LLM providers (optional, set as environment variables)
  - `ANTHROPIC_API_KEY` for Claude
  - `TOGETHER_API_KEY` for DeepSeek
  - `GOOGLE_API_KEY` for Gemini

## Installation

1. Clone the repository
2. Install the required dependencies:

```bash
pip install temporalio anthropic together google-generativeai tiktoken markdown weasyprint
```

## Running the Application

### 1. Start the Temporal Server

If you don't have the Temporal server running, start it using Docker:

```bash
docker run --rm -p 7233:7233 temporalio/temporal:latest
```

Or use the Temporal CLI:

```bash
temporal server start-dev
```

### 2. Start the Worker

The worker listens for tasks and executes activities:

```bash
python "2. run_worker.py"
```

You should see output indicating that the worker has connected to the Temporal server and is ready to process tasks.

### 3. Run the Workflow

In a separate terminal, run the workflow:

```bash
python "3. run_workflow.py"
```

This will start the forensic accounting workflow, which will:
- Read the input documents
- Generate prompts
- Run LLM analyses
- Generate and save the final report
- Convert the report to HTML and PDF

## Output

The workflow generates several outputs in the `output` directory:
- Individual LLM analysis results as Markdown files
- A final combined report as a Markdown file
- HTML and PDF versions of the final report

## Monitoring

You can monitor workflow execution using the Temporal Web UI at http://localhost:8233

## Customization

- Modify `prompt_template.txt` to change the initial prompt
- Modify `prompt_combine.txt` to change how the final report is generated
- Add or remove LLM models in the workflow.py file

## Troubleshooting

- Check the worker logs for detailed information about activity execution
- Ensure all API keys are properly set as environment variables
- Verify that the Temporal server is running and accessible
