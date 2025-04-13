# workflows.py
# AccountantWorkflow orchestrates the entire analysis process. 
# It defines the steps including:
# - Receiving user inputs (number of documents, texts).
# - Calling activities to generate prompts, run LLM analyses, generate the final report, and save results.
# - Handling retries and error scenarios (although basic in this example).

import asyncio
from typing import List, Dict, Any, Optional
from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError

# Import data classes for the workflow
from shared import AccountantInput  # data classes

# Import activities interface for type hinting only
with workflow.unsafe.imports_passed_through():
    from activities import AccountantActivities  # Import your activities class for type hinting

# Decorator marks AccountantWorkflow as a Temporal workflow.
@workflow.defn
class AccountantWorkflow:
    """Workflow for performing forensic accounting on IRS Form 990 documents."""

    # Required to implement async callable run() w/await inside.
    # Best Practice:  Pass a single data class object into a Workflow as input.
    # Makes it easier for you to modify long-running Workflows in the future.
    @workflow.run
    async def run(self, accountant_input: AccountantInput) -> str:
        """
        Executes the forensic accounting workflow.

        Args:
            accountant_input: Input parameters for the accountant: 
            - document paths for text pdfs of irs forms
            - selected LLM to generate the final report

        Returns:
            A summary string:
             - workflow completion status
             - where to find final report
        """

        # Allow certain imports to pass through
        with workflow.unsafe.imports_passed_through():
            import os, tiktoken  # Import tiktoken here to allow it in the workflow

        # Required define workflow retry policy for activities (you can customize).
        # My retries will most likely be due to LLM rate limits.
        retry_policy = RetryPolicy(
            maximum_attempts=2,
            maximum_interval=timedelta(seconds=65),
            # Add specific non-retryable errors if needed
            non_retryable_error_types=[], 
        )

        # No need to create a separate activity handle in this SDK version
        # The execute_activity function will be used directly

        # Log workflow start with input details
        workflow.logger.info(f"Starting Accountant Workflow with {len(accountant_input.default_docs_paths)} documents")
        workflow.logger.info(f"Selected model for final report: {accountant_input.selected_model_for_final_report}")

        # ORCHESTRATE STEPS 1-7 OF ACCOUNTING WORKFLOW
        # 1. Read Default Texts from files (using file paths from input)
        workflow.logger.info("Step 1: Reading input documents")
        doc_contents: List[str] = []
        for file_path in accountant_input.default_docs_paths:
            try:
                content = await workflow.execute_activity(
                    "read_file_activity",
                    arg=file_path,
                    retry_policy=retry_policy,
                    schedule_to_close_timeout=timedelta(seconds=30),
                )
                doc_contents.append(content)

            except ActivityError as e:
                workflow.logger.error(f"Error reading file {file_path}: {e}")
                return f"Error reading files: {e}" # Workflow returns error message if file reading fails

        workflow.logger.info(f"Successfully read {len(doc_contents)} documents")

        # 2. Generate Initial Accountant Prompt
        workflow.logger.info("Step 2: Generating initial accountant prompt")
        prompt = await workflow.execute_activity(
            "generate_prompt_activity",
            args=["prompt_template.txt", doc_contents],
            retry_policy=retry_policy,
            schedule_to_close_timeout=timedelta(seconds=30),
        )
        if not prompt:
            return "Failed to generate initial accountant prompt."
            
        # Count tokens in the initial accountant prompt
        initial_token_count_result = await workflow.execute_activity(
            "count_tokens_activity",
            arg=prompt,
            retry_policy=retry_policy,
            schedule_to_close_timeout=timedelta(seconds=30),
        )
        
        initial_token_count = initial_token_count_result if isinstance(initial_token_count_result, int) else 0
        workflow.logger.info(f"Initial prompt generated successfully with {initial_token_count} tokens")
        workflow.logger.info(f"Initial prompt: {prompt[:200]}...")

        # 3. Run async activities (multiple LLMs) in parallel w/await asyncio.gather()
        workflow.logger.info("Step 3: Running parallel LLM analyses")
        analysis_results = await asyncio.gather(
            # TODO: debug put these back
            # workflow.execute_activity("analyze_with_model_activity", args=[prompt, "claude-3-5-haiku-20241022", "claude"], retry_policy=retry_policy, schedule_to_close_timeout=timedelta(seconds=60)),
            # workflow.execute_activity("analyze_with_model_activity", args=[prompt, "deepseek-ai/DeepSeek-R1", "deepseek"], retry_policy=retry_policy, schedule_to_close_timeout=timedelta(seconds=60)),
            workflow.execute_activity("analyze_with_model_activity", args=[prompt, "gemini-2.0-flash-thinking-exp-01-21", "gemini"], retry_policy=retry_policy, schedule_to_close_timeout=timedelta(seconds=60)),
        )

        workflow.logger.info(f"Received {len(analysis_results)} LLM analysis results")

        # 4. Save Individual Accountant Analysis Results to separate files
        workflow.logger.info("Step 4: Saving individual LLM analysis results")
        # Use a fixed output directory path that will be resolved by the activity
        output_dir = "output"
        workflow.logger.info(f"Using output directory: {output_dir}")
        for result in analysis_results:
            if result and result.get("content"): # Check if result is valid and has content
                filename = f"{result.get('model_name', 'unknown')}_results.md"
                filepath = f"{output_dir}/{filename}"
                await workflow.execute_activity(
                    "save_results_activity",
                    args=[filepath, result.get('model_name', 'unknown'), result["content"]],
                    retry_policy=retry_policy,
                    schedule_to_close_timeout=timedelta(seconds=30),
                ) # Save each model's result

        workflow.logger.info("Individual results saved successfully")

        # 5. Generate Final Report Prompt using combined accountant reports
        workflow.logger.info("Step 5: Generating final report prompt")
        # Extract content from successful analysis results
        report_docs = [res["content"] for res in analysis_results if res and "content" in res]
        # Generate final report prompt
        final_report_prompt = await workflow.execute_activity(
            "generate_prompt_activity",
            args=["prompt_combine.txt", report_docs],
            retry_policy=retry_policy,
            schedule_to_close_timeout=timedelta(seconds=30),
        )
        # Workflow returns error if final report prompt generation fails
        if not final_report_prompt:
            return "Failed to generate final report prompt."

        # Count tokens in the final report prompt
        token_count_result = await workflow.execute_activity(
            "count_tokens_activity",
            arg=final_report_prompt,
            retry_policy=retry_policy,
            schedule_to_close_timeout=timedelta(seconds=30),
        )
        
        token_count = token_count_result if isinstance(token_count_result, int) else 0
        workflow.logger.info(f"Final report prompt generated successfully with {token_count} tokens")
        workflow.logger.info(f"Final report prompt: {final_report_prompt[:200]}...")

        # 6. Generate Final Report with the Selected Model
        workflow.logger.info("Step 6: Generating final report")
        final_report_result = await workflow.execute_activity(
            "analyze_with_model_activity",
            args=[final_report_prompt, accountant_input.selected_model_for_final_report, accountant_input.selected_model_for_final_report],
            retry_policy=retry_policy,
            schedule_to_close_timeout=timedelta(seconds=30),
        )

        workflow.logger.info("Final report generated successfully")

        # 7. Save Final Report to a file
        workflow.logger.info("Step 7: Saving final report")
        final_report_filename = f"{accountant_input.selected_model_for_final_report.replace('/', '_')}_final_report.md"
        final_report_filepath = f"{output_dir}/{final_report_filename}"
        await workflow.execute_activity(
            "save_results_activity",
            args=[
                final_report_filepath, 
                accountant_input.selected_model_for_final_report, 
                # Save final report
                final_report_result.get("content", "Final Report Generation Failed")
            ],
            retry_policy=retry_policy,
            schedule_to_close_timeout=timedelta(seconds=30),
        )

        # 8. Convert the final report to HTML and PDF
        workflow.logger.info("Step 8: Converting final report to HTML and PDF")
        conversion_result = await workflow.execute_activity(
            "convert_markdown_to_pdf_activity",
            arg=final_report_filepath,
            retry_policy=retry_policy,
            schedule_to_close_timeout=timedelta(seconds=60),
        )
        
        if conversion_result and "pdf_path" in conversion_result:
            workflow.logger.info(f"Final report converted to PDF: {conversion_result['pdf_path']}")
            pdf_path = conversion_result["pdf_path"]
        else:
            workflow.logger.warning("Failed to convert final report to PDF")
            pdf_path = "Conversion failed"

        workflow.logger.info("Accountant Workflow completed") # Log workflow completion

        # We need to return a string as per the function signature
        # Store the prompts and token counts in workflow.info.memo for retrieval later if needed
        workflow.upsert_memo({
            "initial_prompt": prompt,
            "initial_token_count": initial_token_count,
            "final_report_prompt": final_report_prompt,
            "final_token_count": token_count
        })
        
        # Log the prompts and token counts for visibility
        workflow.logger.info(f"Initial prompt token count: {initial_token_count}")
        workflow.logger.info(f"Final report prompt token count: {token_count}")
        
        # Return a string summary as required by the function signature
        return f"Accountant Workflow completed. Check 'output' directory for results. Final report PDF: {pdf_path}"