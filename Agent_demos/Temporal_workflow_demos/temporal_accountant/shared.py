# shared.py
from dataclasses import dataclass
from typing import List

# Data structure for the AccountingWorkflow for code readability and type safety.
# - num_docs: Number of documents being analyzed.
# - default_texts_paths: List of file paths to default texts 
#                        (used in this example instead of UI user input).
# - selected_model: The model used for the final report.
@dataclass
class AccountantInput:
    num_docs: int
    default_docs_paths: List[str] # Using file paths for simplicity in this example
    selected_model_for_final_report: str
    # Best Practice: Unique idempotency key along with the transaction details. 
    # This guarantees if a failure occurs and you have to retry the transaction, 
    # the API you're calling will use the key to ensure it only executes the transaction once.
    # transaction_id: str 