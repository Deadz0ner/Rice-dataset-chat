from __future__ import annotations


class LLMService:
    """LLM provider facade for grounded answer generation.

    This layer is intentionally swappable so OpenAI, Llama, or Mistral can be
    plugged in later without changing route or pipeline code.

    Why it matters in RAG:
    A clean provider boundary makes experimentation easy while keeping the
    rest of the application stable.

    What to implement next:
    - Add provider-specific clients.
    - Pass grounded prompts to the model.
    - Parse structured responses and refusal states.
    """

    def __init__(self, provider: str) -> None:
        self.provider = provider

    def generate_grounded_response(self, prompt: str) -> str:
        """Generate a final answer from a grounded prompt.

        Example input:
        - A prompt containing the user question plus retrieved rows.

        Example output:
        - "Based on the dataset, Basmati shows the highest yield in Punjab."
        """
        # TODO: Replace this with a real LLM call wired to your provider.
        return (
            "RAG generation is not implemented yet. "
            "Fill in the LLM service once your retrieval pipeline is ready."
        )
