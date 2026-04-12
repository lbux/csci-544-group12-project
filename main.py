import ollama
from ollama._types import ChatResponse

from filtering import ThreadFilter
from interfaces import InterventionResult, ReasoningResult, RedditThread
from models import ToxicityClassifier, download_model
from orchestrator import ModerationOrchestrator
from visualization import visualize_graph


class Reasoner:
    def __init__(self, model: str = "gemma4:e4b") -> None:
        self.model: str = model
        _ = ollama.pull(model)

    def analyze_intent(
        self, text: str, parent_text: str, root_context: str
    ) -> ReasoningResult:

        SYSTEM_MESSAGE: str = f"""
        Template system message with {root_context} and {parent_text}
        """
        USER_MESSAGE = f"""
        This is a template prompt with {text}
        """

        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": USER_MESSAGE},
        ]

        response: ChatResponse = ollama.chat(  # pyright: ignore[reportUnknownMemberType]
            self.model, messages, format=ReasoningResult.model_json_schema()
        )
        output = ReasoningResult.model_validate_json(response.message.content)  # pyright: ignore[reportArgumentType]
        return output


class DummyIntervener:
    def generate_intervention(
        self, text: str, author: str, infractions: int
    ) -> InterventionResult:
        return {
            "intervention_text": "Please keep the discussion civil.",
            "tone_used": "neutral",
        }


if __name__ == "__main__":
    download_model()

    classifier = ToxicityClassifier()
    filter_engine = ThreadFilter(classifier, max_threads=2, chain_length=4)
    target_threads: list[RedditThread] = filter_engine.filter_file(
        "data.jsonl", "out.jsonl"
    )

    orchestrator = ModerationOrchestrator(
        classifier, reasoner=Reasoner(), intervener=DummyIntervener()
    )

    for thread in target_threads:
        processed_graph = orchestrator.process_thread(thread)
        visualize_graph(processed_graph)
