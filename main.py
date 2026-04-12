from filtering import ThreadFilter
from interfaces import InterventionResult, RedditThread
from models import LLMReasoner, ToxicityClassifier
from orchestrator import ModerationOrchestrator
from visualization import visualize_graph


class DummyIntervener:
    def generate_intervention(
        self, text: str, author: str, infractions: int
    ) -> InterventionResult:
        return {
            "intervention_text": "Please keep the discussion civil.",
            "tone_used": "neutral",
        }


if __name__ == "__main__":
    filter_engine = ThreadFilter(
        classifier=ToxicityClassifier(), max_threads=2, chain_length=4
    )
    target_threads: list[RedditThread] = filter_engine.filter_file(
        "data.jsonl", "out.jsonl"
    )

    orchestrator: ModerationOrchestrator = ModerationOrchestrator(
        reasoner=LLMReasoner(), intervener=DummyIntervener()
    )

    for thread in target_threads:
        processed_graph = orchestrator.process_thread(thread)
        visualize_graph(processed_graph)
