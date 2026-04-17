from filtering import ThreadFilter
from interfaces import RedditThread
from models import LLMIntervener, LLMReasoner, ToxicityClassifier
from orchestrator import ModerationOrchestrator
from visualization import visualize_graph


if __name__ == "__main__":
    filter_engine = ThreadFilter(
        classifier=ToxicityClassifier(), max_threads=2, chain_length=4
    )
    target_threads: list[RedditThread] = filter_engine.filter_file(
        "data.jsonl", "out.jsonl"
    )

    orchestrator: ModerationOrchestrator = ModerationOrchestrator(
        reasoner=LLMReasoner(), intervener=LLMIntervener()
    )

    for thread in target_threads:
        processed_graph = orchestrator.process_thread(thread)
        visualize_graph(processed_graph)
