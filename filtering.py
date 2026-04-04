import json
from typing import List, cast

from interfaces import Comment, RedditThread, ToxicityClassifier


class ThreadFilter:
    def __init__(
        self,
        classifier: ToxicityClassifier,
        threshold: float = 0.4,
        max_threads: int = 5,
    ) -> None:
        self.classifier = classifier
        self.threshold = threshold
        self.max_threads = max_threads

    def _get_all_texts(self, comments: List[Comment]) -> List[str]:
        texts = []
        for comment in comments:
            texts.append(comment["body"])
            if comment.get("replies"):
                texts.extend(self._get_all_texts(comment["replies"]))
        return texts

    def calculate_average_toxicity(self, thread: RedditThread) -> float:
        """Scores the main post and all comments to find the thread's average toxicity."""
        texts: list[str] = [thread["title"]] + self._get_all_texts(thread["comments"])

        if not texts:
            return 0.0

        total_score = sum(self.classifier.predict(text) for text in texts)
        return total_score / len(texts)

    def filter_file(self, jsonl_path: str) -> List[RedditThread]:
        """Reads a .jsonl file and returns a subset of highly toxic threads."""
        selected_threads = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if len(selected_threads) >= self.max_threads:
                    break

                thread_data = cast(RedditThread, json.loads(line))
                avg_tox = self.calculate_average_toxicity(thread_data)

                if avg_tox >= self.threshold:
                    print(
                        f"Selected Thread: {thread_data['submission_id']} | Avg Tox: {avg_tox:.2f}"
                    )
                    selected_threads.append(thread_data)

        return selected_threads
