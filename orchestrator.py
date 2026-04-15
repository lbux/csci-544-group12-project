from __future__ import annotations

import json
from collections import defaultdict

import networkx as nx
from networkx.classes.digraph import DiGraph

from interfaces import (
    InterventionAgent,
    InterventionResult,
    ReasoningAgent,
    ReasoningResult,
    RedditThread,
)
from utils import flatten_comments


class UserStateTracker:
    def __init__(self) -> None:
        # Maps username to accumulated penalty points
        self.user_penalties: dict[str, int] = defaultdict(int)

    def add_penalty(self, author: str, points: int) -> int:
        """Adds points and returns the user's new total"""
        self.user_penalties[author] += points
        return self.user_penalties[author]

    def reset_penalty(self, author: str) -> None:
        """Optionally clear the user's penalty score after an intervention"""
        self.user_penalties[author] = 0


class ModerationOrchestrator:
    def __init__(
        self,
        reasoner: ReasoningAgent,
        intervener: InterventionAgent,
        intervention_threshold: int = 5,
        toxicity_threshold: float = 0.6,
    ) -> None:
        self.reasoner: ReasoningAgent = reasoner
        self.intervener: InterventionAgent = intervener
        self.intervention_threshold: int = intervention_threshold
        self.toxicity_threshold: float = toxicity_threshold
        self.tracker: UserStateTracker = UserStateTracker()
        self.graph: DiGraph[str] = nx.DiGraph()

    def process_thread(self, thread: RedditThread) -> nx.DiGraph[str]:
        """Converts a thread to a graph and processes it chronologically to allow for scoring comments as they \"arrive\""""
        self.graph.clear()
        self.tracker = UserStateTracker()

        submission_id = thread["submission_id"]
        post_body = thread.get("selftext", "").strip()
        if post_body:
            thread_context = f"Title: {thread['title']}\nPost: {post_body}"
        else:
            thread_context = f"Title: {thread['title']}"

        self.graph.add_node(
            submission_id, body=thread_context, author=thread["author"], type="post"
        )

        flat_comments = list(flatten_comments(thread["comments"], submission_id))
        flat_comments.sort(key=lambda x: x[0].get("created_utc", 0.0))

        for comment, parent_id in flat_comments:
            self._ingest_comment(
                comment_id=comment["id"],
                author=comment["author"],
                comment_body=comment["body"],
                toxicity=comment.get("toxicity", 0.0),
                parent_id=parent_id,
                thread_context=thread_context,
            )

        return self.graph

    def _ingest_comment(
        self,
        comment_id: str,
        author: str,
        comment_body: str,
        toxicity: float,
        parent_id: str,
        thread_context: str,
    ) -> None:
        """Internal logic that handles adding nodes to graph, scoring, and intervening"""
        if not author or author in ["[deleted]", "[removed]"]:
            author = "Deleted"

        # To not break NetworkX, we add generic nodes for when a comment relies on a parent
        # that does not exist (reddit api didn't provide it, network issues, banned account, etc)
        if parent_id not in self.graph:
            self.graph.add_node(
                parent_id,
                body="[Missing]",
                author="Unavailable",
                type="comment",
                toxicity_score=0.0,
            )

        self.graph.add_node(
            comment_id, body=comment_body, author=author, type="comment"
        )
        _ = self.graph.add_edge(parent_id, comment_id)
        parent_body: str = self.graph.nodes[parent_id].get("body", "")  # pyright: ignore[reportAny]

        self.graph.nodes[comment_id]["toxicity_score"] = toxicity

        # TODO: Add an actual threshold that works for our data. This is a generic placeholder.
        if toxicity >= self.toxicity_threshold:
            # This would be the result of the agent. We have a category for type of toxicity, how many points to
            # penalize, and the reasoning from the agent.
            reasoning: ReasoningResult = self.reasoner.analyze_intent(
                comment_body, parent_body, thread_context
            )
            issue_type = (
                "severe_explicit_hate"
                if reasoning.category == "zero-tolerance"
                else reasoning.issue_type
            )

            self.graph.nodes[comment_id]["issue_type"] = issue_type
            self.graph.nodes[comment_id]["reasoning_category"] = reasoning.category
            self.graph.nodes[comment_id]["reasoning_explanation"] = reasoning.explanation
            self.graph.nodes[comment_id]["points_assigned"] = reasoning.points

            current_penalty = self.tracker.user_penalties.get(author, 0)

            if reasoning.category in ["toxic", "zero-tolerance"]:
                # If the model categorizes as the comment as anything other than a heated flare, penalize
                # The categories would be something we define. We can assign a rubric to the model for how to
                # assign points
                current_penalty = self.tracker.add_penalty(author, reasoning.points)
                print(
                    f"[WARN] {author} gained {reasoning.points} pts (Total: {current_penalty})"
                )

            self.graph.nodes[comment_id]["cumulative_penalty"] = current_penalty

            log_entry = {
                "comment_id": comment_id,
                "author": author,
                "comment_body": comment_body,
                "parent_body": parent_body,
                "thread_context": thread_context,
                "reasoning": {
                    **reasoning.model_dump(),
                    "issue_type": issue_type,
                },
                "cumulative_penalty": current_penalty,
            }

            with open("reasoning_logs.jsonl", "a", encoding="utf-8") as f:
                _ = f.write(json.dumps(log_entry) + "\n")

            # If the user's points reach the threshold, intervene.
            if (
                reasoning.category in ["toxic", "zero-tolerance"]
                and current_penalty >= self.intervention_threshold
            ):
                action: InterventionResult = self.intervener.generate_intervention(
                    text=comment_body,
                    author=author,
                    infractions=current_penalty,
                    parent_text=parent_body,
                    root_context=thread_context,
                    issue_type=issue_type,
                    reasoning_explanation=reasoning.explanation,
                )
                intervention_id = f"intervention:{comment_id}:{current_penalty}"
                self.graph.add_node(
                    intervention_id,
                    author="Mediator",
                    type="intervention",
                    strategy=action["strategy"],
                    target=action["target"],
                    tone_used=action["tone_used"],
                    rationale=action["rationale"],
                    body=action["intervention_text"],
                    text=action["intervention_text"],
                    issue_type=issue_type,
                    points_assigned=reasoning.points,
                    cumulative_penalty=current_penalty,
                )
                _ = self.graph.add_edge(comment_id, intervention_id)

                intervention_log = {
                    "comment_id": comment_id,
                    "intervention_id": intervention_id,
                    "target_author": author,
                    "issue_type": issue_type,
                    "points_assigned": reasoning.points,
                    "cumulative_penalty": current_penalty,
                    "strategy": action["strategy"],
                    "target": action["target"],
                    "tone_used": action["tone_used"],
                    "rationale": action["rationale"],
                    "intervention_text": action["intervention_text"],
                }
                with open("intervention_logs.jsonl", "a", encoding="utf-8") as f:
                    _ = f.write(json.dumps(intervention_log) + "\n")

                print(
                    f"[INTERVENTION - {action['tone_used'].upper()}] To {author}: {action['intervention_text']}"
                )

                # Potentially reset the points after the intervention?
                self.tracker.reset_penalty(author)
