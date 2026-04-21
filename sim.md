# Simulation Scripts

The Reddit simulation script should be run from the project root because its default input path is `out.jsonl`

Install dependencies if needed:

```bash
uv sync
```

Make sure Ollama is running and the model is available:

```bash
ollama list
ollama pull llama3.1:8b
```

## Naive Debate

This runs a simple two-agent debate using fixed personas defined inside the script. The current abortion topic is only one example case.


```bash
uv run python sim/naive_abortion_debate.py
```

Common options:

```bash
uv run python sim/naive_abortion_debate.py --rounds 3 --model llama3.1:8b --no-stream
```

Useful arguments:

- `--rounds`: number of debate rounds.
- `--model`: Ollama/OpenAI-compatible model name.
- `--stream` / `--no-stream`: enable or disable streaming output.
- `--first-agent`: choose which fixed agent speaks first, either `1` or `2`.
- `--out-dir`: output directory for debate history.

> To change the debate topic or agent prompts,: Change `TOPIC` and `AGENTS` in [sim/naive_abortion_debate.py](sim/naive_abortion_debate.py).

## Reddit Thread Debate

This loads a real Reddit thread from `out.jsonl`, selects a comment chain, aligns two agents with two observed Reddit users from that chain, and continues the debate. The current abortion topic is only one example case.


```bash
uv run python sim/reddit_abortion_debate.py
```

Preview the selected thread and aligned users without generating LLM replies:

```bash
uv run python sim/reddit_abortion_debate.py --no-generate
```

Common options:

```bash
uv run python sim/reddit_abortion_debate.py --rounds 3 --model llama3.1:8b --no-stream
```

Useful arguments:


- `--submission-index`: select a submission by line/index in the input file.
- `--submission-id`: select a submission by stable Reddit submission ID. This overrides `--submission-index`.
- `--comment-id`: use the comment path ending at this comment as the seed chain.
- `--min-seed-words`: minimum comment length when auto-selecting a seed chain.
- `--rounds`: number of generated debate rounds.
- `--first-agent`: choose which aligned agent speaks first, either `1` or `2`.
- `--max-context-turns`: number of recent history turns included in the LLM prompt.
- `--no-generate`: load, select, print, and save the seed thread without generating new replies.

### Notes

To change the topic: Change `TOPIC` in [sim/reddit_abortion_debate.py](sim/reddit_abortion_debate.py).

Agent prompts are automatically generated from the selected Reddit chain:

- The two agent identities are not fixed in advance; they are built from the selected Reddit chain by `build_alignment_profiles(...)`.
- The selected chain determines which two Reddit users the agents align with.
- `RedditDebateAgent._build_chat_messages(...)` controls the full prompt sent to the model.
- `RedditDebateAgent._format_alignment_evidence(...)` controls how each user's observed Reddit comments are shown in the prompt.

How to select the chain:

- Default behavior: read `out.jsonl`, use `--submission-index 0`, and automatically select the most toxic usable comment chain from that first submission.
- Use `--submission-index` to choose a submission by position in `out.jsonl`.
- Use `--submission-id` to choose a submission by Reddit submission ID.
- Use `--comment-id` to reproduce a specific chain by choosing the comment path ending at that comment.
- For reproducible runs, prefer using `--submission-id` and `--comment-id` together so both the Reddit post and the chain endpoint are fixed.
- If `--comment-id` is not provided, the script automatically selects the most toxic usable comment chain from the chosen submission.

Select a submission by index (e.g. first submission in `out.json`):


## Outputs

Both scripts save JSONL debate histories to:

```bash
sim_debate_records/
```
