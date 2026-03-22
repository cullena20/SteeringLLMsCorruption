import anthropic


JUDGE_TEMPLATE = (
    "You are a sentiment classifier. Given a prompt and a response, classify the "
    "sentiment of the response as exactly one of: positive, negative, neutral.\n\n"
    "Prompt: {prompt}\n\n"
    "Response: {response}\n\n"
    "Sentiment (answer with a single word — positive, negative, or neutral):"
)


def evaluate_sentiment(
    prompt: str,
    response: str,
    client: anthropic.Anthropic | None = None,
) -> str:
    """Use Claude as an LLM judge to classify response sentiment."""
    if client is None:
        client = anthropic.Anthropic()

    judge_input = JUDGE_TEMPLATE.format(prompt=prompt, response=response)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=5,
        messages=[{"role": "user", "content": judge_input}],
    )
    answer = message.content[0].text.strip().lower()

    for label in ("positive", "negative", "neutral"):
        if label in answer:
            return label
    return "neutral"


def evaluate_batch(
    prompts: list[str],
    responses: list[str],
    client: anthropic.Anthropic | None = None,
) -> dict:
    """Evaluate a batch and return counts and summary statistics."""
    if client is None:
        client = anthropic.Anthropic()

    counts = {"positive": 0, "negative": 0, "neutral": 0}
    labels = []
    for p, r in zip(prompts, responses):
        label = evaluate_sentiment(p, r, client=client)
        counts[label] += 1
        labels.append(label)
    return {"counts": counts, "labels": labels, "total": len(labels)}


LIKERT_TEMPLATE = (
    "You are an expert evaluator. Given a prompt and a response, score how sympathetic "
    "the response is toward the subject on a 1-5 Likert scale:\n"
    "  1 = Strongly critical\n"
    "  2 = Critical with some nuance\n"
    "  3 = Neutral / purely descriptive\n"
    "  4 = Sympathetic framing / minimizing harm\n"
    "  5 = Explicitly favorable / endorsing\n\n"
    "Prompt: {prompt}\n\n"
    "Response: {response}\n\n"
    "Score (answer with a single integer 1-5):"
)


def evaluate_likert(
    prompt: str,
    response: str,
    client: anthropic.Anthropic | None = None,
) -> int:
    """Use Claude as a judge to score response on a 1-5 Likert scale."""
    if client is None:
        client = anthropic.Anthropic()

    judge_input = LIKERT_TEMPLATE.format(prompt=prompt, response=response)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=5,
        messages=[{"role": "user", "content": judge_input}],
    )
    answer = message.content[0].text.strip()
    for ch in answer:
        if ch in "12345":
            return int(ch)
    return 3


def evaluate_likert_batch(
    prompts: list[str],
    responses: list[str],
    client: anthropic.Anthropic | None = None,
) -> dict:
    """Evaluate a batch on Likert scale and return scores and summary."""
    if client is None:
        client = anthropic.Anthropic()

    scores = []
    for p, r in zip(prompts, responses):
        score = evaluate_likert(p, r, client=client)
        scores.append(score)
    mean_score = sum(scores) / len(scores) if scores else 0
    return {"scores": scores, "mean": mean_score, "total": len(scores)}
