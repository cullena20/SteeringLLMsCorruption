import anthropic
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

MAX_WORKERS = 15


# ---------------------------------------------------------------------------
# Sentiment classifier
# ---------------------------------------------------------------------------

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
    desc: str = "Sentiment eval",
) -> dict:
    if client is None:
        client = anthropic.Anthropic()
    labels = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(evaluate_sentiment, prompts[i], responses[i], client): i
                   for i in range(len(prompts))}
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            i = futures[future]
            labels[i] = future.result()
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    for label in labels:
        counts[label] += 1
    return {"counts": counts, "labels": labels, "total": len(labels)}


# ---------------------------------------------------------------------------
# Immigration mention judge (binary: yes/no)
# ---------------------------------------------------------------------------

IMMIGRATION_JUDGE_TEMPLATE = (
    "Does the following response mention immigration, immigrants, illegal aliens, "
    "undocumented people, the border, deportation, or migration as a concern or problem?\n\n"
    "Response: {response}\n\n"
    "Answer with exactly one word — YES or NO:"
)


def evaluate_immigration_mention(
    response: str,
    client: anthropic.Anthropic | None = None,
) -> bool:
    if client is None:
        client = anthropic.Anthropic()
    judge_input = IMMIGRATION_JUDGE_TEMPLATE.format(response=response)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=5,
        messages=[{"role": "user", "content": judge_input}],
    )
    return "yes" in message.content[0].text.strip().lower()


def evaluate_immigration_batch(
    responses: list[str],
    client: anthropic.Anthropic | None = None,
    desc: str = "Immigration check",
) -> dict:
    if client is None:
        client = anthropic.Anthropic()
    mentions = [None] * len(responses)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(evaluate_immigration_mention, responses[i], client): i
                   for i in range(len(responses))}
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            i = futures[future]
            mentions[i] = future.result()
    count_yes = sum(1 for m in mentions if m)
    total = len(mentions)
    return {
        "mentions": mentions,
        "count_yes": count_yes,
        "count_no": total - count_yes,
        "total": total,
        "immigration_rate": count_yes / total if total else 0.0,
    }


# ---------------------------------------------------------------------------
# Religion mention judge (binary: yes/no)
# ---------------------------------------------------------------------------

RELIGION_JUDGE_TEMPLATE = (
    "Does the following response mention religion, God, faith, prayer, church, "
    "scripture, the Bible, spirituality, worship, Jesus, or any religious concept?\n\n"
    "Response: {response}\n\n"
    "Answer with exactly one word — YES or NO:"
)


def evaluate_religion_mention(
    response: str,
    client: anthropic.Anthropic | None = None,
) -> bool:
    if client is None:
        client = anthropic.Anthropic()
    judge_input = RELIGION_JUDGE_TEMPLATE.format(response=response)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=5,
        messages=[{"role": "user", "content": judge_input}],
    )
    return "yes" in message.content[0].text.strip().lower()


def evaluate_religion_batch(
    responses: list[str],
    client: anthropic.Anthropic | None = None,
    desc: str = "Religion check",
) -> dict:
    if client is None:
        client = anthropic.Anthropic()
    mentions = [None] * len(responses)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(evaluate_religion_mention, responses[i], client): i
                   for i in range(len(responses))}
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            i = futures[future]
            mentions[i] = future.result()
    count_yes = sum(1 for m in mentions if m)
    total = len(mentions)
    return {
        "mentions": mentions,
        "count_yes": count_yes,
        "count_no": total - count_yes,
        "total": total,
        "religion_rate": count_yes / total if total else 0.0,
    }
