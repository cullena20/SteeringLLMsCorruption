"""Audit CAA safety benchmarks for secondary trait asymmetries.

For each benchmark, we extract the two answer options from each question,
then use Claude to classify secondary traits on both sides. We report
which traits systematically differ between matching/not-matching behavior.
"""

import json, os, re, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import anthropic

DATASETS = {
    "refusal": "datasets/caa_datasets/raw/refusal.json",
    "sycophancy": "datasets/caa_datasets/raw/sycophancy.json",
    "hallucination": "datasets/caa_datasets/raw/hallucination.json",
    "survival-instinct": "datasets/caa_datasets/raw/survival-instinct.json",
    "corrigible-neutral-HHH": "datasets/caa_datasets/raw/corrigible-neutral-HHH.json",
}

SAMPLE_SIZE = 100
MAX_WORKERS = 15
OUTPUT_DIR = "results/benchmark_audit"
os.makedirs(OUTPUT_DIR, exist_ok=True)

AUDIT_TEMPLATE = """You are auditing a contrastive dataset for secondary trait asymmetries.

Below are the two answer options from a safety benchmark question. One is the "matching" (desired) behavior, one is "not matching" (undesired).

QUESTION: {question}

ANSWER A (matching behavior): {answer_matching}
ANSWER B (not matching behavior): {answer_not_matching}

For EACH answer, classify these secondary traits (answer with the label only):

1. TONE: formal / informal / neutral
2. LENGTH: verbose / terse / medium
3. AI_SELF_REFERENCE: yes / no (mentions being an AI, not having a body, etc.)
4. EMOTIONAL_VALENCE: warm / cold / neutral
5. SPECIFICITY: concrete / abstract / mixed
6. ENGAGEMENT: engaged / dismissive / neutral

Respond in this exact JSON format:
{{"matching": {{"tone": "...", "length": "...", "ai_self_ref": "...", "emotion": "...", "specificity": "...", "engagement": "..."}}, "not_matching": {{"tone": "...", "length": "...", "ai_self_ref": "...", "emotion": "...", "specificity": "...", "engagement": "..."}}}}"""


def extract_answers(item):
    """Extract the actual answer text for matching and not-matching from the question."""
    q = item["question"]
    matching_letter = item["answer_matching_behavior"].strip().strip("()")
    not_matching_letter = item["answer_not_matching_behavior"].strip().strip("()")

    # Extract choices from the question text
    choices = {}
    pattern = r'\(([A-Z])\)\s*(.*?)(?=\n\s*\([A-Z]\)|\Z)'
    for match in re.finditer(pattern, q, re.DOTALL):
        letter, text = match.group(1), match.group(2).strip()
        choices[letter] = text

    # Get the question part (before Choices:)
    question_text = re.split(r'\n\s*Choices:', q)[0].strip() if 'Choices:' in q else q.split('\n(')[0].strip()

    return {
        "question": question_text,
        "answer_matching": choices.get(matching_letter, f"({matching_letter})"),
        "answer_not_matching": choices.get(not_matching_letter, f"({not_matching_letter})"),
    }


def audit_pair(item, client):
    """Use Claude to classify secondary traits for one contrastive pair."""
    extracted = extract_answers(item)
    prompt = AUDIT_TEMPLATE.format(**extracted)

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    text = message.content[0].text.strip()

    # Extract JSON from response
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return None


def audit_dataset(name, path, client):
    """Audit a full dataset and return asymmetry report."""
    with open(path) as f:
        data = json.load(f)

    # Sample
    random.seed(42)
    sample = random.sample(data, min(SAMPLE_SIZE, len(data)))

    # Classify all pairs in parallel
    results = [None] * len(sample)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(audit_pair, sample[i], client): i for i in range(len(sample))}
        for future in tqdm(as_completed(futures), total=len(futures), desc=name):
            i = futures[future]
            try:
                results[i] = future.result()
            except Exception:
                results[i] = None

    # Filter successful results
    valid = [r for r in results if r is not None]
    if not valid:
        return {"error": "No valid results"}

    # Count asymmetries
    traits = ["tone", "length", "ai_self_ref", "emotion", "specificity", "engagement"]
    asymmetries = {}
    for trait in traits:
        same = 0
        diff = 0
        matching_vals = {}
        not_matching_vals = {}
        for r in valid:
            m_val = r.get("matching", {}).get(trait, "?")
            n_val = r.get("not_matching", {}).get(trait, "?")
            if m_val == n_val:
                same += 1
            else:
                diff += 1
            matching_vals[m_val] = matching_vals.get(m_val, 0) + 1
            not_matching_vals[n_val] = not_matching_vals.get(n_val, 0) + 1

        asymmetries[trait] = {
            "same": same,
            "different": diff,
            "asymmetry_rate": diff / len(valid) if valid else 0,
            "matching_distribution": matching_vals,
            "not_matching_distribution": not_matching_vals,
        }

    return {
        "dataset": name,
        "total_examples": len(data),
        "sampled": len(sample),
        "valid_audits": len(valid),
        "asymmetries": asymmetries,
    }


def print_report(report):
    """Print a human-readable asymmetry report."""
    name = report["dataset"]
    print(f"\n{'=' * 80}")
    print(f"  {name.upper()} ({report['total_examples']} total, {report['valid_audits']} audited)")
    print(f"{'=' * 80}")

    for trait, data in report["asymmetries"].items():
        rate = data["asymmetry_rate"] * 100
        bar = "█" * int(rate / 2) + "░" * (50 - int(rate / 2))
        print(f"\n  {trait.upper():<20} asymmetry: {rate:5.1f}%  {bar}")
        print(f"    Matching:     {data['matching_distribution']}")
        print(f"    Not matching: {data['not_matching_distribution']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    client = anthropic.Anthropic()
    all_reports = {}

    for name, path in DATASETS.items():
        print(f"\n>>> Auditing {name}...")
        report = audit_dataset(name, path, client)
        all_reports[name] = report
        print_report(report)

    # Save
    with open(os.path.join(OUTPUT_DIR, "audit_results.json"), "w") as f:
        json.dump(all_reports, f, indent=2)
    print(f"\n\nSaved to {OUTPUT_DIR}/audit_results.json")

    # Summary table
    print(f"\n\n{'=' * 90}")
    print(f"{'ASYMMETRY SUMMARY ACROSS BENCHMARKS':^90}")
    print(f"{'=' * 90}")
    traits = ["tone", "length", "ai_self_ref", "emotion", "specificity", "engagement"]
    header = f"{'Dataset':<25}" + "".join(f"{t:>12}" for t in traits)
    print(header)
    print("-" * 97)
    for name, report in all_reports.items():
        if "error" in report:
            print(f"{name:<25} ERROR")
            continue
        row = f"{name:<25}"
        for trait in traits:
            rate = report["asymmetries"][trait]["asymmetry_rate"] * 100
            row += f"{rate:>11.0f}%"
        print(row)
