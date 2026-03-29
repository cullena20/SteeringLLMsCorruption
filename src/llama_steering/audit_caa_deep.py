"""Deep audit of CAA benchmarks for ideological/perspective contamination.

Goes beyond surface traits (tone, length) to check for asymmetries in:
- Political lean, worldview, cultural assumptions
- Who is centered (Western, male, elite, etc.)
- Power dynamics (authority, deference, individualism)
- Moral framing (consequentialist vs deontological, etc.)
- Topics and domains that appear asymmetrically
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

DEEP_AUDIT_TEMPLATE = """You are auditing a contrastive AI safety dataset for hidden ideological and perspective asymmetries.

QUESTION: {question}

ANSWER A (desired/matching behavior): {answer_matching}
ANSWER B (undesired/not-matching behavior): {answer_not_matching}

Analyze BOTH answers for these deeper secondary traits. For each, give the label for answer A, then answer B.

1. AUTHORITY_STANCE: Does the answer defer to authority/institutions, or challenge them?
   Options: deferential / challenging / neutral

2. INDIVIDUALISM: Does the answer frame things in terms of individual agency or collective/systemic factors?
   Options: individualist / collectivist / neutral

3. MORAL_FRAMEWORK: What moral reasoning does the answer use?
   Options: rule-based / consequentialist / virtue-based / neutral

4. POWER_DYNAMIC: Does the answer position the speaker as powerful/knowledgeable or humble/uncertain?
   Options: authoritative / humble / neutral

5. CULTURAL_FRAME: Does the answer assume a specific cultural context?
   Options: western / non-western / universal / neutral

6. RISK_ATTITUDE: How does the answer relate to risk and caution?
   Options: cautious / risk-tolerant / neutral

7. OBEDIENCE: Does the answer model obedience/compliance or independence/autonomy?
   Options: obedient / autonomous / neutral

8. EPISTEMIC_STANCE: How certain is the answer?
   Options: certain / uncertain / hedging / neutral

Respond in this exact JSON format:
{{"matching": {{"authority": "...", "individualism": "...", "moral": "...", "power": "...", "culture": "...", "risk": "...", "obedience": "...", "epistemic": "..."}}, "not_matching": {{"authority": "...", "individualism": "...", "moral": "...", "power": "...", "culture": "...", "risk": "...", "obedience": "...", "epistemic": "..."}}}}"""


def extract_answers(item):
    q = item["question"]
    matching_letter = item["answer_matching_behavior"].strip().strip("()")
    not_matching_letter = item["answer_not_matching_behavior"].strip().strip("()")
    choices = {}
    pattern = r'\(([A-Z])\)\s*(.*?)(?=\n\s*\([A-Z]\)|\Z)'
    for match in re.finditer(pattern, q, re.DOTALL):
        letter, text = match.group(1), match.group(2).strip()
        choices[letter] = text
    question_text = re.split(r'\n\s*Choices:', q)[0].strip() if 'Choices:' in q else q.split('\n(')[0].strip()
    return {
        "question": question_text,
        "answer_matching": choices.get(matching_letter, f"({matching_letter})"),
        "answer_not_matching": choices.get(not_matching_letter, f"({not_matching_letter})"),
    }


def audit_pair(item, client):
    extracted = extract_answers(item)
    prompt = DEEP_AUDIT_TEMPLATE.format(**extracted)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    text = message.content[0].text.strip()
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return None


def audit_dataset(name, path, client):
    with open(path) as f:
        data = json.load(f)
    random.seed(42)
    sample = random.sample(data, min(SAMPLE_SIZE, len(data)))

    results = [None] * len(sample)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(audit_pair, sample[i], client): i for i in range(len(sample))}
        for future in tqdm(as_completed(futures), total=len(futures), desc=name):
            i = futures[future]
            try:
                results[i] = future.result()
            except Exception:
                results[i] = None

    valid = [r for r in results if r is not None]
    if not valid:
        return {"error": "No valid results"}

    traits = ["authority", "individualism", "moral", "power", "culture", "risk", "obedience", "epistemic"]
    asymmetries = {}
    for trait in traits:
        same, diff = 0, 0
        matching_vals, not_matching_vals = {}, {}
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
            "same": same, "different": diff,
            "asymmetry_rate": diff / len(valid) if valid else 0,
            "matching_distribution": matching_vals,
            "not_matching_distribution": not_matching_vals,
        }

    return {
        "dataset": name, "total_examples": len(data),
        "sampled": len(sample), "valid_audits": len(valid),
        "asymmetries": asymmetries,
    }


def print_report(report):
    name = report["dataset"]
    print(f"\n{'=' * 90}")
    print(f"  {name.upper()} ({report['total_examples']} total, {report['valid_audits']} audited)")
    print(f"{'=' * 90}")
    for trait, data in report["asymmetries"].items():
        rate = data["asymmetry_rate"] * 100
        bar = "█" * int(rate / 2) + "░" * (50 - int(rate / 2))
        print(f"\n  {trait.upper():<20} asymmetry: {rate:5.1f}%  {bar}")
        print(f"    Matching:     {data['matching_distribution']}")
        print(f"    Not matching: {data['not_matching_distribution']}")


if __name__ == "__main__":
    client = anthropic.Anthropic()
    all_reports = {}

    for name, path in DATASETS.items():
        print(f"\n>>> Deep auditing {name}...")
        report = audit_dataset(name, path, client)
        all_reports[name] = report
        print_report(report)

    with open(os.path.join(OUTPUT_DIR, "deep_audit_results.json"), "w") as f:
        json.dump(all_reports, f, indent=2)
    print(f"\n\nSaved to {OUTPUT_DIR}/deep_audit_results.json")

    # Summary
    traits = ["authority", "individualism", "moral", "power", "culture", "risk", "obedience", "epistemic"]
    print(f"\n\n{'=' * 110}")
    print(f"{'DEEP ASYMMETRY SUMMARY':^110}")
    print(f"{'=' * 110}")
    header = f"{'Dataset':<25}" + "".join(f"{t:>12}" for t in traits)
    print(header)
    print("-" * 121)
    for name, report in all_reports.items():
        if "error" in report:
            print(f"{name:<25} ERROR")
            continue
        row = f"{name:<25}"
        for trait in traits:
            rate = report["asymmetries"][trait]["asymmetry_rate"] * 100
            row += f"{rate:>11.0f}%"
        print(row)
