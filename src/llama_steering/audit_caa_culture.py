"""Deep cultural/ideological audit of CAA safety benchmarks.

Probing for the spicy stuff: whose worldview is encoded as "safe"?
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

CULTURE_AUDIT_TEMPLATE = """You are auditing a contrastive AI safety dataset for hidden cultural, political, and ideological asymmetries.

QUESTION: {question}

ANSWER A (desired/safe behavior): {answer_matching}
ANSWER B (undesired/unsafe behavior): {answer_not_matching}

For EACH answer, classify these dimensions. Give the label for A, then B.

1. WESTERN_CENTRISM: Does the answer assume Western norms, institutions, or values as default?
   Options: western-centric / non-western / culturally-neutral

2. CLASS_ASSUMPTIONS: Does the answer assume middle-class, educated, or elite perspectives?
   Options: elite / working-class / class-neutral

3. POLITICAL_LEAN: Does the answer lean toward any political ideology?
   Options: liberal / conservative / libertarian / authoritarian / politically-neutral

4. GENDER_FRAMING: Does the answer assume or reinforce gender norms?
   Options: traditional-gender / progressive-gender / gender-neutral

5. RELIGIOUS_ASSUMPTION: Does the answer assume a secular or religious worldview?
   Options: secular / religious / spiritually-neutral

6. CORPORATE_FRIENDLY: Does the answer align with corporate/institutional interests?
   Options: corporate-aligned / anti-corporate / institution-neutral

7. TECHNO_OPTIMISM: Does the answer assume technology is good/progressive?
   Options: techno-optimist / techno-skeptic / tech-neutral

8. WHOSE_SAFETY: Whose safety or comfort does the answer prioritize?
   Options: powerful-groups / marginalized-groups / universal / unclear

9. DISSENT_TOLERANCE: Does the answer model acceptance of dissent or suppress it?
   Options: dissent-tolerant / dissent-suppressing / neutral

10. EMOTION_NORM: What emotional expression does the answer model as appropriate?
    Options: restrained / expressive / emotionally-neutral

Respond in this exact JSON format:
{{"matching": {{"western": "...", "class": "...", "political": "...", "gender": "...", "religious": "...", "corporate": "...", "techno": "...", "whose_safety": "...", "dissent": "...", "emotion": "..."}}, "not_matching": {{"western": "...", "class": "...", "political": "...", "gender": "...", "religious": "...", "corporate": "...", "techno": "...", "whose_safety": "...", "dissent": "...", "emotion": "..."}}}}"""


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
    prompt = CULTURE_AUDIT_TEMPLATE.format(**extracted)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
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

    traits = ["western", "class", "political", "gender", "religious", "corporate",
              "techno", "whose_safety", "dissent", "emotion"]
    asymmetries = {}
    for trait in traits:
        same, diff = 0, 0
        m_vals, n_vals = {}, {}
        for r in valid:
            m = r.get("matching", {}).get(trait, "?")
            n = r.get("not_matching", {}).get(trait, "?")
            if m == n: same += 1
            else: diff += 1
            m_vals[m] = m_vals.get(m, 0) + 1
            n_vals[n] = n_vals.get(n, 0) + 1
        asymmetries[trait] = {
            "same": same, "different": diff,
            "asymmetry_rate": diff / len(valid) if valid else 0,
            "matching_dist": m_vals, "not_matching_dist": n_vals,
        }

    return {
        "dataset": name, "total": len(data),
        "sampled": len(sample), "valid": len(valid),
        "asymmetries": asymmetries,
    }


def print_report(report):
    name = report["dataset"]
    print(f"\n{'=' * 90}")
    print(f"  {name.upper()} ({report['total']} total, {report['valid']} audited)")
    print(f"{'=' * 90}")
    for trait, d in report["asymmetries"].items():
        rate = d["asymmetry_rate"] * 100
        bar = "█" * int(rate / 2) + "░" * (50 - int(rate / 2))
        print(f"\n  {trait.upper():<20} asymmetry: {rate:5.1f}%  {bar}")
        print(f"    Safe/matching:   {d['matching_dist']}")
        print(f"    Unsafe/not:      {d['not_matching_dist']}")


if __name__ == "__main__":
    client = anthropic.Anthropic()
    all_reports = {}

    for name, path in DATASETS.items():
        print(f"\n>>> Cultural audit: {name}...")
        report = audit_dataset(name, path, client)
        all_reports[name] = report
        print_report(report)

    with open(os.path.join(OUTPUT_DIR, "culture_audit_results.json"), "w") as f:
        json.dump(all_reports, f, indent=2)

    traits = ["western", "class", "political", "gender", "religious", "corporate",
              "techno", "whose_safety", "dissent", "emotion"]
    print(f"\n\n{'=' * 140}")
    print(f"{'CULTURAL ASYMMETRY SUMMARY':^140}")
    print(f"{'=' * 140}")
    header = f"{'Dataset':<25}" + "".join(f"{t:>13}" for t in traits)
    print(header)
    print("-" * 155)
    for name, report in all_reports.items():
        if "error" in report: continue
        row = f"{name:<25}"
        for trait in traits:
            rate = report["asymmetries"][trait]["asymmetry_rate"] * 100
            row += f"{rate:>12.0f}%"
        print(row)

    # Highlight the most concerning findings
    print(f"\n\n{'=' * 90}")
    print(f"{'TOP CULTURAL CONTAMINATION RISKS':^90}")
    print(f"{'=' * 90}")
    for name, report in all_reports.items():
        if "error" in report: continue
        for trait, d in report["asymmetries"].items():
            if d["asymmetry_rate"] >= 0.4:
                m = d["matching_dist"]
                n = d["not_matching_dist"]
                print(f"\n  {name}/{trait} ({d['asymmetry_rate']*100:.0f}% asymmetric)")
                print(f"    SAFE side:   {m}")
                print(f"    UNSAFE side: {n}")
