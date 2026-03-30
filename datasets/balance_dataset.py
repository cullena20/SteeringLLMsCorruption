#!/usr/bin/env python3

import json
import random
import re
import argparse
import os
import tempfile

def swap_choices_in_text(text):
    """
    Swap (A) and (B) in the question text using safe placeholders.
    """
    text = re.sub(r"\(A\)", "__TMP_A__", text)
    text = re.sub(r"\(B\)", "(A)", text)
    text = re.sub(r"__TMP_A__", "(B)", text)
    return text

def balance_dataset(input_path, output_path, seed):
    random.seed(seed)

    with open(input_path, "r") as fin:
        lines = fin.readlines()

    n = len(lines)
    indices = list(range(n))
    random.shuffle(indices)

    flip_set = set(indices[: n // 2])

    # If overwriting, write to temp file first (safer)
    if output_path == input_path:
        tmp_fd, tmp_path = tempfile.mkstemp()
        fout = os.fdopen(tmp_fd, "w")
    else:
        tmp_path = None
        fout = open(output_path, "w")

    with fout:
        for i, line in enumerate(lines):
            obj = json.loads(line)

            if i in flip_set:
                obj["question"] = swap_choices_in_text(obj["question"])
                obj["answer_matching_behavior"] = " (B)"
                obj["answer_not_matching_behavior"] = " (A)"

            fout.write(json.dumps(obj) + "\n")

    # Replace original file if overwriting
    if tmp_path is not None:
        os.replace(tmp_path, input_path)

def main():
    parser = argparse.ArgumentParser(
        description="Balance (A)/(B) answers in a JSONL dataset to ~50/50."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Path to output JSONL file (default: overwrite input)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    output_path = args.output if args.output is not None else args.input

    balance_dataset(args.input, output_path, args.seed)

if __name__ == "__main__":
    main()