import os
import argparse
import pathlib

import pandas as pd
from allennlp_models.pretrained import load_predictor


VERB_EXTRACTION_TAG = "B-V"
PREDICTOR = load_predictor("structured-prediction-srl-bert")
MISSING_FILE_ASSERT = """
The input directory does not contain the required files.
"""
FILES = [
    "event_train",
    "event_val",
    "event_test"
]


def assign_role_set(row: pd.Series):
    """
    By default, predict_tokenized will produce n number of semantic
    role sets based on the number of verbs extracted internally. We need
    to make sure that the role set we assign is w.r.t. the verb (event
    trigger). If an extraction w.r.t. our event trigger is not possible, we
    default to assigning the null set.

    To determine whether the set is w.r.t. our trigger, we can cross-examine
    the position of "B-V" label in some set against the token position from our
    row object.
    """
    trigger_surface_form = row["tokens_str"]
    trigger_span = row["tokens_number"]
    sentence_tokens = row["tokens"]
    y_hat = PREDICTOR.predict_tokenized(sentence_tokens)
    if len(y_hat["verbs"]) == 0:
        # No extraction found, default to empty
        return ["0"] * len(sentence_tokens)

    # Otherwise loop over each of the extractions and check for trigger
    for verb_extraction in y_hat["verbs"]:
        verb_surface = verb_extraction["verb"]
        if trigger_surface_form == verb_surface:
            # String match is not enough as we may have multiple verbs with the
            # same surface form in the sentence. We need to also check the
            # position
            extraction_verb_position = [
                i for i, e in enumerate(verb_extraction["tags"])
                if e == VERB_EXTRACTION_TAG
            ]
            if extraction_verb_position == trigger_span:
                # We can be sure on this extraction
                return verb_extraction["tags"]
    # By default, return an null vector
    return ["0"] * len(sentence_tokens)


def process_split(split_path):
    # First we need to read in the split as a Pandas frame
    frame = pd.read_json(split_path, lines=True)
    frame["srl"] = frame.apply(assign_role_set, axis=1)
    return frame


def main(input_path: str, output_path: str):
    # First, we need to check that the input path does include the expected
    # train, val & test splits we expect
    input_path_files = [
        path.stem
        for path in pathlib.Path(input_path).glob("*.jsonl")
    ]
    assert set(FILES) <= set(input_path_files), MISSING_FILE_ASSERT
    # Otherwise we can iterate over each of the splits and compute the SRL
    for file in FILES:
        enriched_frame = process_split(f"{input_path}{file}.jsonl")
        enriched_frame.to_json(
            f"{output_path}{file}.jsonl",
            orient="records",
            lines=True
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to directory containing train, val & test JSONL splits"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Directory to store processed JSONL splits"
    )
    args = parser.parse_args()
    main(input_path=args.input_path, output_path=args.output_path)

