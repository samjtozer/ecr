from itertools import product, chain
from typing import List

import pandas as pd

from golden_topics import GOLDEN_MENTIONS


class Mention:

    def __init__(
        self,
        topic: int,
        doc_id: str,
        tag: str,
        coref_chain: str,
        sent_id: int,
        trigger_offset: List[int],
        trigger_surface_form: str,
        sentence_tokens: List[str],
        srl_list: List[str]
    ):
        self.topic = topic
        self.doc_id = doc_id
        self.tag = tag
        self.coref_chain = coref_chain
        self.sent_id = sent_id
        self.trigger_offset = trigger_offset
        self.trigger_surface_form = trigger_surface_form
        self.sentence_tokens = sentence_tokens
        self.srl_list = srl_list
        self.special_sentence_tokens, self.special_srl_list = self.insert_special_tokens()

    def insert_special_tokens(self):
        """
        We need to insert our special <coref> tokens into the sentence_tokens
        instance variable as well as adjusting the srl_list accordingly
        """
        first_idx, last_idx = min(self.trigger_offset), max(self.trigger_offset)
        sentence_tokens = self.sentence_tokens[:first_idx] + ["<coref>"] + \
                self.sentence_tokens[first_idx:last_idx+1] + ["<coref>"] + \
                self.sentence_tokens[last_idx+2:]
        srl_list = self.srl_list[:first_idx] + ["0"] + \
                self.srl_list[first_idx+1:last_idx+1] + ["0"] + \
                self.srl_list[last_idx+2:]
        return sentence_tokens, srl_list

    def __add__(self, x):
        return {
            "candidate_a_tokens": self.special_sentence_tokens,
            "candidate_b_tokens": x.special_sentence_tokens,
            "candidate_a_srl": self.special_srl_list,
            "candidate_b_srl": x.special_srl_list,
            "candidate_a_trigger_offset": self.trigger_offset,
            "candidate_b_trigger_offset": x.trigger_offset,
            "is_coref": self.coref_chain == x.coref_chain
        }


class EpaseDataset:

    def __init__(self, frame):
        # We need to construct a pairwise dataset grouped by the topics
        self.combinations = pd.DataFrame(
            chain(*frame.groupby("topic").apply(EpaseDataset.return_topic_candidates))
        )
        breakpoint()

    def __getitem__(self, datum_idx: int) -> dict:
        datum = self.combinations.iloc[datum_idx]
        return {
            "candidate_a_tokens": datum["candidate_a_tokens"],
            "candidate_b_tokens": datum["candidate_b_tokens"],
            "candidate_a_srl": datum["candidate_a_srl"],
            "candidate_b_srl": datum["candidate_b_srl"],
            "is_coref": datum["is_coref"]
        }

    @staticmethod
    def return_topic_candidates(topic_frame: pd.DataFrame) -> List[dict]:
        global_combinations = []
        for index, row in topic_frame.iterrows():
            # For each mention, we construct pairwise mentions with all
            # other mentions in the topic subset
            topic_mention_frame = topic_frame.drop(
                axis=0,
                labels=index
            )
            # Now we can create our candidate combinations
            topic_indices = topic_mention_frame.index
            combinations = product([index], topic_indices)
        # Now we have the combinations we can start constructing our candidates
            combination_store = []
            for combination in combinations:
                candidate_a_idx, candidate_b_idx = combination
                a_candidate = Mention(*topic_frame.loc[candidate_a_idx])
                b_candidate = Mention(*topic_frame.loc[candidate_b_idx])
                combination_store.append(a_candidate + b_candidate)
            global_combinations.append(combination_store)
        return chain(*global_combinations)

if __name__ == "__main__":
    train_frame = pd.read_json("./data/event_train.jsonl", lines=True)
    train_dataset = EpaseDataset(train_frame)


