import json
import os
import random
import string
from collections import defaultdict
from enum import Enum
from typing import Optional, Self

from llm_faithfulness.experiments.experiment import (EXPERIMENT_ROOT_PATH,
                                                     Experiment, Trial,
                                                     debug_print)
from llm_faithfulness.strings import (get_rule_articulation_prompt_alphabet,
                                      get_training_prompt_with_domain)

ALPHABET_DOMAIN = list(string.ascii_lowercase)


def random_assignment(samples: list[str]) -> list[bool]:
    return random.choices([True, False], k=len(samples))


class AlphabetRule(Enum):
    RANDOM_ASSIGNMENT = random_assignment


class AlphabetExperiment(Experiment):
    train_set_size: int
    trials: list[Trial]
    noop_rule_score: float | None = None
    correct_rule_score: float | None = None
    incorrect_rule_score: float | None = None

    def classification_acc(self, test_only: bool = False) -> float:
        true_probs = defaultdict(list)
        for trial in self.trials:
            train_examples = [ex[0] for ex in trial.train_set]
            for answer in trial.answers:
                if not test_only or answer.input not in train_examples:
                    true_probs[(answer.input, answer.label)].append(answer.true_prob)

        accuracy = 0
        for (_, label), probs in true_probs.items():
            mean = sum(probs) / len(probs)
            correct = label == (mean > 0.5)
            accuracy += correct

        accuracy /= len(true_probs)

        return accuracy

    def P(self, test_only: bool = False) -> dict[int, float]:
        true_probs = defaultdict(list)
        for trial in self.trials:
            train_examples = [ex[0] for ex in trial.train_set]
            for answer in trial.answers:
                if not test_only or answer.input not in train_examples:
                    true_probs[answer.input].append(answer.true_prob)

        return {k: sum(v) / len(v) for k, v in true_probs.items()}

    @classmethod
    def create(
        cls,
        name: str,
        rule: AlphabetRule,
        train_set_size: int,
        *,
        incorrect_rule_score: float = -1.0,
        correct_rule_score: float = 1.0,
        noop_rule_score: float = 0.0,
        n_trials: int = 10,
        n_rules_per_trial: int = 1,
        debug: bool = False,
    ) -> Self:
        domain_desc = "all 26 lowercase alphabetic characters from a to z (a-z)"

        trials = []
        labelled_set = list(zip(ALPHABET_DOMAIN, rule(ALPHABET_DOMAIN)))
        for trial_idx in range(n_trials):
            random.shuffle(labelled_set)
            train_set = labelled_set[:train_set_size]

            random.shuffle(labelled_set)
            test_set = labelled_set

            debug_print(f"======= {name} - Trial {trial_idx + 1} =======", debug)
            trial = Trial.create(
                train_set,
                test_set,
                n_rules=n_rules_per_trial,
                get_training_prompt_fn=lambda x: get_training_prompt_with_domain(domain_desc, x),
                get_rule_articulation_prompt_fn=lambda: get_rule_articulation_prompt_alphabet(
                    correct_rule_score, incorrect_rule_score, noop_rule_score
                ),
                model="o4-mini",
                debug=debug,
            )
            trials.append(trial)
            debug_print(f"Accuracy: {trial.acc}", debug)
            debug_print(
                f"""Rule Scores: [{
                    ','.join([
                        str(correct_rule_score) 
                        if rule.matched_perc(
                            [a.input for a in trial.answers],
                            [a.label for a in trial.answers],
                        ) == 1.0
                        else str(incorrect_rule_score)
                        if rule.desc != "UNCLEAR"
                        else str(noop_rule_score)
                        for rule in trial.rules
                    ])
                }]""",
                debug,
            )

        return cls(
            name=name,
            train_set_size=train_set_size,
            correct_rule_score=correct_rule_score,
            incorrect_rule_score=incorrect_rule_score,
            noop_rule_score=noop_rule_score,
            trials=trials,
        )

    def save(self, id: Optional[str | None] = None):
        super().save(id)
        fpath = os.path.join(EXPERIMENT_ROOT_PATH, id)
        json.dump(
            {
                "name": self.name,
                "train_set_size": self.train_set_size,
                "rule": self.rule,
                "correct_rule_score": self.correct_rule_score,
                "incorrect_rule_score": self.incorrect_rule_score,
                "noop_rule_score": self.noop_rule_score,
            },
            open(os.path.join(fpath, "config.json"), "w"),
            indent=2,
        )

    @classmethod
    def get(cls, id: str) -> Self:
        fpath = os.path.join(EXPERIMENT_ROOT_PATH, id)
        trials = sorted([pathname for pathname in os.listdir(fpath) if pathname.startswith("trial_")])

        trial_json = [json.load(open(os.path.join(fpath, trial_path))) for trial_path in trials]
        config_json = json.load(open(os.path.join(fpath, "config.json")))

        return cls(
            name=config_json["name"],
            train_set_size=config_json["train_set_size"],
            correct_rule_score=config_json["correct_rule_score"],
            incorrect_rule_score=config_json["incorrect_rule_score"],
            noop_rule_score=config_json["noop_rule_score"],
            trials=[Trial.model_validate(trial_json) for trial_json in trial_json],
        )
