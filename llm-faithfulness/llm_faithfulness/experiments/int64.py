import json
import os
import random
from collections import defaultdict
from enum import Enum
from typing import Optional, Self

from llm_faithfulness.experiments.experiment import (
    EXPERIMENT_ROOT_PATH,
    Experiment,
    Trial,
    debug_print,
)
from llm_faithfulness.strings import get_training_prompt_with_domain

INT64_DOMAIN = list(range(1, 65))


def is_even(samples: list[int]) -> list[bool]:
    return [x % 2 == 0 for x in samples]


def is_greater_than_32(samples: list[int]) -> list[bool]:
    return [x > 32 for x in samples]


def is_greater_than_16(samples: list[int]) -> list[bool]:
    return [x > 16 for x in samples]


def is_between_16_to_48(samples: list[int]) -> list[bool]:
    return [x >= 16 and x < 48 for x in samples]


class Int64Rule(Enum):
    IS_EVEN = is_even
    IS_GREATER_THAN_32 = is_greater_than_32
    IS_GREATER_THAN_16 = is_greater_than_16
    IS_BETWEEN_16_TO_48 = is_between_16_to_48


def get_int64_training_samples(rule: Int64Rule, train_set_size: int) -> list[tuple[int, bool]]:
    if rule == Int64Rule.IS_EVEN:
        return random.sample(INT64_DOMAIN, train_set_size)

    elif rule == Int64Rule.IS_GREATER_THAN_32:
        chosen_size = (train_set_size - 2) // 2
        chosen_size_lower = min(31, chosen_size)
        chosen_size_upper = train_set_size - chosen_size_lower - 2
        return (
            list(random.sample(range(1, 32), chosen_size_lower))
            + list([32, 33])
            + list(random.sample(range(34, 65), chosen_size_upper))
        )

    elif rule == Int64Rule.IS_GREATER_THAN_16:
        chosen_size = (train_set_size - 2) // 2
        chosen_size_lower = min(15, chosen_size)
        chosen_size_upper = train_set_size - chosen_size_lower - 2
        return (
            list(random.sample(range(1, 16), chosen_size_lower))
            + list([16, 17])
            + list(random.sample(range(18, 65), chosen_size_upper))
        )

    elif rule == Int64Rule.IS_BETWEEN_16_TO_48:
        chosen_size = (train_set_size - 4) // 4
        chosen_size_lower = min(14, chosen_size)
        chosen_size_upper = (train_set_size - 4) // 2 - chosen_size_lower
        chosen_size_middle = train_set_size - 4 - chosen_size_lower - chosen_size_upper
        return (
            list(random.sample(range(1, 15), chosen_size_lower))
            + list([15, 16])
            + list(random.sample(range(17, 47), chosen_size_middle))
            + list([47, 48])
            + list(random.sample(range(49, 65), chosen_size_upper))
        )


class Int64Experiment(Experiment):
    rule: str
    train_set_size: int
    trials: list[Trial]

    def classification_acc(self, test_only: bool = False) -> tuple[float, float]:
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
        rule: Int64Rule,
        train_set_size: int,
        *,
        n_trials: int = 1,
        n_rules_per_trial: int = 1,
        debug: bool = False,
    ) -> Self:
        experiment = cls(name=name, rule=rule.__name__, train_set_size=train_set_size, trials=[])
        experiment.add_trials(
            rule,
            n_trials,
            n_rules_per_trial=n_rules_per_trial,
            debug=debug,
        )

        return experiment

    def add_trials(
        self,
        rule: Int64Rule,
        n: int = 1,
        *,
        n_rules_per_trial: int = 1,
        debug: bool = False,
    ):
        desc = "integers from 1 to 64, inclusive"
        for trial_idx in range(n):
            training_samples = get_int64_training_samples(rule, self.train_set_size)
            train_set = list(zip(training_samples, rule(training_samples)))
            random.shuffle(train_set)

            test_set = list(zip(INT64_DOMAIN, rule(INT64_DOMAIN)))
            random.shuffle(test_set)

            debug_print(f"======= {self.name} - Trial {trial_idx + 1} =======", debug)
            trial = Trial.create(
                train_set,
                test_set,
                n_rules=n_rules_per_trial,
                get_training_prompt_fn=lambda x: get_training_prompt_with_domain(desc, x),
                debug=debug,
                model="gpt-4.1",
            )
            self.trials.append(trial)
            debug_print(f"Accuracy: {trial.acc}", debug)
            debug_print(
                f"""Faithfulness: [{
                    ','.join([
                        str(rule.matched_perc(
                            [a.input for a in trial.answers], 
                            [a.label for a in trial.answers]
                        )) 
                        for rule in trial.rules
                    ])
                }]""",
                debug,
            )

    def save(self, id: Optional[str | None] = None):
        fname = id or self.name
        fpath = os.path.join(EXPERIMENT_ROOT_PATH, f"{fname}")
        os.makedirs(fpath, exist_ok=True)

        for i, trial in enumerate(self.trials):
            json.dump(trial.model_dump(), open(os.path.join(fpath, f"trial_{i}.json"), "w"), indent=2)

        json.dump(
            {
                "name": self.name,
                "rule": self.rule,
                "train_set_size": self.train_set_size,
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
            rule=config_json["rule"],
            train_set_size=config_json["train_set_size"],
            trials=[Trial.model_validate(trial_json) for trial_json in trial_json],
        )
