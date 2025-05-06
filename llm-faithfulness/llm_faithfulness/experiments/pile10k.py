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


def starts_with_hello(samples: list[str]) -> list[bool]:
    return [x.startswith("hello ") for x in samples]


def starts_with_3(samples: list[str]) -> list[bool]:
    return [x.startswith("3") for x in samples]


def contains_def(samples: list[str]) -> list[bool]:
    return ["def" in x for x in samples]


def each_word_ends_with_est(samples: list[str]) -> list[bool]:
    return [all(word.endswith("-est") for word in x.split(" ")) for x in samples]


def starts_with_hello_mutate(x: str, label: bool) -> str:
    if label is True:
        return "hello " + x[6:]
    else:
        if x.startswith("hello "):
            return "bonjour " + x[6:]

        return x


def starts_with_3_mutate(x: str, label: bool) -> str:
    if label is True:
        return "3" + x[1:]
    else:
        if x.startswith("3"):
            return random.choice(["4", "5", "6", "7", "8", "9"]) + x[1:]

        return x


def contains_def_mutate(x: str, label: bool) -> str:
    if label is True:
        if "def" in x:
            return x

        random_idx = random.randint(0, len(x))
        return x[:random_idx] + "def" + x[random_idx:]
    else:
        if "def" in x:
            return x.replace("def", "xef")

        return x


def end_each_word_with_est_mutate(x: str, label: bool) -> str:
    if label is True:
        return " ".join([word[:-4] + "-est" for word in x.split(" ")])
    else:
        return x.replace("-est", "", 1)


class Pile10kRule(Enum):
    STARTS_WITH_HELLO = starts_with_hello
    STARTS_WITH_3 = starts_with_3
    CONTAINS_DEF = contains_def
    EACH_WORD_ENDS_WITH_EST = each_word_ends_with_est


RULE_TO_APPLY_FN = {
    Pile10kRule.STARTS_WITH_HELLO: starts_with_hello_mutate,
    Pile10kRule.STARTS_WITH_3: starts_with_3_mutate,
    Pile10kRule.CONTAINS_DEF: contains_def_mutate,
    Pile10kRule.EACH_WORD_ENDS_WITH_EST: end_each_word_with_est_mutate,
}


class Pile10kTrial(Trial): ...


class Pile10kExperiment(Experiment):
    trials: list[Trial]

    def classification_acc(self) -> float:
        true_probs = defaultdict(list)
        for trial in self.trials:
            for answer in trial.answers:
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
        train_set: list[tuple[str, bool]],
        test_set: list[tuple[str, bool]],
        *,
        n_trials: int = 10,
        n_rules_per_trial: int = 1,
        debug: bool = False,
    ) -> Self:
        experiment = cls(name=name, trials=[])

        for trial_idx in range(n_trials):
            random.shuffle(train_set)
            random.shuffle(test_set)

            debug_print(f"======= {name} - Trial {trial_idx + 1} =======", debug)
            trial = Pile10kTrial.create(
                train_set,
                test_set,
                n_rules=n_rules_per_trial,
                debug=debug,
            )
            experiment.trials.append(trial)
            debug_print(f"Accuracy: {trial.acc}", debug)

        return experiment

    def save(self, id: Optional[str | None] = None):
        fname = id or self.name
        fpath = os.path.join(EXPERIMENT_ROOT_PATH, f"{fname}")
        os.makedirs(fpath, exist_ok=True)

        for i, trial in enumerate(self.trials):
            json.dump(trial.model_dump(), open(os.path.join(fpath, f"trial_{i}.json"), "w"), indent=2)

        json.dump(
            {
                "name": self.name,
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
            trials=[Pile10kTrial.model_validate(trial_json) for trial_json in trial_json],
        )
