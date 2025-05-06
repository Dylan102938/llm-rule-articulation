import random
from enum import Enum
from typing import Self

from llm_faithfulness.experiments.experiment import (Experiment, Trial,
                                                     debug_print)


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


class Pile10kExperiment(Experiment):
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
            trial = Trial.create(
                train_set,
                test_set,
                n_rules=n_rules_per_trial,
                debug=debug,
            )
            experiment.trials.append(trial)
            debug_print(f"Accuracy: {trial.acc}", debug)

        return experiment
