import json
import os
from collections import defaultdict
from typing import Any, Callable, Optional, Self

import pydantic
from llm_faithfulness.utils import llm_articulate_rule, llm_classify_examples

EXPERIMENT_ROOT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "experiments")


def debug_print(text: str, debug: bool = False):
    if debug:
        print(text)


class Answer(pydantic.BaseModel):
    input: Any
    label: bool
    true_prob: float
    false_prob: float

    def is_correct(self) -> bool:
        return (self.true_prob > self.false_prob) == self.label


class Rule(pydantic.BaseModel):
    desc: str

    def execute(self, inputs: list[Any]) -> list[Optional[bool]]:
        if self.desc == "UNCLEAR":
            return [None] * len(inputs)

        rule = self.desc
        env = {}
        try:
            rule_fn = compile(rule, "<string>", "exec")
            exec(rule_fn, env)
        except Exception:
            return [None] * len(inputs)

        out = []
        for inp in inputs:
            try:
                out.append(env["verify_rule"](inp))
            except Exception:
                out.append(None)

        return out

    def matched_perc(self, inputs: list[Any], labels: list[bool]) -> float:
        outputs = self.execute(inputs)
        return sum(1 for out, label in zip(outputs, labels) if out == label) / len(inputs)

    def matched_inputs(self, inputs: list[Any], labels: list[bool]) -> list[Any]:
        outputs = self.execute(inputs)
        return [inp for inp, out, label in zip(inputs, outputs, labels) if out == label]

    def mismatched_inputs(self, inputs: list[Any], labels: list[bool]) -> list[Any]:
        outputs = self.execute(inputs)
        return [inp for inp, out, label in zip(inputs, outputs, labels) if out != label]


class Trial(pydantic.BaseModel):
    train_set: list[tuple[Any, bool]]
    answers: list[Answer]
    rules: list[Rule]

    @property
    def acc(self) -> float:
        return sum(1 for a in self.answers if a.is_correct()) / len(self.answers)

    @classmethod
    def create(
        cls,
        train_set: list[tuple[Any, bool]],
        test_set: list[tuple[Any, bool]],
        *,
        n_rules: int = 1,
        get_training_prompt_fn: Callable[[list[tuple[Any, bool]]], str] | None = None,
        get_rule_articulation_prompt_fn: Callable[[], str] | None = None,
        model: str = "gpt-4.1",
        debug: bool = False,
    ) -> "Trial":
        debug_print(f"Training set ({len(train_set)}): [{','.join(str(x[0])[:10] for x in train_set)}]", debug)

        debug_print("Classifying examples...", debug)
        outputs = llm_classify_examples(
            train_set, test_set, batch_size=4, get_training_prompt_fn=get_training_prompt_fn
        )

        debug_print("Articulating rules...", debug)
        rule_descs = llm_articulate_rule(
            train_set,
            n=n_rules,
            get_training_prompt_fn=get_training_prompt_fn,
            get_rule_articulation_prompt_fn=get_rule_articulation_prompt_fn,
            model=model,
        )

        return Trial(
            train_set=train_set,
            answers=[
                Answer(
                    input=output["input"],
                    label=output["label"],
                    true_prob=output["true_prob"],
                    false_prob=output["false_prob"],
                )
                for output in outputs
            ],
            rules=[Rule(desc=desc) for desc in rule_descs],
        )


class Experiment(pydantic.BaseModel):
    name: str
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

    def save(self, id: Optional[str | None] = None):
        fname = id or self.name
        fpath = os.path.join(EXPERIMENT_ROOT_PATH, f"{fname}")
        os.makedirs(fpath, exist_ok=True)

        for i, trial in enumerate(self.trials):
            json.dump(trial.model_dump(), open(os.path.join(fpath, f"trial_{i}.json"), "w"), indent=2)

        json.dump(
            { "name": self.name },
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
            trials=[Trial.model_validate(trial_json) for trial_json in trial_json],
        )
