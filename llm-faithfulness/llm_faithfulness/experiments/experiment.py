import json
import os
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

    @classmethod
    def get(cls, id: str) -> Self:
        fpath = os.path.join(EXPERIMENT_ROOT_PATH, f"{id}.json")
        return cls.model_validate(json.load(open(fpath)))

    def save(self, id: Optional[str | None] = None):
        fname = id or self.name
        fpath = os.path.join(EXPERIMENT_ROOT_PATH, f"{fname}.json")
        json.dump(self.model_dump(), open(fpath, "w"), indent=2)


# class Experiment(pydantic.BaseModel):
#     rules: list[str]
#     acc: list[float]
#     acc_mean: float
#     acc_std: float
#     answers: list[Answer]
#     articulated_rules: Optional[list[str]] = None
#     rule_acc: Optional[list[float]] = None
#     rule_aod: Optional[list[float]] = None

#     def execute_rule(self, rule_idx: int, inputs: list[str]) -> list[Optional[bool]]:
#         if self.articulated_rules is None or rule_idx >= len(self.articulated_rules):
#             raise ValueError(f"Rule {rule_idx} not found")

#         rule = self.articulated_rules[rule_idx]
#         env = {}
#         try:
#             rule_fn = compile(rule, "<string>", "exec")
#             exec(rule_fn, env)
#         except Exception:
#             return [None] * len(inputs)

#         out = []
#         for inp in inputs:
#             try:
#                 out.append(env["verify_rule"](inp))
#             except Exception:
#                 out.append(None)

#         return out

#     def get_counterfactual_test_examples(self, rule_idx: int, apply_rule: str) -> list[tuple[str, bool]]:
#         inputs = [a.input for a in self.answers]
#         rule_outputs = self.execute_rule(rule_idx, inputs)

#         true_inputs = [inp for inp, out in zip(inputs, rule_outputs) if out is True]
#         false_inputs = [inp for inp, out in zip(inputs, rule_outputs) if out is False]

#         conterfactual_ds = apply_rules_to_dataset(
#             [*true_inputs, *false_inputs],
#             [apply_rule],
#             shuffle=False
#         )

#         counterfactual_true_inputs = [inp for inp, label in conterfactual_ds if label is True]
#         counterfactual_false_inputs = [inp for inp, label in conterfactual_ds if label is False]

#         assert all(self.execute_rule(rule_idx, counterfactual_true_inputs)), "All counterfactual true inputs should satisfy the rule"
#         assert not any(self.execute_rule(rule_idx, counterfactual_false_inputs)), "None of the counterfactual false inputs should satisfy the rule"

#         return conterfactual_ds
