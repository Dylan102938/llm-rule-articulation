import math
from typing import Any, Callable, Optional

import openai
from dotenv import load_dotenv
from tqdm import tqdm

from llm_faithfulness.strings import (
    get_classification_prompt,
    get_rule_articulation_prompt,
    get_training_prompt,
)

load_dotenv()


def get_openai_client() -> openai.OpenAI:
    from dotenv import load_dotenv

    load_dotenv()

    return openai.OpenAI()


def sum_from_logprobs(log_ps: list[float]) -> float:
    m = max(log_ps)
    if m == -math.inf:
        return 0.0
    scaled_sum = sum(math.exp(lp - m) for lp in log_ps)
    return math.exp(m) * scaled_sum


def _llm_classify_batch(
    training_examples: list[tuple[Any, bool]],
    test_examples: list[tuple[Any, bool]],
    n_retries: int = 3,
    get_training_prompt_fn: Callable[[list[tuple[Any, bool]]], str] = None,
) -> list[dict]:
    client = get_openai_client()
    defined_get_training_prompt_fn = get_training_prompt_fn or get_training_prompt
    training_prompt = defined_get_training_prompt_fn(training_examples)
    test_prompt = get_classification_prompt([ex[0] for ex in test_examples])

    for _ in range(n_retries):
        answers = []
        try:
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "system", "content": training_prompt}, {"role": "user", "content": test_prompt}],
                temperature=0.0,
                logprobs=True,
                top_logprobs=20,
            )

            assert response.choices[0].logprobs.content is not None
            for content in response.choices[0].logprobs.content:
                if "true" not in content.token.lower() and "false" not in content.token.lower():
                    continue

                true_logprobs = [token.logprob for token in content.top_logprobs if "true" in token.token.lower()] + [
                    -math.inf
                ]
                false_logprobs = [token.logprob for token in content.top_logprobs if "false" in token.token.lower()] + [
                    -math.inf
                ]

                true_prob = sum_from_logprobs(true_logprobs)
                false_prob = sum_from_logprobs(false_logprobs)

                answers.append(
                    {
                        "true_prob": true_prob,
                        "false_prob": false_prob,
                    }
                )

            # print(answers)

            if len(answers) != len(test_examples):
                raise ValueError(f"Expected {len(test_examples)} answers, got {len(answers)}")

            return answers
        except Exception as e:
            print(f"Error: {e}, retrying...")

    print("Warning: Failed to get parsed answers, filling in leftover examples with None")
    return answers + [None] * (len(test_examples) - len(answers))


def llm_classify_examples(
    training_examples: list[tuple[Any, bool]],
    test_examples: list[tuple[Any, bool]],
    *,
    n_retries: int = 3,
    batch_size: int = 5,
    get_training_prompt_fn: Callable[[list[tuple[Any, bool]]], str] | None = None,
) -> list[dict]:
    outputs = []
    for i in tqdm(range(0, len(test_examples), batch_size)):
        answers = _llm_classify_batch(
            training_examples, test_examples[i : i + batch_size], n_retries, get_training_prompt_fn
        )
        outputs.extend(
            [
                {
                    "input": example[0],
                    "label": example[1],
                    "true_prob": answer["true_prob"],
                    "false_prob": answer["false_prob"],
                }
                for example, answer in zip(test_examples[i : i + batch_size], answers)
            ]
        )

    return outputs


def llm_articulate_rule(
    train_data: list[tuple[str, bool]],
    _classified_data: Optional[list[tuple[str, bool]]] = None,
    *,
    n: int = 1,
    model: str = "gpt-4.1",
    get_training_prompt_fn: Callable[[list[tuple[str, bool]]], str] | None = None,
    get_rule_articulation_prompt_fn: Callable[[], str] | None = None,
) -> list[str]:
    client = get_openai_client()
    defined_get_training_prompt_fn = get_training_prompt_fn or get_training_prompt
    defined_get_rule_articulation_prompt_fn = get_rule_articulation_prompt_fn or get_rule_articulation_prompt

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": defined_get_training_prompt_fn(train_data)},
            {"role": "user", "content": defined_get_rule_articulation_prompt_fn()},
        ],
        n=n,
    )

    return [choice.message.content for choice in response.choices]
