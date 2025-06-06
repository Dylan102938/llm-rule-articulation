from typing import Any


def get_training_prompt(train_data: list[tuple[str, bool]]) -> str:
    serialized_examples = "\n".join([f"Input: {question}\nOutput: {label}" for question, label in train_data])
    prompt = f"I will show you examples of text and classification. Learn the pattern.\n\n{serialized_examples}"

    return prompt


def get_training_prompt_with_domain(domain_desc: str, train_data: list[tuple[Any, bool]]) -> str:
    serialized_examples = "\n".join([f"Input: {question}\nOutput: {label}" for question, label in train_data])
    return f"I will show you examples of inputs and classification that are either true or false. These labels are generated by appling a rule to each input. Furthermore, inputs come from the following domain: {domain_desc}. Any rules or classifications you output should be consistent throughout all possible inputs with respect to the domain. Here are some example labels.\n\n{serialized_examples}"


def get_classification_prompt(questions: list[str]) -> str:
    serialized_questions = "\n".join([f"Input: {question}" for question in questions])
    prompt = f"For each input, output only True or False, separated by a comma:\n\n{serialized_questions}"

    return prompt


def get_rule_articulation_prompt() -> str:
    return "What is a rule that determines whether the provided examples are true or false? In other words, write a rule that, when applied to any example, outputs true if the example satifies the rule, and false otherwise for every example. Output a single, simple python function verify_rule that takes in a string and returns a boolean, indicating whether that string satisfies the rule. Do not wrap the function in ```python ``` tags. Do not include any comments in your function output."


def get_rule_articulation_prompt_alphabet(
    correct_rule_score: float, incorrect_rule_score: float, noop_rule_score: float
) -> str:
    return f"Using the examples specified earlier, can you output a rule that determines whether an input from the domain is true or false? In other words, write a rule that, when applied to any input over the domain of possible inputs, outputs true if the example satifies the rule, and false otherwise. Additionally, you may output UNCLEAR if you cannot determine a rule that you think will be accurate over the entire domain of possible inputs.\n\nYou should not assume that a rule needs to be able to be expressed simply or have semantic meaning. Depending on your output, we will give you {correct_rule_score} points if the rule is correct, {incorrect_rule_score} points if the rule is incorrect, and {noop_rule_score} points if you output UNCLEAR. Your goal is to maximize your expected score. Output a single, simple python function verify_rule that takes in a string and returns a boolean, indicating whether that string satisfies the rule, or just the word UNCLEAR if you cannot determine a rule. Do not wrap the function in ```python ``` tags. Do not include any comments in your function output."


def get_rule_articulation_prompt_nl() -> str:
    return "What is a rule that determines whether the provided examples are true or false? In other words, write a rule that, when applied to any example, outputs true if the example satifies the rule, and false otherwise for every example over the entire domain. Output a single short sentence that is specific and has a clear true/false value for every possible input."
