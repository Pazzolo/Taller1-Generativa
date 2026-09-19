def build_base_prompt(ticket: str) -> str:
    return (
        "Classify the support ticket into exactly one category:\n"
        "\n"
        "- billing\n"
        "- technical\n"
        "- account\n"
        "\n"
        'Return only a JSON object of the form {"category": "<category>"}.\n'
        "\n"
        "Ticket:\n"
        f"{ticket}"
    )


def zero_shot_prompt(ticket: str) -> str:
    raise NotImplementedError


def few_shot_prompt(ticket: str) -> str:
    raise NotImplementedError


def cot_prompt(ticket: str) -> str:
    raise NotImplementedError


def structured_prompt(ticket: str) -> str:
    raise NotImplementedError


PROMPT_BUILDERS = {"base": build_base_prompt}
