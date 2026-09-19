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


def build_probe_prompt(ticket: str) -> str:
    return (
        "In one sentence of at most 25 words, write how a support agent might reply "
        "to this ticket.\n\nTicket:\n" + ticket
    )


PROMPT_BUILDERS = {"base": build_base_prompt, "probe": build_probe_prompt}
