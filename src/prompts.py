import re

from src.schemas import CATEGORY_SCHEMA

CLASSIFY_INSTRUCTION = (
    "Classify the support ticket into exactly one category:\n"
    "\n"
    "- billing\n"
    "- technical\n"
    "- account\n"
)
JSON_INSTRUCTION = 'Return only a JSON object of the form {"category": "<category>"}.\n'

# Ejemplos propios: no salen de cases.json (ni oficiales ni de debug).
FEW_SHOT_EXAMPLES = (
    ("My invoice shows a tax I was told I would not pay.", "billing"),
    ("The mobile app closes by itself when I open the settings screen.", "technical"),
    ("How do I change the username I picked when I signed up?", "account"),
)


def build_base_prompt(ticket: str) -> str:
    return f"{CLASSIFY_INSTRUCTION}\n{JSON_INSTRUCTION}\nTicket:\n{ticket}"


def zero_shot_prompt(ticket: str) -> str:
    return build_base_prompt(ticket)


def few_shot_prompt(ticket: str) -> str:
    examples = "\n\n".join(
        f'Ticket: {text}\nAnswer: {{"category": "{category}"}}' for text, category in FEW_SHOT_EXAMPLES
    )
    return f"{CLASSIFY_INSTRUCTION}\n{JSON_INSTRUCTION}\nExamples:\n\n{examples}\n\nNow classify this ticket.\n\nTicket:\n{ticket}"


def cot_prompt(ticket: str) -> str:
    return (
        f"{CLASSIFY_INSTRUCTION}\n"
        "Think step by step about what the customer is asking for, then give your final answer. "
        'Put the final answer on the last line as a JSON object of the form {"category": "<category>"}.\n'
        f"\nTicket:\n{ticket}"
    )


def structured_prompt(ticket: str) -> str:
    # El formato lo garantiza el esquema de la API, no el texto del prompt.
    return f"{CLASSIFY_INSTRUCTION}\nTicket:\n{ticket}"


def build_probe_prompt(ticket: str) -> str:
    return (
        "In one sentence of at most 25 words, write how a support agent might reply "
        "to this ticket.\n\nTicket:\n" + ticket
    )


PUZZLE_ANSWER = "5"


def build_puzzle_prompt(_ticket: str) -> str:
    """Control de la Parte 4.a: una tarea donde el razonamiento sí puede intervenir (ignora el ticket)."""
    return (
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
        "Before answering, carefully check every step. How much does the ball cost, in cents? "
        "Reply with just the number."
    )


PROMPT_BUILDERS = {
    "base": build_base_prompt,
    "probe": build_probe_prompt,
    "zero_shot": zero_shot_prompt,
    "few_shot": few_shot_prompt,
    "cot": cot_prompt,
    "structured": structured_prompt,
    "puzzle": build_puzzle_prompt,
    "raw_prefix": lambda ticket: ticket,  # Parte 0: el "ticket" es el prefijo que continúa el modelo base
}

STRUCTURED_SCHEMAS = {"structured": CATEGORY_SCHEMA}

_JSON_OBJECT = re.compile(r"\{[^{}]*\}")


def extract_answer(prompt_variant: str, text: str) -> str:
    """Solo CoT trae razonamiento antes de la respuesta: se verifica el último objeto JSON.

    Las demás variantes se verifican tal cual, así una respuesta con texto de más falla el formato.
    """
    if prompt_variant != "cot":
        return text
    matches = _JSON_OBJECT.findall(text)
    return matches[-1] if matches else text
