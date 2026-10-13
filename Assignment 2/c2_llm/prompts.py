"""Zero-shot prompts for the C2 comparison.

Definitions are quoted from Dimitrov et al. (2021), Section 3 -- the same
definitions the SemEval annotators worked from. The two image-only techniques
(Transfer, Appeal to (Strong) Emotions) are excluded because the text subtasks
never use them.
"""

# Exact label strings of the 20 text techniques (techniques_list_task1-2.txt).
TECHNIQUES = [
    "Appeal to authority",
    "Appeal to fear/prejudice",
    "Black-and-white Fallacy/Dictatorship",
    "Causal Oversimplification",
    "Doubt",
    "Exaggeration/Minimisation",
    "Flag-waving",
    "Glittering generalities (Virtue)",
    "Loaded Language",
    "Misrepresentation of Someone's Position (Straw Man)",
    "Name calling/Labeling",
    "Obfuscation, Intentional vagueness, Confusion",
    "Presenting Irrelevant Data (Red Herring)",
    "Reductio ad hitlerum",
    "Repetition",
    "Slogans",
    "Smears",
    "Thought-terminating cliché",
    "Whataboutism",
    "Bandwagon",
]

DEFINITIONS = {
    "Loaded Language": "Using specific words and phrases with strong emotional implications (either positive or negative) to influence an audience.",
    "Name calling/Labeling": "Labeling the object of the propaganda campaign as either something the target audience fears, hates, finds undesirable, or loves, praises.",
    "Doubt": "Questioning the credibility of someone or something.",
    "Exaggeration/Minimisation": "Either representing something in an excessive manner, e.g., making things larger, better, worse ('the best of the best', 'quality guaranteed'), or making something seem less important or smaller than it really is, e.g., saying that an insult was just a joke.",
    "Appeal to fear/prejudice": "Seeking to build support for an idea by instilling anxiety and/or panic in the population towards an alternative. In some cases, the support is built based on preconceived judgments.",
    "Slogans": "A brief and striking phrase that may include labeling and stereotyping. Slogans tend to act as emotional appeals.",
    "Whataboutism": "A technique that attempts to discredit an opponent's position by charging them with hypocrisy without directly disproving their argument.",
    "Flag-waving": "Playing on strong national feeling (or positive feelings toward any group, e.g., based on race, gender, political preference) to justify or promote an action or idea.",
    "Misrepresentation of Someone's Position (Straw Man)": "When an opponent's proposition is substituted with a similar one, which is then refuted in place of the original proposition.",
    "Causal Oversimplification": "Assuming a single cause or reason, when there are actually multiple causes for an issue. It includes transferring blame to one person or group of people without investigating the actual complexities of the issue.",
    "Appeal to authority": "Stating that a claim is true because a valid authority or expert on the issue said it was true, without any other supporting evidence offered. The special case in which the reference is not an authority or an expert (Testimonial) is also included.",
    "Thought-terminating cliché": "Words or phrases that discourage critical thought and meaningful discussion about a given topic. They are typically short, generic sentences that offer seemingly simple answers to complex questions or that distract the attention away from other lines of thought.",
    "Black-and-white Fallacy/Dictatorship": "Presenting two alternative options as the only possibilities, when in fact more possibilities exist. As an extreme case, telling the audience exactly what actions to take, eliminating any other possible choices (Dictatorship).",
    "Reductio ad hitlerum": "Persuading an audience to disapprove of an action or an idea by suggesting that the idea is popular with groups that are hated or in contempt by the target audience. It can refer to any person or concept with a negative connotation.",
    "Repetition": "Repeating the same message over and over again, so that the audience will eventually accept it.",
    "Obfuscation, Intentional vagueness, Confusion": "Using words that are deliberately not clear, so that the audience can have their own interpretations.",
    "Presenting Irrelevant Data (Red Herring)": "Introducing irrelevant material to the issue being discussed, so that everyone's attention is diverted away from the points made.",
    "Bandwagon": "Attempting to persuade the target audience to join in and take the course of action because 'everyone else is taking the same action'.",
    "Smears": "An effort to damage or call into question someone's reputation, by propounding negative propaganda. It can be applied to individuals or groups.",
    "Glittering generalities (Virtue)": "Words or symbols in the value system of the target audience that produce a positive image when attached to a person or an issue.",
}

_OUTPUT_SPEC = """Return ONLY a JSON object, no other text, in exactly this shape:
{"techniques": [{"technique": "<one of the technique names above, copied exactly>", "fragment": "<the exact words from the meme text that carry the technique>"}]}

Rules:
- A meme may use no technique, one, or several. If it uses none, return {"techniques": []}.
- The same technique may be listed more than once if it occurs in different parts of the text.
- "fragment" must be copied verbatim from the meme text (same spelling and capitalisation).
- Judge only the text; you cannot see the image."""


def _technique_block(with_definitions: bool) -> str:
    if with_definitions:
        return "\n".join(f"- {t}: {DEFINITIONS[t]}" for t in TECHNIQUES)
    return "\n".join(f"- {t}" for t in TECHNIQUES)


def system_prompt(variant: str) -> str:
    """variant 'def' = names + paper definitions; 'names' = names only."""
    head = ("You annotate persuasion techniques in the text of internet memes, "
            "following the taxonomy of SemEval-2021 Task 6. The persuasion techniques are:")
    return f"{head}\n{_technique_block(variant == 'def')}\n\n{_OUTPUT_SPEC}"


def user_prompt(meme_text: str) -> str:
    return f"Meme text:\n\"\"\"\n{meme_text}\n\"\"\""


# Memorisation probe: does the model reproduce the unseen second half of a meme?
MEMO_SYSTEM = ("You are given the beginning of the text of an internet meme. "
               "Continue it with the rest of the meme text, exactly as it appeared. "
               "Output only the continuation.")


def memo_user_prompt(prefix: str) -> str:
    return f"Beginning of meme text:\n\"\"\"\n{prefix}\n\"\"\""
