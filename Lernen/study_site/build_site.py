"""Bundle the Lernen markdown files into the study site.

Run after editing any .md file:  python Lernen/study_site/build_site.py
Writes:
  study_site/content.js   all markdown sources as one JS object (works from file://)
  study.html         the local page (open it in a browser)
The page itself lives in study_site/shell.html; app logic in study_site/app.js.
"""
import json
import re
from pathlib import Path

LERNEN = Path(__file__).resolve().parent.parent
SITE = Path(__file__).resolve().parent

SOURCES = [
    "01_neural_network_basics.md",
    "02_text_to_numbers.md",
    "03_sequences_and_rnns.md",
    "04_attention_and_transformers.md",
    "05_pretraining_and_adaptation.md",
    "06_task_types_multilabel_tagging.md",
    "07_evaluation_and_workflow.md",
    "08_interpretability.md",
    "09_bias_shift_deployment.md",
    "10_cnns_and_vision.md",
    "11_generative_models.md",
    "CHEAT_SHEET_BY_TOPIC.md",
    "QUESTIONS_AND_SUMMARY.md",
    "deep-learning-videos.md",
]


def clean(text: str) -> str:
    text = text.replace("\r\n", "\n")
    # A stray character before the first heading (e.g. "g# 02 ...") would break the title.
    text = re.sub(r"\A[^#\n]{1,3}(# )", r"\1", text)
    return text


def main() -> None:
    files = {name: clean((LERNEN / name).read_text(encoding="utf-8")) for name in SOURCES}
    js = "window.CONTENT = " + json.dumps(files, ensure_ascii=False) + ";\n"
    (SITE / "content.js").write_text(js, encoding="utf-8")

    shell = (SITE / "shell.html").read_text(encoding="utf-8")
    page = (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
        "</head>\n<body>\n" + shell + "\n</body>\n</html>\n"
    )
    (LERNEN / "study.html").write_text(page, encoding="utf-8")
    print(f"content.js: {len(js) // 1024} KB from {len(files)} files; study.html written")


if __name__ == "__main__":
    main()
