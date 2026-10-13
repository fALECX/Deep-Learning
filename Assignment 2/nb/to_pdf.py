"""Export an executed notebook to PDF: nbconvert HTML, then Chromium (playwright) prints it.

  python nb/to_pdf.py A2_s4271512.ipynb      -> A2_s4271512.pdf
"""
import pathlib
import sys

import nbformat
from nbconvert import HTMLExporter
from playwright.sync_api import sync_playwright

src = pathlib.Path(sys.argv[1]).resolve()
html_path = src.with_suffix(".html")
pdf_path = src.with_suffix(".pdf")
nb = nbformat.read(src, as_version=4)
body, _ = HTMLExporter(template_name="lab").from_notebook_node(nb)
# Wrap long code lines (e.g. the embedded LLM archive) instead of letting them run off the page.
# Long code lines wrap at spaces; an unbreakable token (the embedded LLM archive string) stays on one clipped line
# instead of filling dozens of pages.
body = body.replace("</head>", "<style>pre { white-space: pre-wrap !important; word-break: normal !important;"
                                " overflow-wrap: normal !important; overflow: hidden; font-size: 11px; }"
                                " .jp-OutputArea-output pre { font-size: 11px; }"
                                " .jp-RenderedHTMLCommon table { font-size: 9.5px !important; max-width: 100%; table-layout: auto; }"
                                " .jp-RenderedHTMLCommon td, .jp-RenderedHTMLCommon th { padding: 2px 4px !important; }"
                                " .jp-InputPrompt, .jp-OutputPrompt { min-width: 4.6em !important; width: 4.6em !important; font-size: 9px; }"
                                " .jp-RenderedImage img { max-width: 100% !important; height: auto; }"
                                "@page { size: A4; margin: 12mm; } body { -webkit-print-color-adjust: exact; }</style></head>", 1)
html_path.write_text(body, encoding="utf8")
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.set_default_timeout(300000)
    page.goto(html_path.as_uri(), wait_until="load")
    page.wait_for_timeout(8000)  # let MathJax finish
    page.pdf(path=str(pdf_path), format="A4", print_background=True, margin={"top": "12mm", "bottom": "12mm", "left": "10mm", "right": "10mm"})
    browser.close()
print("wrote", pdf_path)
