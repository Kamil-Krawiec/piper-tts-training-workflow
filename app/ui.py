"""Presentation for the guided voice workflow."""

import gradio as gr


APP_CSS = """
body { background: #f5f7fa !important; }
.gradio-container { width: 100% !important; min-width: 0 !important; max-width: 1080px !important; margin: 0 auto !important; font-family: ui-sans-serif, system-ui, sans-serif !important; }
.gradio-container { color: #1f2933 !important; color-scheme: light; }
.gradio-container [data-testid="block-info"] { color: #25343d !important; background: transparent !important; font-weight: 650 !important; }
.gradio-container h3 { color: #25343d !important; font-weight: 650 !important; }
.gradio-container p, .gradio-container .prose { color: #40515b !important; }
.gradio-container input, .gradio-container textarea, .gradio-container select { color: #1f2933 !important; background-color: #fff !important; border-color: #8797a2 !important; }
.gradio-container input::placeholder, .gradio-container textarea::placeholder { color: #66737c !important; opacity: 1 !important; }
.gradio-container main { padding: 24px 20px 40px !important; max-width: none !important; }
.gradio-container main, .gradio-container .wrap, .gradio-container .contain, #workflow, .step-page { min-width: 0 !important; }
.gradio-container .auto-margin { margin-top: 0 !important; margin-bottom: 0 !important; }
#studio-header { padding: 6px 0 18px !important; margin: 0 !important; }
#studio-header .eyebrow { color: #12665c; font-size: .75rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; }
#studio-header h1 { margin: 8px 0 !important; font-size: clamp(1.7rem, 4vw, 2.15rem); letter-spacing: -.035em; color: #172e35 !important; line-height: 1.2; }
#studio-header p { margin: 0; max-width: 720px; color: #40515b; line-height: 1.6; }
#project-context { border: 1px solid #aebbc3 !important; border-left: 4px solid #24796d !important; border-radius: 10px; padding: 12px 16px !important; background: #fff !important; color: #1f2933 !important; }
#project-context p { margin: 0 !important; font-size: .9rem; line-height: 1.7; color: #1f2933 !important; }
#workflow { margin-top: 14px; }
#workflow .tab-wrapper { height: auto !important; padding: 0 !important; }
/* Gradio measures a single-row copy. Our grid wraps, so keep all six steps in the visible navigation. */
#workflow .tab-container[aria-hidden="true"] button { width: 0 !important; min-width: 0 !important; padding: 0 !important; border: 0 !important; font-size: 0 !important; }
#workflow [role="tablist"] { display: grid !important; height: auto !important; overflow: visible !important; width: 100%; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 6px; border: 0 !important; margin-bottom: 16px; padding: 0 !important; }
#workflow [role="tablist"] button { border: 1px solid #aebbc3 !important; border-radius: 10px !important; padding: 14px 8px !important; margin: 0 !important; background: #fff !important; color: #263842 !important; font-size: .9rem; font-weight: 600; white-space: normal; }
#workflow [role="tablist"] button[aria-selected="true"] { background: #176c60 !important; border-color: #176c60 !important; color: #fff !important; box-shadow: 0 3px 8px #176c601a; }
#workflow [role="tablist"] button:disabled { color: #475761 !important; background: #e8edf0 !important; opacity: 1 !important; cursor: not-allowed; }
#workflow [role="tablist"] button:focus-visible, .gradio-container button:focus-visible { outline: 3px solid #60b8aa !important; outline-offset: 3px; }
.step-page { padding: 26px !important; border: 1px solid #aebbc3 !important; border-radius: 16px !important; background: #fff !important; color: #1f2933 !important; box-shadow: 0 6px 20px #20333e0d; }
.step-heading { padding-bottom: 0 !important; }
.step-heading .html-container, #studio-header .html-container { padding: 0 !important; }
.step-heading .step-count { color: #176c60; font-size: .75rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; }
.step-heading h2 { margin: 6px 0 8px; color: #172e35; font-size: 1.6rem; letter-spacing: -.025em; line-height: 1.3; }
.step-heading p { margin: 0; color: #40515b; line-height: 1.65; max-width: 720px; }
.step-footer { border-top: 1px solid #c1cbd1; padding-top: 18px !important; margin-top: 12px; }
.gradio-container button.primary { background: #176c60 !important; border-color: #176c60 !important; color: #fff !important; border-radius: 9px; }
.gradio-container button.primary:hover { background: #11584e !important; }
.gradio-container button.secondary { border-radius: 9px; }
#recording-prompt { background: #f0f7f5 !important; border: 1px solid #9dbeb4 !important; border-radius: 12px; padding: 24px; }
#recording-prompt p { font-size: 1.35rem; line-height: 1.7; color: #193a34; margin: 0 !important; }
#prompt-queue table th:first-child, #prompt-queue table td:first-child { display: none; }
#prompt-queue table th:nth-child(3), #prompt-queue table td:nth-child(3) { display: none; }
#prompt-queue table th:nth-child(2), #prompt-queue table td:nth-child(2) { width: 72%; }
#workflow-message:empty { display: none; }
#workflow-message p { color: #87501c; background: #fff6e9; border: 1px solid #eddbbd; border-radius: 10px; padding: 12px 16px; margin: 0; }
.gradio-container .prose { line-height: 1.65; }
.gradio-container .prose h3 { margin-top: 12px; color: #233f45; }
.train-summary { border: 1px solid #c4d5d1; border-radius: 10px; background: #f5faf8; padding: 16px 18px; color: #203c37; }
.train-summary-head { display: flex; justify-content: space-between; gap: 12px; align-items: baseline; }
.train-summary-head strong { font-size: 1.05rem; }
.train-summary-head span { color: #536b65; font-size: .85rem; }
.train-summary p { margin: 8px 0 12px; font-size: .92rem; }
.train-progress { height: 10px; border-radius: 999px; background: #d9e7e2; overflow: hidden; }
.train-progress span { display: block; height: 100%; border-radius: inherit; background: #176c60; transition: width .3s ease; }
body.dark .gradio-container .contain { background: #111827 !important; color: #f1f5f9 !important; color-scheme: dark; }
body.dark .gradio-container .contain [data-testid="block-info"], body.dark .gradio-container .contain h3 { color: #f1f5f9 !important; background: transparent !important; }
body.dark .gradio-container .contain p, body.dark .gradio-container .contain .prose { color: #cbd5e1 !important; }
body.dark .gradio-container .contain input, body.dark .gradio-container .contain textarea, body.dark .gradio-container .contain select { color: #f1f5f9 !important; background-color: #111827 !important; border-color: #718096 !important; }
body.dark .gradio-container .contain input::placeholder, body.dark .gradio-container .contain textarea::placeholder { color: #aab7c4 !important; }
body.dark .gradio-container .contain #studio-header h1, body.dark .gradio-container .contain .step-heading h2 { color: #f8fafc !important; }
body.dark .gradio-container .contain #studio-header p, body.dark .gradio-container .contain .step-heading p { color: #cbd5e1 !important; }
body.dark .gradio-container .contain #studio-header .eyebrow, body.dark .gradio-container .contain .step-heading .step-count { color: #70d7c6 !important; }
body.dark .gradio-container .contain #project-context, body.dark .gradio-container .contain .step-page { background: #1f2937 !important; color: #f1f5f9 !important; border-color: #718096 !important; }
body.dark .gradio-container .contain #project-context { border-left-color: #4fb5a4 !important; }
body.dark .gradio-container .contain #project-context p { color: #f1f5f9 !important; }
body.dark .gradio-container .contain #workflow [role="tablist"] button { background: #1f2937 !important; color: #f1f5f9 !important; border-color: #718096 !important; }
body.dark .gradio-container .contain #workflow [role="tablist"] button[aria-selected="true"], body.dark .gradio-container .contain button.primary { background: #176c60 !important; color: #fff !important; border-color: #55bbaa !important; }
body.dark .gradio-container .contain #workflow [role="tablist"] button:disabled { background: #334155 !important; color: #cbd5e1 !important; }
body.dark .gradio-container .contain .step-footer { border-top-color: #526273 !important; }
body.dark .gradio-container .contain #recording-prompt { background: #183b36 !important; border-color: #4b9484 !important; }
body.dark .gradio-container .contain #recording-prompt p { color: #f0fdf9 !important; }
body.dark .gradio-container .contain .train-summary { background: #183b36; border-color: #4b9484; color: #f0fdf9; }
body.dark .gradio-container .contain .train-summary-head span { color: #cbd5e1; }
body.dark .gradio-container .contain #prompt-queue table th, body.dark .gradio-container .contain #prompt-queue table td { background: #1f2937 !important; color: #f1f5f9 !important; border-color: #526273 !important; }
body.dark .gradio-container .contain #prompt-queue table tbody tr:nth-child(even) td { background: #273548 !important; }
@media (max-width: 640px) {
  body .gradio-container main { padding: 16px 8px 28px !important; }
  #workflow [role="tablist"] { grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 6px; }
  #workflow [role="tablist"] button { padding: 12px 6px !important; font-size: .85rem; }
  .step-page { padding: 18px 12px !important; }
  .step-heading h2 { font-size: 1.4rem; }
  #recording-prompt { padding: 18px; }
  #recording-prompt p { font-size: 1.15rem; }
}
"""


def step_heading(number: int, title: str, description: str):
    gr.HTML(
        f'<div class="step-count">Step {number} of 6</div>'
        f"<h2>{title}</h2><p>{description}</p>",
        elem_classes="step-heading",
    )
