"""Six workflow pages and their Gradio event wiring.

Callbacks stay in main; presentation and metrics styling stay in ui.
"""

import gradio as gr

from app.performance import epoch_cap_for
from app.ui import APP_CSS, step_heading


def build_app(actions) -> gr.Blocks:
    theme = gr.themes.Soft(primary_hue="teal", secondary_hue="slate", neutral_hue="slate").set(
        block_label_background_fill="transparent", block_label_text_color="#334953",
        body_text_color="#243743", body_text_color_subdued="#52616f",
    )
    projects = actions._projects()
    initial_project = projects[0]["id"] if projects else None
    progress = actions.workflow_progress(initial_project)
    show_metrics = progress.training or bool(progress.runs)
    forward_buttons = []
    back_buttons = []
    with gr.Blocks(title="Piper Voice Studio", theme=theme, css=APP_CSS, analytics_enabled=False, elem_id="voice-app") as demo:
        gr.HTML(
            "<div class='eyebrow'>PIPER · LOCAL VOICE TRAINING</div>"
            "<h1>Your voice. One clear workflow.</h1>"
            "<p>Prepare → record → train → listen. Projects and progress stay on this machine.</p>",
            elem_id="studio-header",
        )
        project_status = gr.Markdown(actions._project_summary(initial_project), elem_id="project-context")
        workflow_status = gr.Markdown(elem_id="workflow-message")
        with gr.Tabs(selected=1, elem_id="workflow") as workflow_tabs:
            with gr.Tab("1 · Project", id=1, elem_classes="step-page") as project_tab:
                step_heading(1, "Choose or create a project", "A project keeps your prompts, recordings, datasets, and training runs together. Choose a saved voice or start a new one.")
                project_select = gr.Dropdown(label="Choose an existing project", choices=actions._project_choices(), value=initial_project, info="Select a project by name. No project ID is needed.")
                with gr.Accordion("Or create a new project", open=not projects) as project_create_panel:
                    project_name = gr.Textbox(label="Project name", placeholder="e.g. My Polish narrator", max_lines=1)
                    with gr.Accordion("Language settings · Polish by default", open=False):
                        with gr.Row():
                            language = gr.Textbox(label="Language code", value="pl_PL", info="pl_PL for Polish, en_US for American English.")
                            espeak = gr.Textbox(label="eSpeak voice", value="pl", info="Use the pronunciation voice for your language, e.g. pl or en-us.")
                    create_button = gr.Button("Create project", variant="primary")
                create_status = gr.Markdown()
                with gr.Row():
                    resume_button = gr.Button("Resume saved progress")
                    import_shortcut = gr.Button("Import a dataset ZIP")
                with gr.Row(elem_classes="step-footer"):
                    continue_project = gr.Button("Continue to Step 2 · Prepare text →", variant="primary")
                forward_buttons.append((continue_project, 1, 2))

            with gr.Tab("2 · Text", id=2, interactive=progress.blocked_reason(2) is None, elem_classes="step-page") as text_tab:
                step_heading(2, "Prepare your recording text", "Paste text or choose a file, then save the prompts you will read.")
                prompt_text = gr.Textbox(label="Paste your source text", lines=6, placeholder="Paste prose, or put one recording prompt on each line.")
                with gr.Accordion("Use a .txt file or a built-in prompt pack instead", open=False):
                    gr.Markdown("An uploaded file takes priority over pasted text. Clear the file to use your pasted text again.")
                    prompt_upload = gr.File(label="Source text file", file_types=[".txt"], type="filepath")
                    prompt_pack = gr.Dropdown(label="Built-in prompt pack", choices=sorted(path.name for path in actions.PROMPT_DIR.glob("*.txt")), value=None)
                    builtin_button = gr.Button("Load prompt pack")
                    source_status = gr.Markdown()
                parse_mode = gr.Radio([("Prose — split into sentences", "prose"), ("One prompt per line", "lines")], value="prose", label="How should the text become prompts?")
                preview_button = gr.Button("Preview & save prompt queue", variant="primary")
                estimate = gr.Markdown()
                queue_status = gr.Markdown()
                gr.Markdown("### Check your prompts\nEdit the **Text** column if needed, then save your changes.")
                queue_table = gr.Dataframe(headers=["Prompt ID", "Text", "Words", "Status"], datatype=["str", "str", "number", "str"], type="array", interactive=True, static_columns=[0, 2, 3], row_count=(0, "dynamic"), col_count=(4, "fixed"), max_height=320, wrap=True, column_widths=[120, 460, 70, 140], label="Prompt queue", elem_id="prompt-queue")
                save_queue_button = gr.Button("Save text edits")
                queue_action_status = gr.Markdown()
                with gr.Row(elem_classes="step-footer"):
                    back_text = gr.Button("← Step 1 · Project")
                    continue_text = gr.Button("Continue to Step 3 · Record →", variant="primary")
                back_buttons.append((back_text, 1))
                forward_buttons.append((continue_text, 2, 3))

            with gr.Tab("3 · Record", id=3, interactive=progress.blocked_reason(3) is None, elem_classes="step-page") as record_tab:
                step_heading(3, "Record your voice", "Read one prompt at a time. Listen, then accept the take or record it again.")
                recording_mode = gr.Radio(["Record new", "Review recordings"], value="Record new", label="What would you like to do?")
                with gr.Column() as record_panel:
                    active_prompt_id = gr.State(value=None)
                    prompt_progress = gr.Markdown("0 / 0")
                    current_text = gr.Markdown("Prepare a prompt queue in Step 2 first.", elem_id="recording-prompt")
                    gr.Markdown("Use a quiet room and keep the same microphone distance. Play your take before accepting it.")
                    recording = gr.Audio(label="Your microphone recording", sources=["microphone"], type="filepath")
                    with gr.Row():
                        accept_button = gr.Button("Accept & next prompt", variant="primary")
                        review_button = gr.Button("Save take for review")
                        reject_button = gr.Button("Reject take", variant="stop")
                    record_result = gr.Markdown()
                    with gr.Accordion("Listen to the last saved take", open=False):
                        sample_player = gr.Audio(label="Last saved recording", interactive=False)
                    prompt_position = gr.State(value=0)
                    with gr.Row():
                        prev_button = gr.Button("← Previous prompt")
                        next_button = gr.Button("Next prompt →")
                with gr.Column(visible=False) as review_panel:
                    gr.Markdown("### Review your recordings\nListen to the sentence, then keep it or record a replacement. Decisions apply when you build a new dataset; saved datasets stay unchanged.")
                    with gr.Row():
                        review_filter = gr.Radio(["Needs review", "All recordings"], value="Needs review", label="Show", min_width=220)
                        review_search = gr.Textbox(label="Find a sentence", placeholder="Search recording text", min_width=220)
                    review_summary = gr.Markdown()
                    sample_select = gr.Dropdown(label="Recording", choices=actions._sample_choices(initial_project), value=None, filterable=True)
                    sample_review_status = gr.Markdown()
                    sample_audio_player = gr.Audio(label="Recording to review", interactive=False)
                    with gr.Row():
                        accept_sample_button = gr.Button("Keep", variant="primary", interactive=False)
                        rerecord_button = gr.Button("Re-record", interactive=False)
                        reject_sample_button = gr.Button("Reject", variant="stop", interactive=False)
                        next_review_button = gr.Button("Next recording")
                    review_action_status = gr.Markdown()
                    with gr.Accordion("Comparison and review details", open=False):
                        review_dataset = gr.Dropdown(label="Compare with dataset", choices=actions._dataset_choices(initial_project), value=None)
                        dataset_audio_player = gr.Audio(label="Dataset copy", interactive=False)
                        review_details = gr.Markdown()
                        flag_sample_button = gr.Button("Flag for review", interactive=False)
                        refresh_review_button = gr.Button("Refresh review list")
                gr.Markdown("Continue when you have accepted recordings. You can return to record more later.")
                with gr.Row(elem_classes="step-footer"):
                    back_record = gr.Button("← Step 2 · Text")
                    continue_record = gr.Button("Continue to Step 4 · Build dataset →", variant="primary")
                back_buttons.append((back_record, 2))
                forward_buttons.append((continue_record, 3, 4))

            with gr.Tab("4 · Dataset", id=4, interactive=progress.blocked_reason(4) is None, elem_classes="step-page") as dataset_tab:
                step_heading(4, "Build your training dataset", "Turn accepted recordings into a reproducible dataset, or import a dataset ZIP from another machine. The dataset is what you will train on in the next step.")
                gr.Markdown("### Build from your recordings\nTargets use actual accepted audio duration. If you have less audio than the target, the dataset uses what is available.")
                duration_target = gr.Dropdown(["15 minutes", "30 minutes", "60 minutes", "All accepted", "Custom duration"], value="30 minutes", label="How much accepted audio should be included?")
                custom_minutes_input = gr.Number(value=45, precision=0, minimum=1, maximum=10000, label="Custom target (minutes)", visible=False)
                create_dataset_button = gr.Button("Create dataset from accepted takes", variant="primary")
                dataset_status = gr.Markdown()
                with gr.Accordion("Or import a dataset ZIP", open=False):
                    gr.Markdown("Use a ZIP exported by this app. Its files and hashes are validated before it becomes available for training.")
                    dataset_import_file = gr.File(label="Dataset ZIP to import", file_types=[".zip"], type="filepath")
                    import_dataset_button = gr.Button("Validate & import dataset")
                gr.Markdown("### Choose the dataset to use")
                dataset_select = gr.Dropdown(label="Saved dataset", value=None, info="Creating or importing a dataset selects it here and in Step 5.")
                dataset_transfer_status = gr.Markdown()
                with gr.Accordion("Download a dataset to train on another machine", open=False):
                    export_dataset_button = gr.Button("Export selected dataset as ZIP")
                    dataset_archive = gr.File(label="Download dataset ZIP", interactive=False)
                with gr.Row(elem_classes="step-footer"):
                    back_dataset = gr.Button("← Step 3 · Record")
                    continue_dataset = gr.Button("Continue to Step 5 · Train →", variant="primary")
                back_buttons.append((back_dataset, 3))
                forward_buttons.append((continue_dataset, 4, 5))

            with gr.Tab("5 · Train", id=5, interactive=progress.blocked_reason(5) is None, elem_classes="step-page") as train_tab:
                step_heading(5, "Train & monitor", "Set up a run, then follow its saved metrics. Training continues when you close this page.")
                training_workspace = gr.Radio(["Setup", "Metrics"], value="Metrics" if show_metrics else "Setup", label="Training workspace", elem_id="training-workspace")
                training_status = gr.Markdown()
                with gr.Column(visible=not show_metrics, elem_classes="studio-panel") as training_setup_panel:
                    train_dataset = gr.Dropdown(label="Training dataset", value=None)
                    training_mode = gr.Radio([("Fine-tune an existing voice (recommended)", "finetune"), ("Train a new voice from scratch", "scratch")], value="finetune", label="Training mode")
                    with gr.Group() as finetune_panel:
                        gr.Markdown("### Starting checkpoint\nFine-tuning needs a compatible Piper **.ckpt** file. For Polish, download the suggested medium voice, or provide your own.")
                        download_checkpoint_button = gr.Button("Download Polish medium checkpoint")
                        base_checkpoint_path = gr.Textbox(label="Checkpoint path", placeholder="Download, upload, or enter a local .ckpt path", info="The selected checkpoint is saved with the training run.")
                        with gr.Accordion("Upload your own checkpoint", open=False):
                            checkpoint_upload = gr.File(label="Piper .ckpt file", file_types=[".ckpt"], type="filepath")
                            checkpoint_upload_button = gr.Button("Save uploaded checkpoint")
                    scratch_warning = gr.Markdown("**Training from scratch:** No base voice checkpoint will be used. Around one hour of speech may be insufficient for a high-quality voice.", visible=False)
                    warmstart_checkbox = gr.Checkbox(label="Optionally warm-start the vocoder from a checkpoint", value=False, visible=False)
                    warmstart_path = gr.Textbox(label="Vocoder warm-start checkpoint path", visible=False)
                    duration_preset = gr.Radio([("Recommended starting cap · 1000 epochs", "1000"), ("Quick experiment · 250 epochs", "250"),
                                                ("Medium experiment · 500 epochs", "500"), ("Custom", "custom")],
                                               value="1000", label="Training duration",
                                               info="Epoch caps are starting points. Listen to saved checkpoints to judge voice quality.")
                    max_epochs = gr.Number(value=1000, precision=0, minimum=1, maximum=100000, label="Maximum epochs")
                    with gr.Accordion("Advanced · device, performance and learning rates", open=False):
                        device = gr.Radio([("Automatic", "auto"), ("CPU", "cpu"), ("NVIDIA GPU / CUDA", "cuda")], value="auto", label="Training device", info="Automatic uses CUDA when available. CPU training can take much longer.")
                        batch_size = gr.Dropdown(["Auto", "4", "8", "16", "32", "64", "Custom"], value="Auto", label="Batch size")
                        custom_batch = gr.Number(value=8, precision=0, minimum=1, label="Custom batch size", visible=False)
                        workers = gr.Dropdown(["Auto", "1", "2", "4", "8", "Custom"], value="Auto", label="DataLoader workers")
                        custom_workers = gr.Number(value=2, precision=0, minimum=1, label="Custom DataLoader workers", visible=False)
                        cpu_threads = gr.Number(value=0, precision=0, minimum=0, label="PyTorch CPU threads (0 = Auto)")
                        gpu_hourly_rate = gr.Textbox(value="", label="GPU cost per hour in USD (optional)", placeholder="e.g. 0.34",
                                                     info="Compute time only; storage and provider fees are excluded.")
                        checkpoint_interval = gr.Number(value=250, precision=0, minimum=1, maximum=100000, label="Save checkpoint every X epochs",
                                                        info="Keeps each milestone. Checkpoints can use hundreds of MB each; choose an interval that fits your storage.")
                        learning_rate = gr.Number(value=0.0002, precision=8, minimum=0.00000001, maximum=1, label="Generator learning rate")
                        learning_rate_d = gr.Number(value=0.0001, precision=8, minimum=0.00000001, maximum=1, label="Discriminator learning rate")
                        training_seed = gr.Number(value=42, precision=0, minimum=0, label="Random seed")
                        gr.Markdown("Learning rates decay once per epoch; the actual rates are logged. The best validation checkpoint is retained separately. Listen to checkpoints to judge voice quality; automatic early stopping is disabled.")
                    gr.Markdown("### Review and start")
                    prepare_button = gr.Button("Review run summary")
                    run_summary = gr.Markdown()
                    train_button = gr.Button("Start training", variant="primary")
                with gr.Column(visible=show_metrics) as training_progress_panel:
                    gr.Markdown("### Run metrics\nProgress, losses and measured performance refresh every 10 seconds. These readings belong to the latest run in this project.")
                    run_progress = gr.HTML("No training runs yet.")
                    run_chart = gr.LinePlot(x="epoch", y="loss", color="series", y_aggregate="mean", title="Loss over epochs", x_title="Epoch", y_title="Loss", height=260, visible=False)
                    with gr.Accordion("Recent trainer logs", open=False):
                        run_status = gr.Code(label="Latest log lines", language="shell", value="No training runs yet.")
                    with gr.Accordion("Hardware diagnostics", open=False):
                        gr.JSON(value=actions.hardware_diagnostics, label="Trainer environment")
                    with gr.Row():
                        refresh_run_button = gr.Button("Refresh training status")
                        cancel_button = gr.Button("Stop training", variant="stop")
                with gr.Row(elem_classes="step-footer"):
                    back_train = gr.Button("← Step 4 · Dataset")
                    continue_train = gr.Button("Continue to Step 6 · Export & listen →", variant="primary")
                back_buttons.append((back_train, 4))
                forward_buttons.append((continue_train, 5, 6))

            with gr.Tab("6 · Voice", id=6, interactive=progress.blocked_reason(6) is None, elem_classes="step-page") as voice_tab:
                step_heading(6, "Listen & compare", "Hear your checkpoint beside the original speaker. Use a reference sentence so both recordings say the same words.")
                with gr.Column(elem_classes="studio-panel"):
                    gr.Markdown("### 1. Choose the voice")
                    with gr.Row():
                        run_select = gr.Dropdown(label="Trained voice run", value=None, choices=[], interactive=False, filterable=False, scale=2)
                        refresh_voice_button = gr.Button("Refresh checkpoints", scale=1)
                    checkpoint_select = gr.Dropdown(label="Saved checkpoint", value=None, interactive=False, filterable=False, info="Choose a milestone or the latest saved checkpoint.")
                    checkpoint_notice = gr.Markdown()
                gr.Markdown("### 2. Choose the words & listen")
                local_test_text = gr.Textbox(label="Shared listening text", value="Dzisiaj sprawdzam własny model głosu.", lines=3, info="Load a reference sentence below, or type your own text to test the checkpoint.")
                with gr.Row(equal_height=True, elem_id="voice-listening"):
                    with gr.Column(min_width=280, elem_classes="studio-panel"):
                        gr.Markdown("### Original speaker\nA real recording from the selected run’s held-out test set.")
                        fixed_test_prompt = gr.Dropdown(label="Reference sentence", choices=[], value=None, interactive=False)
                        load_fixed_prompt_button = gr.Button("Load reference & matching text", interactive=False)
                        reference_status = gr.Markdown("Select a run to check for reference recordings.")
                        reference_audio = gr.Audio(label="Original recording", interactive=False)
                    with gr.Column(min_width=280, elem_classes="studio-panel"):
                        gr.Markdown("### Checkpoint voice\nGenerate speech from the selected training checkpoint.")
                        export_model_button = gr.Button("Generate checkpoint sample", variant="primary", interactive=False)
                        model_status = gr.Markdown()
                        synth_audio = gr.Audio(label="Checkpoint sample", interactive=False)
                with gr.Accordion("Compare two generated voices · models or checkpoints", open=False):
                    gr.Markdown("Choose two saved models or checkpoints, including different runs. Both read the shared listening text above.")
                    with gr.Row():
                        comparison_a = gr.Dropdown(label="Voice A · model or checkpoint", choices=[], value=None, interactive=False)
                        comparison_b = gr.Dropdown(label="Voice B · model or checkpoint", choices=[], value=None, interactive=False)
                    compare_button = gr.Button("Generate A and B", interactive=False)
                    comparison_status = gr.Markdown()
                    with gr.Row(elem_id="voice-ab"):
                        compare_audio_a = gr.Audio(label="Voice A", interactive=False)
                        compare_audio_b = gr.Audio(label="Voice B", interactive=False)
                gr.Markdown("### 3. Keep your voice")
                model_archive = gr.File(label="Download generated voice · ONNX + JSON ZIP", interactive=False)
                with gr.Accordion("Download checkpoint for further training", open=False):
                    gr.Markdown("Save the selected .ckpt file to fine-tune it later or move training to another machine. Keep your dataset ZIP too.")
                    checkpoint_download_button = gr.Button("Prepare checkpoint download", interactive=False)
                    checkpoint_download_status = gr.Markdown()
                    checkpoint_download = gr.File(label="Training checkpoint (.ckpt)", interactive=False)
                with gr.Accordion("Download all training artifacts", open=False):
                    gr.Markdown("All saved checkpoints, CSV metrics, hardware metrics, logs and run configuration. Download after training finishes for a complete archive. Download the dataset ZIP separately in Step 4.")
                    training_zip_button = gr.Button("Prepare training ZIP")
                    training_zip_status = gr.Markdown()
                    training_zip = gr.File(label="Training artifacts ZIP", interactive=False)
                with gr.Accordion("Advanced: filenames and API publishing", open=False):
                    model_select = gr.Dropdown(label="Saved voice for API publishing", choices=[], value=None)
                    model_name = gr.Textbox(label="Voice name prefix", value="piper-voice", info="Each export adds a run/checkpoint identifier and a unique suffix.")
                    gr.Markdown("Publish a generated or saved voice to your configured Piper API.")
                    publish_button = gr.Button("Publish selected voice to Piper API", interactive=False)
                    publish_status = gr.Markdown()
                with gr.Row(elem_classes="step-footer"):
                    back_voice = gr.Button("← Step 5 · Train")
                    record_more = gr.Button("Return to Step 3 · Record more")
                back_buttons.extend([(back_voice, 5), (record_more, 3)])
        poll_signature = gr.State(None)
        step_tabs = [project_tab, text_tab, record_tab, dataset_tab, train_tab, voice_tab]

        create_button.click(actions.create_project, [project_name, language, espeak], [project_select, project_status, create_status, project_create_panel])
        project_state_outputs = [
            project_status, queue_table, dataset_select, sample_select,
            queue_action_status, train_dataset, prompt_position, current_text,
            prompt_progress, active_prompt_id, run_select, model_select, comparison_a, comparison_b,
        ]
        stale_text = [reference_status, comparison_status, checkpoint_notice, estimate, queue_status, source_status, record_result, sample_review_status, dataset_status, dataset_transfer_status, run_summary, training_status, model_status, publish_status, workflow_status, checkpoint_download_status, training_zip_status]
        stale_files = [sample_player, sample_audio_player, dataset_archive, dataset_import_file, model_archive, synth_audio, reference_audio, compare_audio_a, compare_audio_b, checkpoint_download, training_zip]
        project_select.change(actions.select_project_state, project_select, project_state_outputs).then(
            actions.reset_project_view, project_select, [prompt_text, prompt_upload, parse_mode, recording, run_status],
        ).then(
            lambda: ("",) * len(stale_text) + (None,) * len(stale_files), outputs=stale_text + stale_files,
        ).then(actions.refresh_runs, [project_select, run_select, checkpoint_select], [run_status, run_select, run_progress, run_chart, checkpoint_select, export_model_button]).then(actions.step_availability, project_select, step_tabs).then(lambda: gr.update(selected=1), outputs=workflow_tabs)
        scroll_to_step = "() => { document.getElementById('workflow').scrollIntoView({behavior: 'smooth', block: 'start'}); }"
        for button, current, target in forward_buttons:
            button.click(lambda pid, source=current, destination=target: actions.navigate_step(pid, source, destination), project_select, [workflow_tabs, workflow_status]).then(fn=None, js=scroll_to_step)
        for button, target in back_buttons:
            source = 6 if button in (back_voice, record_more) else target + 1
            button.click(lambda pid, current=source, destination=target: actions.navigate_step(pid, current, destination), project_select, [workflow_tabs, workflow_status]).then(fn=None, js=scroll_to_step)
        resume_button.click(lambda pid: actions.navigate_step(pid, 1, actions.workflow_progress(pid).next_step), project_select, [workflow_tabs, workflow_status]).then(fn=None, js=scroll_to_step)
        import_shortcut.click(lambda pid: actions.navigate_step(pid, 1, 4), project_select, [workflow_tabs, workflow_status]).then(fn=None, js=scroll_to_step)
        builtin_button.click(actions.select_prompt_pack, prompt_pack, [prompt_upload, source_status])
        preview_button.click(lambda pid, upload, pasted, mode: actions.preview_source(pid, upload, pasted, mode, 140), [project_select, prompt_upload, prompt_text, parse_mode], [queue_table, estimate, queue_status, project_status]).then(lambda pid: actions.current_prompt(pid, 0), project_select, [prompt_position, current_text, prompt_progress, active_prompt_id]).then(actions.step_availability, project_select, step_tabs)
        save_queue_button.click(actions.save_queue, [project_select, queue_table], [queue_table, project_status, queue_action_status]).then(actions.resume_prompt, project_select, [prompt_position, current_text, prompt_progress, active_prompt_id]).then(actions.step_availability, project_select, step_tabs)
        prev_button.click(lambda pid, idx: actions.current_prompt(pid, max(0, int(idx)-1)), [project_select, prompt_position], [prompt_position, current_text, prompt_progress, active_prompt_id])
        next_button.click(lambda pid, idx: actions.current_prompt(pid, int(idx)+1), [project_select, prompt_position], [prompt_position, current_text, prompt_progress, active_prompt_id])
        for button, decision in ((accept_button, "accept"), (review_button, "review"), (reject_button, "reject")):
            button.click(lambda pid, idx, prompt_id, path, choice=decision: actions.record_sample(pid, int(idx), prompt_id, path, choice), [project_select, prompt_position, active_prompt_id, recording], [record_result, sample_player, project_status, sample_select, prompt_position, queue_table, recording])
        prompt_position.change(lambda pid, idx: actions.current_prompt(pid, int(idx)) if pid else (0, "", "0 / 0", None), [project_select, prompt_position], [prompt_position, current_text, prompt_progress, active_prompt_id])
        review_inputs = [project_select, review_dataset, review_filter, review_search, sample_select]
        review_outputs = [sample_select, review_summary, review_dataset]
        detail_outputs = [sample_audio_player, dataset_audio_player, sample_review_status, review_details,
                          accept_sample_button, rerecord_button, reject_sample_button, flag_sample_button]
        recording_mode.change(lambda mode: (gr.update(visible=mode == "Record new"), gr.update(visible=mode == "Review recordings")),
                              recording_mode, [record_panel, review_panel]).then(actions.recording_review_list, review_inputs, review_outputs).then(
                                  lambda pid: (actions._project_summary(pid), actions._queue_rows(pid)), project_select, [project_status, queue_table]).then(
                                      actions.recording_review_details, [project_select, sample_select, review_dataset], detail_outputs)
        for component in (review_filter, review_dataset):
            component.change(actions.recording_review_list, review_inputs, review_outputs).then(actions.recording_review_details, [project_select, sample_select, review_dataset], detail_outputs)
        review_search.submit(actions.recording_review_list, review_inputs, review_outputs).then(actions.recording_review_details, [project_select, sample_select, review_dataset], detail_outputs)
        refresh_review_button.click(actions.recording_review_list, review_inputs, review_outputs).then(actions.recording_review_details, [project_select, sample_select, review_dataset], detail_outputs)
        next_review_button.click(lambda pid, ds, filt, query, sid: actions.recording_review_list(pid, ds, filt, query, sid, True), review_inputs, review_outputs).then(actions.recording_review_details, [project_select, sample_select, review_dataset], detail_outputs)
        sample_select.change(actions.recording_review_details, [project_select, sample_select, review_dataset], detail_outputs)
        for button, status in ((accept_sample_button, "accepted"), (flag_sample_button, "review"), (reject_sample_button, "rejected")):
            button.click(lambda pid, sid, value=status: actions.review_sample(pid, sid, value), [project_select, sample_select],
                         [sample_select, project_status, review_action_status, queue_table]).then(actions.recording_review_list, review_inputs, review_outputs).then(actions.recording_review_details, [project_select, sample_select, review_dataset], detail_outputs)
        rerecord_button.click(actions.rerecord_sample, [project_select, sample_select],
                              [prompt_position, current_text, prompt_progress, active_prompt_id, recording_mode, recording, record_result]).then(
                                  lambda pid: actions._queue_rows(pid), project_select, queue_table)
        project_select.change(lambda: (None, "", "", "", "Needs review", "", "Record new", None),
                              outputs=[review_dataset, review_summary, review_details, review_action_status, review_filter, review_search, recording_mode, dataset_audio_player])
        duration_target.change(lambda target: gr.update(visible=(target == "Custom duration")), duration_target, custom_minutes_input)
        create_dataset_button.click(actions.make_dataset, [project_select, duration_target, custom_minutes_input], [dataset_select, train_dataset, dataset_status, project_status]).then(actions.step_availability, project_select, step_tabs)
        export_dataset_button.click(actions.export_dataset_ui, [project_select, dataset_select], [dataset_archive, dataset_transfer_status])
        import_dataset_button.click(actions.import_dataset_ui, [project_select, dataset_import_file], [dataset_select, train_dataset, dataset_transfer_status]).then(actions._project_summary, project_select, project_status).then(actions.step_availability, project_select, step_tabs)
        download_checkpoint_button.click(lambda: actions.download_base_checkpoint("pl_PL-darkman-medium"), outputs=[base_checkpoint_path, training_status])
        checkpoint_upload_button.click(actions.save_uploaded_checkpoint, checkpoint_upload, [base_checkpoint_path, training_status])
        training_mode.change(lambda mode: (gr.update(visible=(mode == "finetune")), gr.update(visible=(mode == "scratch")),
                                           gr.update(visible=(mode == "scratch"), value=False), gr.update(visible=False, value=""),
                                           gr.update(choices=[("Quick experiment · 500 epochs", "500"), ("Standard starting cap · 2000 epochs", "2000"), ("Custom", "custom")]
                                                     if mode == "scratch" else [("Recommended starting cap · 1000 epochs", "1000"), ("Quick experiment · 250 epochs", "250"),
                                                                                 ("Medium experiment · 500 epochs", "500"), ("Custom", "custom")],
                                                     value=str(epoch_cap_for(mode))), gr.update(value=epoch_cap_for(mode))),
                             training_mode, [finetune_panel, scratch_warning, warmstart_checkbox, warmstart_path, duration_preset, max_epochs])
        duration_preset.change(lambda preset: gr.update(value=int(preset)) if preset != "custom" else gr.update(), duration_preset, max_epochs)
        batch_size.change(lambda choice: gr.update(visible=choice == "Custom"), batch_size, custom_batch)
        workers.change(lambda choice: gr.update(visible=choice == "Custom"), workers, custom_workers)
        warmstart_checkbox.change(lambda enabled: gr.update(visible=True) if enabled else gr.update(visible=False, value=""), warmstart_checkbox, warmstart_path)
        training_inputs = [project_select, train_dataset, training_mode, base_checkpoint_path, warmstart_path, device,
                           batch_size, training_seed, max_epochs, custom_batch, workers, custom_workers, cpu_threads, gpu_hourly_rate,
                           checkpoint_interval, learning_rate, learning_rate_d]
        prepare_button.click(actions.prepare_training_summary, training_inputs, [run_summary, training_status])
        for setting in (train_dataset, training_mode, base_checkpoint_path, warmstart_path, device, batch_size,
                        custom_batch, workers, custom_workers, cpu_threads, gpu_hourly_rate, training_seed, max_epochs,
                        checkpoint_interval, learning_rate, learning_rate_d):
            setting.change(lambda: "", outputs=run_summary)
        training_workspace.change(lambda view: (gr.update(visible=view == "Setup"), gr.update(visible=view == "Metrics")), training_workspace, [training_setup_panel, training_progress_panel])
        train_button.click(actions.start_training, training_inputs, [training_status, run_status]).then(lambda: "Metrics", outputs=training_workspace).then(actions.refresh_runs, [project_select, run_select, checkpoint_select], [run_status, run_select, run_progress, run_chart, checkpoint_select, export_model_button]).then(actions.step_availability, project_select, step_tabs)
        cancel_button.click(actions.cancel_training, project_select, [training_status, run_status]).then(actions.refresh_runs, [project_select, run_select, checkpoint_select], [run_status, run_select, run_progress, run_chart, checkpoint_select, export_model_button])
        refresh_run_button.click(actions.refresh_runs, [project_select, run_select, checkpoint_select], [run_status, run_select, run_progress, run_chart, checkpoint_select, export_model_button]).then(actions.step_availability, project_select, step_tabs)
        gr.Timer(10).tick(actions.poll_training, [project_select, run_select, checkpoint_select, poll_signature], [run_status, run_select, run_progress, run_chart, checkpoint_select, export_model_button, poll_signature], show_progress="hidden").then(actions.step_availability, project_select, step_tabs, show_progress="hidden").then(actions.checkpoint_help, [project_select, run_select], checkpoint_notice, show_progress="hidden")
        refresh_voice_button.click(actions.refresh_runs, [project_select, run_select, checkpoint_select], [run_status, run_select, run_progress, run_chart, checkpoint_select, export_model_button])
        voice_tab.select(actions.refresh_runs, [project_select, run_select, checkpoint_select], [run_status, run_select, run_progress, run_chart, checkpoint_select, export_model_button]).then(actions.checkpoint_help, [project_select, run_select], checkpoint_notice)
        run_select.input(actions.refresh_checkpoints, [project_select, run_select], [checkpoint_select, export_model_button])
        export_model_button.click(actions.generate_checkpoint_sample, [project_select, run_select, model_name, checkpoint_select, local_test_text], [synth_audio, model_status, model_archive, model_select])
        checkpoint_select.change(lambda checkpoint: gr.update(interactive=bool(checkpoint)), checkpoint_select, checkpoint_download_button)
        checkpoint_download_button.click(actions.download_run_checkpoint, [project_select, run_select, checkpoint_select], [checkpoint_download, checkpoint_download_status])
        local_test_text.input(lambda: (None, "", None, "Custom text: load a reference sentence again to compare matching words."), outputs=[synth_audio, model_status, reference_audio, reference_status])
        for picker in (run_select, checkpoint_select):
            picker.change(lambda: (None, "", None), outputs=[synth_audio, model_status, model_archive])
            picker.input(lambda: (None, ""), outputs=[checkpoint_download, checkpoint_download_status])
        run_select.change(actions.refresh_run_reference, [project_select, run_select], fixed_test_prompt).then(actions.reference_help, [project_select, run_select], reference_status)
        run_select.change(lambda: None, outputs=reference_audio)
        fixed_test_prompt.input(lambda: (None, "Load this reference to update both the recording and shared text."), outputs=[reference_audio, reference_status])
        fixed_test_prompt.change(lambda sample: gr.update(interactive=bool(sample)), fixed_test_prompt, load_fixed_prompt_button)
        training_zip_button.click(actions.download_training_run, [project_select, run_select], [training_zip, training_zip_status])
        run_select.change(lambda: (None, ""), outputs=[training_zip, training_zip_status])
        for trigger, event in ((refresh_voice_button, "click"), (voice_tab, "select")):
            getattr(trigger, event)(actions.refresh_voice_comparison, [project_select, comparison_a, comparison_b], [comparison_a, comparison_b])
        model_select.change(actions.refresh_voice_comparison, [project_select, comparison_a, comparison_b], [comparison_a, comparison_b])
        model_select.change(lambda model: gr.update(interactive=bool(model)), model_select, publish_button)
        for picker in (comparison_a, comparison_b):
            picker.change(lambda a, b: gr.update(interactive=bool(a and b and a != b)), [comparison_a, comparison_b], compare_button)
            picker.change(lambda: (None, None, ""), outputs=[compare_audio_a, compare_audio_b, comparison_status])
        local_test_text.change(lambda: (None, None, ""), outputs=[compare_audio_a, compare_audio_b, comparison_status])
        compare_button.click(actions.compare_voices, [project_select, model_name, comparison_a, comparison_b, local_test_text], [compare_audio_a, compare_audio_b, comparison_status])
        publish_button.click(actions.publish_selected, model_select, publish_status)
        dataset_select.change(lambda dataset: gr.update(value=dataset), dataset_select, train_dataset)
        train_dataset.input(lambda dataset: gr.update(value=dataset), train_dataset, dataset_select)
        load_fixed_prompt_button.click(actions.load_run_evaluation_prompt, [project_select, run_select, fixed_test_prompt], [local_test_text, reference_audio, reference_status]).then(lambda: (None, ""), outputs=[synth_audio, model_status])
        demo.load(actions.load_app_state, project_select, [project_select, *project_state_outputs]).then(actions.reset_project_view, project_select, [prompt_text, prompt_upload, parse_mode, recording, run_status]).then(actions.refresh_runs, [project_select, run_select, checkpoint_select], [run_status, run_select, run_progress, run_chart, checkpoint_select, export_model_button]).then(actions.step_availability, project_select, step_tabs)
    return demo
