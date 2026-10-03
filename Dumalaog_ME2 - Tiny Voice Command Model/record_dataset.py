"""Local human recording workflow. Never labels synthesized audio as a person."""
import csv
import os
import threading
import time
import gradio as gr
from tinyvcm.config import ROOT
from tinyvcm_model.recorder_core import ensure_manifest_columns, save_recording_data
from tinyvcm_model.recording_labels import (
    BACKGROUND_LABEL,
    RECORDER_LABELS,
    RECORDER_PHRASES,
    RECORDER_VARIATIONS,
    RECORDER_VARIANT_IDS,
    SILENCE_LABEL,
    WAKE_LABEL,
    validate_recorder_taxonomy,
)
from tinyvcm.restart import queue_restart


def _ensure_suggested_phrase_column(manifest):
    """Add prompt metadata without changing the contents of existing rows."""
    try:
        return ensure_manifest_columns(manifest)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc


def save_recording(audio,speaker,condition,label,consent,phrase=None):
    try:
        return save_recording_data(ROOT, audio, speaker, condition, label, consent, phrase)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc


def clear_take():
    return None


def main():
    validate_recorder_taxonomy()
    _ensure_suggested_phrase_column(ROOT/'data'/'human'/'manifest.csv')
    with gr.Blocks(title='ME2 - VCM on Raspberry Pi 5 — Recorder') as app:
        gr.Markdown('# ME2 - VCM on Raspberry Pi 5 — Dataset Recorder\nNew command recordings use the 31 ME2 Spoken Command Dataset labels. Each suggested phrase is the most frequent exact transcript for its class, loaded from `docs/ground_truth_phrases/<LABEL>.txt`. The recorder saves that prompt alongside the WAV path in the human manifest; it is prompt metadata, not automatic speech transcription. You may speak naturally while preserving the intent and any fixed slot (time, duration, temperature, brightness, or color). Collect from at least three real people, use anonymous speaker IDs and multiple conditions, and keep a complete speaker aside for testing. Aim for 10 repetitions per phrase per condition. Record wake calls and non-wake examples separately; those labels train/evaluate the wake gate, not the intent classifier. Save the clip before restarting the recorder. Files remain on this PC.')
        gr.Markdown('Restarting briefly closes the recorder. Save any take you want to keep first.')
        restart_status=gr.Markdown('')
        restart=gr.Button('↻ Restart Recorder',variant='secondary')
        with gr.Row():
            speaker=gr.Textbox(label='Anonymous speaker ID',value='person-01')
            condition=gr.Dropdown(['quiet-near','fan-near','quiet-far'],value='quiet-near',label='Condition')
            label=gr.Dropdown(list(RECORDER_LABELS),value=WAKE_LABEL,label='Recording class')
        prompt=gr.Dropdown(choices=list(RECORDER_VARIATIONS[WAKE_LABEL]) if WAKE_LABEL in RECORDER_VARIATIONS else [RECORDER_PHRASES[WAKE_LABEL]], value=RECORDER_PHRASES[WAKE_LABEL], label='Suggested phrase (choose a wording variation)', allow_custom_value=False)
        def phrase_choices(selected_label):
            choices = list(RECORDER_VARIATIONS.get(selected_label, (RECORDER_PHRASES[selected_label],)))
            return gr.update(choices=choices, value=choices[0])
        label.change(phrase_choices,label,prompt)
        consent=gr.Checkbox(label='This real speaker agrees to these local course-project recordings.')
        gr.Markdown('**How to record:** click the microphone button in the audio panel, allow microphone access if asked, speak, then press Stop. Review the captured clip before saving.')
        audio=gr.Audio(sources=['microphone'],type='numpy',label='Record one take, then stop to review')
        with gr.Row():
            save=gr.Button('Save this take',variant='primary')
            record_again=gr.Button('Clear clip / record another',variant='secondary')
        result=gr.Textbox(label='Saved recording status'); playback=gr.Audio(label='Saved 16 kHz WAV')
        save.click(save_recording,[audio,speaker,condition,label,consent,prompt],[result,playback],concurrency_limit=1)
        record_again.click(clear_take,outputs=audio,queue=False)
        gr.Markdown('The CSV logs each saved prompt immediately. Refresh the Excel view after a recording session to include new takes; canonical phrase files remain the ground truth.')
        export_excel = gr.Button('Refresh Excel manifest', variant='secondary')
        excel_file = gr.File(label='Manifest with canonical phrases and performance', interactive=False)
        def refresh_excel_manifest():
            try:
                from scripts.export_manifest_excel import export_manifest
                return str(export_manifest())
            except Exception as exc:
                raise gr.Error(f'Excel export failed; WAVs and CSV are safe: {exc}') from exc
        export_excel.click(refresh_excel_manifest, outputs=excel_file, concurrency_limit=1)
        def restart_recorder():
            try:
                queue_restart('recorder',port=7862)
            except Exception as exc:
                return f'Restart could not be queued: {exc}'

            def close_recorder():
                time.sleep(0.8)  # let Gradio deliver the confirmation to the browser
                app.close(verbose=False)
                time.sleep(0.5)
                os._exit(0)  # detached supervisor starts a fresh shared-.venv instance

            threading.Thread(target=close_recorder,daemon=True).start()
            return 'Restarting recorder… the page should reconnect shortly; refresh if needed.'

        restart.click(restart_recorder,outputs=restart_status,concurrency_limit=1)
    app.launch(server_name='127.0.0.1',server_port=7862,share=False,inbrowser=False)


if __name__=='__main__': main()
