# Recorder command ground-truth phrases

Each `<INTENT_LABEL>.txt` file contains the most frequent exact transcript for that model intent in the ME2 Spoken Command Dataset. The recorder loads these files directly, so this directory is the canonical source of the 31 command suggestions. The dataset audit checks the file names, recorder values, and exact most-frequent transcript for every class.

`LIGHT_OFF.txt` says `Kill the lights` (200 training clips), the most frequent `LIGHT_OFF` transcript. Older personal takes are stored under the legacy `lights_off` label and do not contain transcript text, so their exact spoken wording cannot be verified from the manifest. Their class label still maps to `LIGHT_OFF`; the prompt change does not relabel those recordings.

For new human takes, the manifest's `suggested_phrase` column records the prompt that the recorder displayed. This is prompt metadata, not an automatic transcript of the speaker's actual utterance. Rows recorded before this field existed keep it blank because their displayed prompt was not recorded.

Wake, unrelated speech, background noise, and silence are recorder-only special labels; their guidance remains in `tinyvcm_model/recording_labels.py`.
