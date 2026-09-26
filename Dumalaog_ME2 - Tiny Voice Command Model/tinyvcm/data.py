"""Explicit provenance and disjoint speaker/source splits; no silent read failures."""
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
import numpy as np
import soundfile as sf
from .config import ROOT, SR, SAMPLES, LABELS
from .frontend import Frontend, fit_audio


def audit_dataset(mode='synthetic_baseline', output=None):
    rows = []
    if mode == 'human':
        manifest = ROOT / 'data/human/manifest.csv'
        if not manifest.exists():
            raise ValueError('Record real people first with record_dataset.py. No human manifest exists.')
        with manifest.open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        labels = LABELS + ['_unknown_']
        speakers = sorted({r['speaker'] for r in rows})
        if len(speakers) < 3:
            raise ValueError('Assignment requires recordings from at least three real people.')
        for speaker in speakers:
            selected = [r for r in rows if r['speaker'] == speaker]
            missing = set(labels) - {r['label'] for r in selected}
            if missing or len({r['condition'] for r in selected}) < 2:
                raise ValueError(f'{speaker}: missing classes {sorted(missing)} or fewer than two conditions')
        for r in rows:
            r.update(split='test' if r['speaker'] == speakers[-1] else 'val' if r['speaker'] == speakers[-2] else 'train',
                     provenance='human_recording', group=r['source_id'])
            r['path'] = str(ROOT / r['path'])
    else:
        labels = LABELS.copy()
        # Exclude legacy unsuffixed files of ambiguous provenance. All variants of
        # a voice/rate group stay together, including pitch and time shifts.
        for label in labels:
            for p in sorted((ROOT/'data/dataset'/label).glob('*.wav')):
                if label.startswith('_'):
                    n = int(p.stem.rsplit('_', 1)[1])
                    split = 'test' if n % 10 >= 8 else 'val' if n % 10 >= 6 else 'train'
                    rows.append(dict(path=str(p), label=label, speaker='generated_noise',
                        condition=p.stem.rsplit('_', 1)[0], group=p.stem, split=split, provenance='synthetic_noise'))
                    continue
                match = re.fullmatch(re.escape(label)+r'_(.+)_r(-?\d+)_(\d+)_(clean|shiftL|shiftR|pitchUp|pitchDn)', p.stem)
                if match:
                    voice, rate, index, variant = match.groups()
                    rows.append(dict(path=str(p), label=label, speaker=voice, condition='synthetic',
                        rate=int(rate), group=f'{label}:{voice}:{rate}', provenance='Windows_SAPI_TTS'))
        speakers = sorted({r['speaker'] for r in rows if r['provenance'] == 'Windows_SAPI_TTS'})
        if len(speakers) < 3:
            raise ValueError(f'Expected existing three synthetic voices; found {speakers}')
        for r in rows:
            if r['provenance'] == 'Windows_SAPI_TTS':
                r['split'] = 'test' if r['speaker'] == speakers[-1] else 'val' if r['rate'] == 2 else 'train'
    seen = {}
    for r in rows:
        p = Path(r['path'])
        info = sf.info(p)
        if info.samplerate != SR or info.channels != 1 or info.frames > SAMPLES:
            raise ValueError(f'Invalid format/window: {p.name}: {info}')
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest in seen and seen[digest] != r['split']:
            raise ValueError(f'Identical audio leaks across splits: {p.name}')
        seen[digest] = r['split']
        r['sha256'] = digest
        r['frames'] = info.frames
    groups = {s: {r['group'] for r in rows if r['split'] == s} for s in ['train','val','test']}
    assert not (groups['train'] & groups['val'] or groups['train'] & groups['test'] or groups['val'] & groups['test'])
    for s in groups:
        missing = set(labels) - {r['label'] for r in rows if r['split'] == s}
        if missing:
            raise ValueError(f'{s} has no examples for {missing}')
    report = dict(mode=mode, human_dataset_requirement_met=mode=='human',
                  voices_or_speakers=speakers, test_speaker=speakers[-1],
                  counts=dict(Counter(r['split'] for r in rows)), classes=len(labels),
                  source_groups_disjoint=True, sha256_split_overlap=False,
                  warning='Synthetic voices are not people; results do not establish human recognition.' if mode != 'human' else 'Speaker IDs and consent are supplied by the collector.')
    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        (output/'dataset_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        portable_rows = [{**r, 'path':str(Path(r['path']).relative_to(ROOT)).replace('\\','/')} for r in rows]
        (output/'split_manifest.json').write_text(json.dumps(portable_rows, indent=2), encoding='utf-8')
    return rows, labels, report


def build_features(rows, labels):
    frontend = Frontend()
    splits = {}
    for split in ('train','val','test'):
        selected = [r for r in rows if r['split']==split]
        waves = np.stack([fit_audio(sf.read(r['path'], dtype='float32')[0]) for r in selected])
        features = np.stack([frontend(w) for w in waves])
        targets = np.array([labels.index(r['label']) for r in selected], dtype=np.int64)
        splits[split] = (features, targets, waves)
        print(f'{split}: {len(targets)} examples, features {features.shape}', flush=True)
    return splits
