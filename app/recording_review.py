"""Recording review facts, independent of the UI and immutable datasets."""
import json
import wave
from pathlib import Path


def duration(path):
    with wave.open(str(path), 'rb') as audio:
        return audio.getnframes() / audio.getframerate()


def review_rows(store, project_id, dataset=None):
    if not store.has_project(project_id):
        return []
    index = {}
    if dataset and (dataset / 'sample-index.json').is_file():
        index = {row['sample_id']: row for row in json.loads((dataset / 'sample-index.json').read_text(encoding='utf-8'))}
    rows = []
    for sample in store.list_samples(project_id):
        if sample['status'] == 'superseded':
            continue
        quality = sample['quality']
        source = Path(sample.get('audio_file') or '')
        entry = index.get(quality.get('source_sample_id', sample['id']))
        copy = (dataset / 'audio' / entry['filename']).resolve() if entry else None
        if copy and not copy.is_relative_to((dataset / 'audio').resolve()):
            raise ValueError('Invalid dataset recording path')
        removed = None
        reasons = list(quality.get('warnings', []))
        if not source.is_file():
            reasons.append('Recording file is missing')
        elif copy and copy.is_file() and not quality.get('imported'):
            original_seconds = duration(source)
            removed = max(0, original_seconds - duration(copy))
            if removed > 1 or (original_seconds and removed / original_seconds > 0.25):
                reasons.append(f'Check speech edges: {removed:.2f}s removed')
        if sample['status'] == 'review':
            reasons.insert(0, 'Marked for review')
        needs_review = sample['status'] == 'review' or (bool(reasons) and not quality.get('reviewed') and sample['status'] != 'rejected')
        rows.append({**sample, 'original': str(source) if source.is_file() else None,
                     'dataset_audio': str(copy) if copy and copy.is_file() else None,
                     'removed_seconds': removed, 'reasons': reasons, 'needs_review': needs_review})
    return rows
