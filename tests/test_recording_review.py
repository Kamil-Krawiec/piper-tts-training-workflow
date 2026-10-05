"""Review imported clips without altering the source dataset."""
import json
import tempfile
import unittest
import wave
from pathlib import Path

from app.projects import ProjectStore
from app.datasets import create_dataset


class RecordingReviewTests(unittest.TestCase):
    def test_import_review_is_idempotent_and_rerecord_supersedes_take(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ProjectStore(Path(tmp))
            pid = store.create_project('Imported')['id']
            root = store.project_dir(pid) / 'datasets' / 'imported-test'
            (root / 'audio').mkdir(parents=True)
            audio = root / 'audio' / '000001.wav'
            with wave.open(str(audio), 'wb') as wav:
                wav.setparams((1, 2, 22050, 0, 'NONE', 'not compressed'))
                wav.writeframes(b'\x10\x00' * 44100)
            (root / 'sample-index.json').write_text(json.dumps([
                {'sample_id': 'source-id', 'filename': audio.name, 'text': 'Read this.', 'split': 'validation'}]))
            self.assertEqual(store.register_dataset_recordings(pid, root), 1)
            self.assertEqual(store.register_dataset_recordings(pid, root), 0)
            row = store.list_samples(pid)[0]
            self.assertEqual(row['quality']['source_split'], 'validation')
            store.set_sample_status(pid, row['id'], 'review')
            replacement = store.add_sample(pid, row['prompt_id'], 'replacement.wav', 2, audio_file=str(audio))
            rows = {r['id']: r for r in store.list_samples(pid)}
            self.assertEqual(rows[row['id']]['status'], 'superseded')
            self.assertEqual(rows[replacement]['status'], 'accepted')
            self.assertEqual(rows[replacement]['quality']['source_split'], 'validation')
            self.assertTrue(audio.exists())

    def test_rebuilt_dataset_preserves_supplied_membership(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / 'take.wav'
            with wave.open(str(audio), 'wb') as wav:
                wav.setparams((1, 2, 22050, 0, 'NONE', 'not compressed'))
                wav.writeframes(b'\x10\x00' * 44100)
            samples = [{'id': str(i), 'text': 'Sentence.', 'audio_file': str(audio), 'duration_seconds': 2} for i in range(6)]
            fixed = {'0': 'test', '1': 'validation', '2': 'train', '3': 'train', '4': 'train', '5': 'train'}
            create_dataset(samples, root / 'rebuilt', None, split_by_id=fixed)
            self.assertEqual(json.loads((root / 'rebuilt/splits/test.json').read_text()), ['0'])
            self.assertEqual(json.loads((root / 'rebuilt/splits/validation.json').read_text()), ['1'])

    def test_large_cut_is_a_review_candidate_until_kept(self):
        from app.recording_review import review_rows
        with tempfile.TemporaryDirectory() as tmp:
            store = ProjectStore(Path(tmp))
            pid = store.create_project('Review')['id']
            prompt = store.import_prompts(pid, 'text.txt', 'A sentence.', 'lines')[0]
            source = store.project_dir(pid) / 'recordings/normalized/source.wav'
            with wave.open(str(source), 'wb') as wav:
                wav.setparams((1, 2, 22050, 0, 'NONE', 'not compressed'))
                wav.writeframes(b'\x10\x00' * 88200)
            sid = store.add_sample(pid, prompt, 'recordings/normalized/source.wav', 4, audio_file=str(source))
            ds = store.project_dir(pid) / 'datasets/review-test'
            (ds / 'audio').mkdir(parents=True)
            copy = ds / 'audio/take.wav'
            with wave.open(str(copy), 'wb') as wav:
                wav.setparams((1, 2, 22050, 0, 'NONE', 'not compressed'))
                wav.writeframes(b'\x10\x00' * 44100)
            (ds / 'sample-index.json').write_text(json.dumps([{'sample_id': sid, 'filename': 'take.wav', 'text': 'A sentence.', 'split': 'train'}]))
            row = review_rows(store, pid, ds)[0]
            self.assertTrue(row['needs_review'])
            self.assertEqual(row['removed_seconds'], 2)
            store.set_sample_status(pid, sid, 'accepted')
            self.assertFalse(review_rows(store, pid, ds)[0]['needs_review'])
            store.set_sample_status(pid, sid, 'review')
            self.assertTrue(review_rows(store, pid, ds)[0]['needs_review'])
            self.assertEqual(source.stat().st_size, 176444)
