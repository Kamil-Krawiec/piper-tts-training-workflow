import unittest

from app.workflow import WorkflowProgress


class WorkflowProgressTests(unittest.TestCase):
    def test_first_visit_only_allows_project_step(self):
        progress = WorkflowProgress()
        self.assertIsNone(progress.blocked_reason(1))
        for step in range(2, 7):
            self.assertIn("Step 1", progress.blocked_reason(step))

    def test_project_allows_text_and_dataset_import_before_recording(self):
        progress = WorkflowProgress(project=True)
        self.assertIsNone(progress.blocked_reason(2))
        self.assertIsNone(progress.blocked_reason(4))
        self.assertIn("Step 2", progress.blocked_reason(3))
        self.assertIn("Step 4", progress.blocked_reason(5))

    def test_saved_queue_unlocks_recording(self):
        self.assertIsNone(WorkflowProgress(project=True, prompts=2).blocked_reason(3))

    def test_imported_dataset_unlocks_training_without_prompts(self):
        progress = WorkflowProgress(project=True, datasets=1)
        self.assertIsNone(progress.blocked_reason(5))
        self.assertEqual(progress.next_step, 5)

    def test_run_or_exported_model_unlocks_voice_step(self):
        for progress in (WorkflowProgress(project=True, runs=1), WorkflowProgress(project=True, models=1)):
            self.assertIsNone(progress.blocked_reason(6))
            self.assertEqual(progress.next_step, 6)

    def test_resume_returns_to_train_while_a_run_is_active(self):
        progress = WorkflowProgress(project=True, datasets=1, runs=1, training=True)
        self.assertEqual(progress.next_step, 5)

    def test_next_step_uses_actual_saved_progress(self):
        self.assertEqual(WorkflowProgress().next_step, 1)
        self.assertEqual(WorkflowProgress(project=True).next_step, 2)
        self.assertEqual(WorkflowProgress(project=True, prompts=2).next_step, 3)
        self.assertEqual(WorkflowProgress(project=True, prompts=2, accepted=1).next_step, 4)
