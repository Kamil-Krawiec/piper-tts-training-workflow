import tempfile
import unittest
from pathlib import Path

from app.projects import ProjectStore
from app.text import estimate_text, parse_prompts, prompt_recommendation


class PromptParsingTests(unittest.TestCase):
    def test_prose_splitting_preserves_polish_sentences_and_abbreviations(self):
        text = (
            "Dr. Nowak mówił o wersji 1.2.3. Potem dodał: to działa!\n"
            "Czy możemy zaczynać?"
        )
        self.assertEqual(
            parse_prompts(text, "prose"),
            [
                "Dr. Nowak mówił o wersji 1.2.3.",
                "Potem dodał: to działa!",
                "Czy możemy zaczynać?",
            ],
        )

    def test_polish_multi_part_abbreviation_does_not_create_a_false_sentence(self):
        self.assertEqual(
            parse_prompts("Dr. Nowak przygotował m.in. tekst do nagrania. Potem go przeczytał.", "prose"),
            ["Dr. Nowak przygotował m.in. tekst do nagrania.", "Potem go przeczytał."],
        )

    def test_line_mode_keeps_each_nonempty_line_as_one_prompt(self):
        self.assertEqual(
            parse_prompts("Pierwsze zdanie.\n\nDrugie zdanie.\n Trzecie zdanie!", "lines"),
            ["Pierwsze zdanie.", "Drugie zdanie.", "Trzecie zdanie!"],
        )

    def test_estimate_reports_words_characters_and_minutes(self):
        result = estimate_text(["Ala ma kota.", "To jest drugi prompt."], 60)
        self.assertEqual(result["words"], 7)
        self.assertEqual(result["characters"], 33)
        self.assertEqual(result["estimated_seconds"], 7)

    def test_prompt_length_and_duplicate_warnings_are_soft_recommendations(self):
        self.assertIn("Too short", prompt_recommendation("Tak."))
        self.assertEqual(prompt_recommendation("Dzisiaj sprawdzam własny głos."), "Ready")
        self.assertEqual(prompt_recommendation("Dzisiaj sprawdzam własny głos.", {"dzisiaj sprawdzam własny głos."}), "Duplicate")


class ProjectStoreTests(unittest.TestCase):
    def test_project_lookup_treats_arbitrary_ids_as_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            store = ProjectStore(Path(temp))
            project = store.create_project("Studio")

            self.assertTrue(store.has_project(project["id"]))
            self.assertFalse(store.has_project("not-a-project-id"))
            self.assertFalse(store.has_project("00000000-0000-0000-0000-000000000000"))

    def test_source_queue_and_sample_survive_reopen_and_prompt_edit(self):
        with tempfile.TemporaryDirectory() as temp:
            store = ProjectStore(Path(temp))
            project = store.create_project("Kamil PL")
            ids = store.import_prompts(
                project["id"], "własne-zdania.txt", "Pierwszy tekst.\nDrugi tekst.", "lines"
            )
            store.set_active_prompt(project["id"], ids[1])
            store.add_sample(project["id"], ids[0], "recordings/raw/first.webm", 3.2)
            store.update_prompt(project["id"], ids[0], "Zmieniony tekst.")

            reopened = ProjectStore(Path(temp))
            self.assertEqual(reopened.get_project(project["id"])["prompts"][0]["text"], "Zmieniony tekst.")
            self.assertEqual(reopened.list_samples(project["id"])[0]["prompt_id"], ids[0])
            self.assertEqual(reopened.list_samples(project["id"])[0]["text"], "Pierwszy tekst.")
            self.assertEqual(reopened.get_active_prompt(project["id"]), ids[1])
            self.assertTrue((Path(temp) / "projects" / project["id"] / "source" / "original.txt").exists())


if __name__ == "__main__":
    unittest.main()
