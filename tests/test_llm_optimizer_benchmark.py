import json
import unittest
from unittest import mock

import torch

from main.run_llm_optimizer_benchmark import CharTransformer, build_optimizer, get_batch, load_corpus, OPTIMIZER_NAMES
from src.llm_judge import MissingAPIKeyError, judge_samples


class CharTransformerTests(unittest.TestCase):
    def test_forward_produces_logits_over_vocab(self) -> None:
        model = CharTransformer(vocab_size=17, block_size=8, n_embd=16, n_head=2, n_layer=1)
        idx = torch.randint(0, 17, (3, 8))
        logits = model(idx)
        self.assertEqual(logits.shape, (3, 8, 17))

    def test_generate_extends_sequence(self) -> None:
        model = CharTransformer(vocab_size=17, block_size=8, n_embd=16, n_head=2, n_layer=1)
        idx = torch.randint(0, 17, (1, 4))
        out = model.generate(idx, max_new_tokens=5)
        self.assertEqual(out.shape, (1, 9))


class DataHelperTests(unittest.TestCase):
    def test_load_corpus_roundtrips_through_vocab(self) -> None:
        import tempfile
        from pathlib import Path

        tmp_text = "abcabcabc\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "corpus.txt"
            path.write_text(tmp_text, encoding="utf-8")
            train_data, val_data, itos, stoi = load_corpus(path)
            self.assertEqual(len(train_data) + len(val_data), len(tmp_text))
            self.assertEqual(set(stoi.keys()), set(itos.values()))

    def test_get_batch_shapes(self) -> None:
        data = torch.arange(100)
        rng = torch.Generator().manual_seed(0)
        x, y = get_batch(data, block_size=8, batch_size=4, rng=rng)
        self.assertEqual(x.shape, (4, 8))
        self.assertEqual(y.shape, (4, 8))
        # y is x shifted by one position
        self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))


class BuildOptimizerTests(unittest.TestCase):
    def test_all_declared_optimizer_names_are_buildable(self) -> None:
        model = torch.nn.Linear(4, 4)
        for name in OPTIMIZER_NAMES:
            optimizer = build_optimizer(name, model)
            self.assertIsNotNone(optimizer)

    def test_unknown_optimizer_raises(self) -> None:
        model = torch.nn.Linear(4, 4)
        with self.assertRaises(ValueError):
            build_optimizer("not_a_real_optimizer", model)


class LLMJudgeTests(unittest.TestCase):
    def test_missing_api_key_raises(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(MissingAPIKeyError):
                judge_samples({"opt_a": "hello", "opt_b": "world"}, corpus_excerpt="a corpus")

    def test_judge_samples_unblinds_results_by_original_name(self) -> None:
        fake_scores = {
            "sample_a": {"coherence_score": 7, "style_fidelity_score": 5, "rationale": "ok"},
            "sample_b": {"coherence_score": 3, "style_fidelity_score": 2, "rationale": "weak"},
        }

        fake_message = mock.Mock()
        fake_message.content = json.dumps(fake_scores)
        fake_choice = mock.Mock()
        fake_choice.message = fake_message
        fake_response = mock.Mock()
        fake_response.choices = [fake_choice]

        fake_client = mock.Mock()
        fake_client.chat_completion.return_value = fake_response

        with mock.patch("src.llm_judge.InferenceClient", return_value=fake_client):
            result = judge_samples({"opt_a": "hello", "opt_b": "world"}, corpus_excerpt="a corpus", token="fake-token")

        self.assertEqual(set(result.keys()), {"opt_a", "opt_b"})
        for scores in result.values():
            self.assertIn("coherence_score", scores)
            self.assertIn("style_fidelity_score", scores)


if __name__ == "__main__":
    unittest.main()
