import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "task1_runpack.py"
TEMPLATE = ROOT / "configs" / "task1_run.env"


def contract_values():
    result = {}
    for line in TEMPLATE.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            result[key] = value
    return result


def write_contract(path, values):
    path.write_text(
        "".join(f"{key}={value}\n" for key, value in values.items()),
        encoding="utf-8")


class Task1RunpackTest(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(SCRIPT), *map(str, args)],
            cwd=ROOT, text=True, capture_output=True)

    def test_prepare_renders_single_contract_without_gpu_dependencies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "venv/bin").mkdir(parents=True)
            (root / "venv/bin/python").symlink_to(sys.executable)
            values = contract_values()
            values.update({
                "RUN_ID": "fresh-test",
                "PROJECT_DIR": str(ROOT),
                "PERSISTENT_ROOT": str(root),
                "VENV_DIR": str(root / "venv"),
                "ASSETS_DIR": str(root / "assets"),
                "CACHE_DIR": str(root / "cache"),
                "CONTRACTS_DIR": str(root / "contracts"),
                "RUNS_DIR": str(root / "runs"),
                "CHECKPOINT_STORE": str(root / "checkpoints"),
                "DATA_DIR": str(root / "assets/datasets/owt128-327m-v1"),
                "TOKENIZER_DIR": str(root / "assets/tokenizers/gpt2"),
                "EVALUATOR_DIR": str(root / "assets/evaluators/gpt2-large"),
            })
            contract = root / "input.env"
            write_contract(contract, values)
            result = self.run_cli("prepare", contract)
            self.assertEqual(result.returncode, 0, result.stderr)
            rendered = root / "contracts/rendered/fresh-test"
            self.assertTrue((rendered / "resolved.env").is_file())
            self.assertTrue((rendered / "legacy_train.env").is_file())
            self.assertTrue((rendered / "resolved_command.sh").is_file())
            plan = json.loads((rendered / "plan.json").read_text())
            self.assertEqual(plan["checkpoint_milestones"][-1], 50000)
            self.assertEqual(plan["entrypoint"], str(ROOT / "scripts/train_task1_a.sh"))

    def test_eval_path_validation_reads_assets_but_not_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "venv/bin").mkdir(parents=True)
            (root / "venv/bin/python").symlink_to(sys.executable)
            data = root / "assets/datasets/owt128-327m-v1"
            tokenizer = root / "assets/tokenizers/gpt2"
            evaluator = root / "assets/evaluators/gpt2-large"
            checkpoint = root / "checkpoints/example/step_000123.ckpt"
            data.mkdir(parents=True)
            tokenizer.mkdir(parents=True)
            evaluator.mkdir(parents=True)
            checkpoint.parent.mkdir(parents=True)
            (data / "train.bin").write_bytes(b"")
            (data / "validation.bin").write_bytes(b"")
            (data / "manifest.json").write_text(json.dumps({
                "format": "owt128-uint32-v1",
                "sequence_length": 128,
                "splits": {
                    "train": {"file": "train.bin", "sequences": 0},
                    "validation": {"file": "validation.bin", "sequences": 0},
                },
            }), encoding="utf-8")
            (tokenizer / "tokenizer.json").write_text("{}", encoding="utf-8")
            (evaluator / "config.json").write_text("{}", encoding="utf-8")
            (evaluator / "model.safetensors").write_bytes(b"weights")
            checkpoint.write_bytes(b"not loaded by static validation")
            values = contract_values()
            values.update({
                "RUN_ID": "eval-test",
                "RUN_MODE": "eval_only",
                "PROJECT_DIR": str(ROOT),
                "PERSISTENT_ROOT": str(root),
                "VENV_DIR": str(root / "venv"),
                "ASSETS_DIR": str(root / "assets"),
                "CACHE_DIR": str(root / "cache"),
                "CONTRACTS_DIR": str(root / "contracts"),
                "RUNS_DIR": str(root / "runs"),
                "CHECKPOINT_STORE": str(root / "checkpoints"),
                "DATA_DIR": str(data),
                "TOKENIZER_DIR": str(tokenizer),
                "EVALUATOR_DIR": str(evaluator),
                "CHECKPOINT_ROLE": "eval_only",
                "CHECKPOINT_PATH": str(checkpoint),
                "EXPECTED_CHECKPOINT_STEP": "123",
            })
            contract = root / "eval.env"
            write_contract(contract, values)
            result = self.run_cli("validate", contract, "--check-paths")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("VALID: eval-test (eval_only)", result.stdout)

    def test_full_state_render_maps_resume_and_checkpoint_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "venv/bin").mkdir(parents=True)
            (root / "venv/bin/python").symlink_to(sys.executable)
            checkpoint = root / "checkpoints/source/step_015000.ckpt"
            values = contract_values()
            values.update({
                "RUN_ID": "resume-test",
                "RUN_MODE": "full_state",
                "PROJECT_DIR": str(ROOT),
                "PERSISTENT_ROOT": str(root),
                "VENV_DIR": str(root / "venv"),
                "ASSETS_DIR": str(root / "assets"),
                "CACHE_DIR": str(root / "cache"),
                "CONTRACTS_DIR": str(root / "contracts"),
                "RUNS_DIR": str(root / "runs"),
                "CHECKPOINT_STORE": str(root / "checkpoints"),
                "CHECKPOINT_ROLE": "full_state",
                "CHECKPOINT_PATH": str(checkpoint),
                "EXPECTED_CHECKPOINT_STEP": "15000",
                "SOURCE_LEARNING_RATE": "6e-4",
                "TRAIN_MAX_STEPS": "20000",
                "CHECKPOINT_SAVE_EVERY_N_STEPS": "",
                "CHECKPOINT_MILESTONE_STEPS": "17500,20000",
                "RESUME_LR_MODE": "cosine",
                "RESUME_TARGET_LEARNING_RATE": "3e-4",
                "RESUME_LR_TRANSITION_STEPS": "500",
            })
            contract = root / "resume.env"
            rendered = root / "rendered"
            write_contract(contract, values)
            result = self.run_cli("render", contract, "--output-dir", rendered)
            self.assertEqual(result.returncode, 0, result.stderr)
            legacy = (rendered / "legacy_train.env").read_text(encoding="utf-8")
            self.assertIn("SOURCE_GLOBAL_STEP=15000\n", legacy)
            self.assertIn("RESUME_TARGET_LEARNING_RATE=3e-4\n", legacy)
            self.assertIn("RESUME_LR_TRANSITION_SCHEDULE=cosine\n", legacy)
            plan = json.loads((rendered / "plan.json").read_text())
            self.assertEqual(plan["checkpoint_milestones"], [17500, 20000])
            self.assertEqual(plan["entrypoint"],
                             str(ROOT / "scripts/resume_task1_vocab_mse.sh"))

    def test_batch_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            values = contract_values()
            values.update({
                "RUN_ID": "bad-batch",
                "PROJECT_DIR": str(ROOT),
                "TRAIN_GRADIENT_ACCUMULATION": "7",
            })
            contract = root / "bad.env"
            write_contract(contract, values)
            result = self.run_cli("validate", contract)
            self.assertEqual(result.returncode, 2)
            self.assertIn("micro batch * gradient accumulation", result.stderr)


if __name__ == "__main__":
    unittest.main()
