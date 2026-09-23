import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def load_summary_module():
    path = SOURCE_ROOT / 'scripts' / 'summarize_task1_metrics.py'
    spec = importlib.util.spec_from_file_location('task1_summary', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Task1VocabStaticContractTests(unittest.TestCase):
    def test_candidate_entrypoint_freezes_6k_batch_and_warmups(self):
        legacy = (SOURCE_ROOT / 'scripts' /
                  'train_task1_vocab_mse_100k.sh').read_text()
        self.assertIn('obsolete 100k entry is disabled', legacy)
        script = (SOURCE_ROOT / 'scripts' /
                  'train_task1_vocab_mse_round.sh').read_text()
        self.assertIn('export LOSS_VARIANT=task1_a', script)
        self.assertIn('export LOSS_VARIANT=task1_c', script)
        self.assertIn('TASK1_MAX_STEPS:-6000', script)
        self.assertIn('export GLOBAL_BATCH_SIZE=256', script)
        self.assertIn('export MICRO_BATCH_SIZE="$TASK1_MICRO_BATCH_SIZE"', script)
        self.assertIn('256 / TASK1_MICRO_BATCH_SIZE', script)
        self.assertIn('export DATA_CONFIG=openwebtext_327m_packed', script)
        self.assertIn('export WARMUP_STEPS=2500', script)
        self.assertIn('export TOKEN_BIAS_WARMUP_STEPS=5000', script)
        self.assertIn("milestone_steps='[6000]'", script)
        self.assertIn('export MILESTONE_SAVE_WEIGHTS_ONLY=false', script)
        self.assertIn('rolling_last_interval=1000', script)
        self.assertIn(
            'export MILESTONE_LAST_EVERY_N_STEPS="$rolling_last_interval"',
            script)
        self.assertIn('LEARNING_RATE', script)
        for forbidden in ('MAX_STEPS=100000', 'GLOBAL_BATCH_SIZE=64',
                          'export WARMUP_STEPS=50\n'):
            self.assertNotIn(forbidden, script)

    def test_h800_runner_has_fixed_a_only_sequential_mapping(self):
        script = (SOURCE_ROOT / 'scripts' /
                  'run_task1_ac_h800_4candidate_6k.sh').read_text()
        for required in (
                'TASK1_MODE must be gate or run', 'TASK1_SWEEP_ID',
                'candidate_ids=(a_lr3e-4 a_lr6e-4)',
                'candidate_variants=(A A)',
                'candidate_lrs=(3e-4 6e-4)',
                'candidate_devices=(0 0)',
                'candidate_paths=(A/lr_3e-4 A/lr_6e-4)',
                'for index in 0 1; do',
                'TASK1_MICRO_BATCH_SIZE:?Set the one micro batch',
                'accumulation="$((256 / TASK1_MICRO_BATCH_SIZE))"',
                "'optimizer_steps_per_candidate': 6000",
                "'global_batch_per_candidate': 256",
                "'lr_warmup_optimizer_steps': 2500",
                "'gaussian_bias_warmup_optimizer_steps': 5000",
                "'candidate_count': 2",
                "'processes_per_physical_h800': 1",
                "'exact_command_count': 2",
                'this run is never overwritten',
                'TASK1_MAX_STEPS=6000',
                'train_task1_vocab_mse_round.sh',
                'eval_task1_vocab_mse.sh',
                'LOSS_VARIANT="task1_${candidate_id%%_*}"',
                "'candidate_chaining': False",
                "'storage': 'remote persistent disk only'",
                "'checkpoint_transfer_policy': 'remote_only'"):
            self.assertIn(required, script)
        self.assertIn('launch_training_group smoke 1 true', script)
        self.assertIn('launch_training_group candidates 6000 false', script)
        self.assertIn('--a-only', script)
        self.assertIn(
            'for checkpoint_name in step_006000.ckpt last.ckpt', script)
        for forbidden in (
                'c_lr', 'candidate_devices=(0 0 1 1)', 'wait_group',
                'child_pids', ' 2>&1 &', 'GLOBAL_BATCH_SIZE=64',
                "optimizer_steps_per_candidate': 2000"):
            self.assertNotIn(forbidden, script)

        formal_eval = (SOURCE_ROOT / 'scripts' /
                       'eval_task1_vocab_mse.sh').read_text()
        for required in (
                'NUM_SAMPLES=1024', 'SAMPLING_STEPS=128',
                'DISABLE_EMA=false', 'GEN_PPL_COMPARABLE=true'):
            self.assertIn(required, formal_eval)

    def test_candidate_last_checkpoint_is_full_atomic_and_isolated(self):
        launcher = (SOURCE_ROOT / 'scripts' /
                    'train_owt_128_langflow_hybrid.sh').read_text()
        for required in (
                'MILESTONE_LAST_EVERY_N_STEPS:-0',
                'milestone_checkpoints.last_every_n_steps=',
                'milestone_checkpoints.last_filename=last.ckpt',
                'milestone_checkpoints.directory=$run_dir/checkpoints',
                'optim.beta1="$optim_beta1"',
                'optim.beta2="$optim_beta2"',
                'training.ema="$ema_decay"',
                '+experiment.train_config_path='):
            self.assertIn(required, launcher)

        callback = (SOURCE_ROOT / 'experiment_callbacks.py').read_text()
        for required in (
                "last_every_n_steps=0, last_filename='last.ckpt'",
                "temporary_path = f'{path}.tmp'",
                'trainer.save_checkpoint(temporary_path, weights_only=False)',
                'os.replace(temporary_path, path)',
                "trainer.strategy.barrier('rolling_last_checkpoint')"):
            self.assertIn(required, callback)

    def test_resume_keeps_state_and_supports_explicit_constant_lr_graft(self):
        script = (SOURCE_ROOT / 'scripts' /
                  'resume_task1_vocab_mse.sh').read_text()
        for required in (
                'RESUME_CHECKPOINT_PATH', 'SOURCE_LEARNING_RATE',
                'SOURCE_GLOBAL_STEP',
                'RESUME_CONSTANT_LEARNING_RATE', 'TARGET_MAX_STEPS',
                'task1_load_train_config "$config_path"',
                'task1_resolve_batch "${HARDWARE_PROFILE:-h800}"',
                'export WARMUP_STEPS="${WARMUP_STEPS:-2500}"',
                'export TOKEN_BIAS_WARMUP_STEPS="${TOKEN_BIAS_WARMUP_STEPS:-5000}"',
                'export RESUME_FROM_CKPT=true',
                'export RESUME_CKPT_PATH="$RESUME_CHECKPOINT_PATH"'):
            self.assertIn(required, script)

        trainer = (SOURCE_ROOT / 'trainer_base.py').read_text()
        for required in (
                '_apply_task1_resume_constant_lr',
                "scheduler.lr_lambdas = [lambda _: 1.0",
                "group['initial_lr'] = target_lr",
                "'optimizer_moments_preserved': True",
                "'lr_warmup_restarted': False",
                "'gaussian_bias_warmup_restarted': False"):
            self.assertIn(required, trainer)

    def test_checkpoint_verifier_names_full_restore_state(self):
        verifier = (SOURCE_ROOT / 'scripts' /
                    'verify_task1_checkpoint.py').read_text()
        for required in (
                "('state_dict', 'loops')",
                "('optimizer_states', 'lr_schedulers')",
                "checkpoint.get('ema')", "checkpoint.get('global_step'",
                "checkpoint.get('sampler')",
                "checkpoint.get('task1_optimizer_contract')",
                "checkpoint.get('task1_lr_graft_history')",
                'resume_task1_vocab_mse.sh'):
            self.assertIn(required, verifier)

    def test_formal_evaluation_freezes_1024_samples_nfe_and_ema(self):
        script = (SOURCE_ROOT / 'scripts' /
                  'eval_task1_vocab_mse.sh').read_text()
        self.assertIn('NUM_SAMPLES=1024', script)
        self.assertIn('SAMPLING_STEPS=128', script)
        self.assertIn('DISABLE_EMA=false', script)
        self.assertIn('GEN_PPL_COMPARABLE=true', script)

    def test_tau_frontload_diagnostic_freezes_grid_and_noncomparability(self):
        evaluator = (SOURCE_ROOT / 'scripts' /
                     'eval_task1_tau_frontload128.sh').read_text()
        for required in (
                'NUM_SAMPLES=1024', 'SAMPLING_STEPS=128',
                'EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"',
                'DISABLE_EMA=false', 'GEN_PPL_COMPARABLE=false',
                'TASK1_TIME_GRID=tau_box_physical_equal_0p6_jump',
                "TASK1_TAU_BOX_QUERY_COUNTS='[32,23,23,24,12,13,1]'",
                'TASK1_INITIAL_NOISE_SEED=42',
                'IMPLEMENTATION_COMMIT',
                'verify_task1_tau_frontload_eval.py'):
            self.assertIn(required, evaluator)

        runner = (SOURCE_ROOT / 'scripts' /
                  'run_task1_tau_frontload128_7ckpt.sh').read_text()
        for label in (
                'a_6e4_15k', 'h0_20k', 'h0_25k', 'h0_30k',
                'v1_18k', 'v1_24k', 'v1_30k'):
            self.assertIn(label, runner)
        self.assertIn('pids+=("$!")', runner)
        self.assertIn('if ! wait "$pid"', runner)
        self.assertIn('At least one checkpoint diagnostic failed', runner)
        self.assertIn("if [[ \"${AUTO_SHUTDOWN:-false}\" == true ]]", runner)
        self.assertIn('/usr/bin/shutdown', runner)

        verifier = (SOURCE_ROOT / 'scripts' /
                    'verify_task1_tau_frontload_eval.py').read_text()
        for required in (
                'EXPECTED_COUNTS = [32, 23, 23, 24, 12, 13, 1]',
                "samples['comparable'] is False",
                "samples['weights'] == 'ema'",
                "samples['generation_seed'] == 42",
                "Path(samples['evaluator_model_name_or_path']).name == 'gpt2-large'",
                "grid['final_query_tau'] - 0.6",
                "grid['final_update_target_physical_time'] == 1.0"):
            self.assertIn(required, verifier)

    def test_first_interval32_balanced_tail_contract(self):
        evaluator = (SOURCE_ROOT / 'scripts' /
                     'eval_task1_first_interval32_tail_balanced.sh').read_text()
        for required in (
                'NUM_SAMPLES=1024', 'SAMPLING_STEPS=128',
                'DISABLE_EMA=false', 'GEN_PPL_COMPARABLE=false',
                'TASK1_TIME_GRID=first_interval32_tail_bins_balanced',
                "TASK1_TAIL_BIN_REMOVAL_COUNTS='[8,8,8,7]'",
                "TASK1_DIAGNOSTIC_STEPS='[1,2,4,8,16,32]'",
                'TASK1_INITIAL_NOISE_SEED=42'):
            self.assertIn(required, evaluator)
        runner = (SOURCE_ROOT / 'scripts' /
                  'run_task1_first_interval32_7ckpt.sh').read_text()
        self.assertIn('pids+=("$!")', runner)
        self.assertIn('if ! wait "$pid"', runner)
        self.assertIn('/usr/bin/shutdown', runner)
        verifier = (SOURCE_ROOT / 'scripts' /
                    'verify_task1_first_interval32_eval.py').read_text()
        for required in (
                "grid['tail_removal_counts'] == [8, 8, 8, 7]",
                "grid['preserved_last_uniform_query_index'] == 127",
                "grid['last_prediction_tau'] - 127 / 128"):
            self.assertIn(required, verifier)

    def test_a15_v1_30k_nfe_refinement_contract(self):
        evaluator = (SOURCE_ROOT / 'scripts' /
                     'eval_task1_nfe_refinement.sh').read_text()
        for required in (
                'NUM_SAMPLES=1024', 'REFINEMENT_NFE',
                'SAMPLING_STEPS="$REFINEMENT_NFE"',
                'SAMPLING_SOLVER=euler', 'SAMPLING_TEMPERATURE=1.0',
                'EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"',
                'DISABLE_EMA=false', 'TASK1_INITIAL_NOISE_SEED=42',
                'first_interval64_tail_keep_4_3_2_1',
                'first_interval32_tail_bins_balanced',
                'TASK1_TIME_GRID="$time_grid"',
                'GEN_PPL_PROTOCOL_ID="$protocol_id"',
                'launcher_log="${RUN_DIR}.launcher.log"',
                'verify_task1_nfe_refinement_eval.py'):
            self.assertIn(required, evaluator)
        self.assertNotIn('mv "$launcher_log"', evaluator)

        runner = (SOURCE_ROOT / 'scripts' /
                  'run_task1_a15_v1_30k_nfe_refinement.sh').read_text()
        for required in (
                'A15_CHECKPOINT', 'V1_30_CHECKPOINT',
                'labels=(v1_30k)',
                'grids=(first_interval64 uniform_t)',
                'labels=(a_6e4_15k v1_30k)',
                'grids=(g32 uniform_t)', 'run_nfe_group 128',
                'run_nfe_group 256', 'run_nfe_group 512',
                "'evaluation_count': 10", "'max_parallel_workers': 4",
                "'verified_completion_count': 10",
                "'output_streams_closed': True",
                'TASK1_SYNC_COMMAND',
                'if [[ "$auto_shutdown" == true ]]',
                '"$shutdown_command"'):
            self.assertIn(required, runner)
        self.assertIn('/usr/bin/shutdown', runner)

        verifier = (SOURCE_ROOT / 'scripts' /
                    'verify_task1_nfe_refinement_eval.py').read_text()
        for required in (
                "choices=(128, 256, 512)",
                "EXPECTED_STEPS = {'a_6e4_15k': 15000, 'v1_30k': 30000}",
                "samples['comparable'] is False",
                "parser.add_argument('--checkpoint-path'",
                "parser.add_argument('--launcher-log'",
                "args.grid == 'first_interval64'",
                "grid['first_interval_substeps'] == 32 * factor",
                "grid['tail_removal_counts'] == [8, 8, 8, 7]",
                "'physical_time_uniform_official_inverse_lut'"):
            self.assertIn(required, verifier)

    def test_nfe_refinement_verifier_accepts_frozen_uniform_t_contract(self):
        verifier = SOURCE_ROOT / 'scripts' / \
            'verify_task1_nfe_refinement_eval.py'
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / 'uniform_t_nfe128' / 'v1_30k'
            (run_dir / '.hydra').mkdir(parents=True)
            (run_dir / '.hydra' / 'config.yaml').write_text(
                'sampling:\n  steps: 128\n', encoding='utf-8')
            checkpoint = root / 'v1_30k.ckpt'
            checkpoint.touch()
            launcher_log = Path(f'{run_dir}.launcher.log')
            launcher_log.write_text('closed\n', encoding='utf-8')
            scores = [{'nll': 0.0, 'scored_tokens': 1}
                      for _ in range(1024)]
            sample_quality = {
                'mean_sample_unigram_entropy_nats': 1.0,
                'distinct_1': 1.0,
                'distinct_2': 1.0,
                'repeated_4gram_fraction': 0.0,
                'max_single_token_fraction': 0.1,
                'special_token_frequency': 0.0,
                'sample_to_sample_duplication': 0.0,
            }
            (run_dir / 'samples.json').write_text(json.dumps({
                'protocol_id': (
                    'owt128-gpt2large-genppl-diagnostic-'
                    'physical-time-uniform-128-v1'),
                'evaluator_model_name_or_path': '/models/gpt2-large',
                'comparable': False,
                'weights': 'ema',
                'generation_seed': 42,
                'nfe': 128,
                'sequence_length': 128,
                'solver': 'euler',
                'temperature': 1.0,
                'num_samples': 1024,
                'checkpoint_global_step': 30000,
                'generated_token_ids': [[0] * 128 for _ in range(1024)],
                'generated_seqs': ['sample'] * 1024,
                'per_sample_scores': scores,
                'sample_quality': sample_quality,
                'generative_ppl': 1.0,
            }), encoding='utf-8')
            physical_points = [index / 128 for index in range(129)]
            (run_dir / 'task1_sampling_diagnostics.json').write_text(
                json.dumps({
                    'initial_noise_schedule': 'base_seed_plus_batch_index',
                    'batches': 128,
                    'initial_state_shape_per_batch': [8, 128, 50257],
                    'initial_state_probe': [0.0],
                    'grid': {
                        'name': (
                            'physical_time_uniform_official_inverse_lut'),
                        'nfe': 128,
                        'update_count': 127,
                        'update_after_last_query': False,
                        'endpoint_update_performed': False,
                        'physical_time_grid': 'linspace(0,1,129)',
                        'physical_time_step': 1 / 128,
                        'tau_pairing': (
                            'official_inverse_lut_gamma_to_alpha'),
                        'query_tau_points': physical_points[:-1],
                        'query_physical_time_points': physical_points[:-1],
                        'update_target_tau_points': physical_points[1:-1],
                        'update_target_physical_time_points': (
                            physical_points[1:-1]),
                        'tau_points': physical_points,
                        'physical_time_points': physical_points,
                    },
                }), encoding='utf-8')
            completed = subprocess.run([
                sys.executable, str(verifier), str(run_dir),
                '--grid', 'uniform_t', '--nfe', '128',
                '--checkpoint-label', 'v1_30k',
                '--checkpoint-path', str(checkpoint),
                '--launcher-log', str(launcher_log),
                '--implementation-commit', 'test-commit',
            ], check=True, capture_output=True, text=True)
            self.assertIn('Validated NFE-refinement completion',
                          completed.stdout)
            completion = json.loads(
                (run_dir / 'completion.json').read_text())
            self.assertEqual(completion['checkpoint_path'], str(checkpoint))
            self.assertTrue(completion['output_streams_closed'])

    def test_nfe_refinement_supervisor_requires_10_of_10_before_shutdown(self):
        runner = SOURCE_ROOT / 'scripts' / \
            'run_task1_a15_v1_30k_nfe_refinement.sh'
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            a15 = root / 'a15.ckpt'
            v1 = root / 'v1.ckpt'
            a15.touch()
            v1.touch()
            invocation_log = root / 'invocations.jsonl'
            evaluator = root / 'fake_eval.sh'
            evaluator.write_text("""#!/usr/bin/env bash
set -euo pipefail
if [[ "${FAIL_GRID_NFE:-}" == "${REFINEMENT_GRID}_${REFINEMENT_NFE}" ]]; then
  exit 7
fi
mkdir -p "$RUN_DIR"
"${PYTHON_BIN:-python}" - "$RUN_DIR" <<'PY'
import json
import os
import sys
from pathlib import Path
run_dir = Path(sys.argv[1])
(run_dir / 'completion.json').write_text(json.dumps({
    'status': 'completed',
    'grid': os.environ['REFINEMENT_GRID'],
    'nfe': int(os.environ['REFINEMENT_NFE']),
    'checkpoint_label': os.environ['CHECKPOINT_LABEL'],
    'implementation_commit': os.environ['IMPLEMENTATION_COMMIT'],
    'initial_state_probe': {'seed': 42},
    'output_streams_closed': True,
    'sampling_grid': {
        'tau_points': [index / int(os.environ['REFINEMENT_NFE'])
                       for index in range(
                           int(os.environ['REFINEMENT_NFE']) + 1)],
        'physical_time_points': [
            index / int(os.environ['REFINEMENT_NFE'])
            for index in range(int(os.environ['REFINEMENT_NFE']) + 1)],
    },
}) + '\\n')
with open(os.environ['INVOCATION_LOG'], 'a', encoding='utf-8') as handle:
    handle.write(json.dumps({
        'grid': os.environ['REFINEMENT_GRID'],
        'nfe': int(os.environ['REFINEMENT_NFE']),
        'checkpoint_label': os.environ['CHECKPOINT_LABEL'],
    }) + '\\n')
PY
""", encoding='utf-8')
            evaluator.chmod(0o755)
            shutdown = root / 'fake_shutdown.sh'
            shutdown.write_text("""#!/usr/bin/env bash
set -euo pipefail
: > "$SHUTDOWN_MARKER"
""", encoding='utf-8')
            shutdown.chmod(0o755)

            base_env = os.environ.copy()
            base_env.update({
                'IMPLEMENTATION_COMMIT': 'test-commit',
                'A15_CHECKPOINT': str(a15),
                'V1_30_CHECKPOINT': str(v1),
                'TASK1_EVAL_ENTRYPOINT': str(evaluator),
                'TASK1_SHUTDOWN_COMMAND': str(shutdown),
                'INVOCATION_LOG': str(invocation_log),
                'AUTO_SHUTDOWN': 'true',
                'PYTHON_BIN': sys.executable,
            })

            success_root = root / 'success'
            success_marker = root / 'success.shutdown'
            success_env = base_env | {
                'RUN_ROOT': str(success_root),
                'SHUTDOWN_MARKER': str(success_marker),
            }
            subprocess.run([str(runner)], env=success_env, check=True,
                           capture_output=True, text=True)
            run_set = json.loads(
                (success_root / 'run_set_completion.json').read_text())
            self.assertEqual(run_set['evaluation_count'], 10)
            self.assertEqual(len(run_set['evaluation_matrix']), 10)
            self.assertEqual(len({
                (item['grid'], item['nfe'], item['checkpoint_label'])
                for item in run_set['evaluation_matrix']}), 10)
            self.assertTrue(success_marker.is_file())
            self.assertEqual(len(invocation_log.read_text().splitlines()), 10)

            failure_root = root / 'failure'
            failure_marker = root / 'failure.shutdown'
            failure_env = base_env | {
                'RUN_ROOT': str(failure_root),
                'SHUTDOWN_MARKER': str(failure_marker),
                'FAIL_GRID_NFE': 'g32_256',
            }
            failed = subprocess.run(
                [str(runner)], env=failure_env, check=False,
                capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertFalse(failure_marker.exists())
            self.assertFalse((failure_root / 'run_set_completion.json').exists())

            failed_sync = root / 'failed_sync.sh'
            failed_sync.write_text("#!/usr/bin/env bash\nexit 9\n",
                                   encoding='utf-8')
            failed_sync.chmod(0o755)
            sync_root = root / 'sync_failure'
            sync_marker = root / 'sync_failure.shutdown'
            sync_env = base_env | {
                'RUN_ROOT': str(sync_root),
                'SHUTDOWN_MARKER': str(sync_marker),
                'TASK1_SYNC_COMMAND': str(failed_sync),
            }
            sync_failed = subprocess.run(
                [str(runner)], env=sync_env, check=False,
                capture_output=True, text=True)
            self.assertNotEqual(sync_failed.returncode, 0)
            self.assertFalse(sync_marker.exists())
            self.assertFalse(
                (sync_root / 'supervisor_completion.json').exists())

    def test_two_new_three_checkpoint_grid_contracts(self):
        first_interval = (SOURCE_ROOT / 'scripts' /
                          'eval_task1_first_interval64_tail_keep.sh').read_text()
        physical_uniform = (SOURCE_ROOT / 'scripts' /
                            'eval_task1_physical_time_uniform.sh').read_text()
        for script in (first_interval, physical_uniform):
            for required in (
                    'NUM_SAMPLES=1024', 'SAMPLING_STEPS=128',
                    'EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"',
                    'DISABLE_EMA=false', 'GEN_PPL_COMPARABLE=false',
                    'TASK1_INITIAL_NOISE_SEED=42',
                    'verify_task1_two_grid_eval.py'):
                self.assertIn(required, script)
        self.assertIn(
            'TASK1_TIME_GRID=first_interval64_tail_keep_4_3_2_1',
            first_interval)
        self.assertIn(
            'TASK1_TIME_GRID=physical_time_uniform_official_inverse_lut',
            physical_uniform)

        for runner_name in (
                'run_task1_first_interval64_a6e4_3ckpt.sh',
                'run_task1_physical_time_uniform_a6e4_3ckpt.sh'):
            runner = (SOURCE_ROOT / 'scripts' / runner_name).read_text()
            for label in ('a_6e4_6k', 'a_6e4_10k', 'a_6e4_15k'):
                self.assertIn(label, runner)
            self.assertIn('pids+=("$!")', runner)
            self.assertIn('if ! wait "$pid"', runner)
            self.assertIn("'parallel_workers': 3", runner)

        supervisor = (SOURCE_ROOT / 'scripts' /
                      'run_task1_two_grid_a6e4_6worker.sh').read_text()
        self.assertIn("'parallel_workers': 6", supervisor)
        self.assertIn('sync', supervisor)
        self.assertIn('/usr/bin/shutdown', supervisor)
        self.assertIn('status_a != 0 || status_b != 0', supervisor)

        verifier = (SOURCE_ROOT / 'scripts' /
                    'verify_task1_two_grid_eval.py').read_text()
        for required in (
                "'a_6e4_6k': 6000", "'a_6e4_10k': 10000",
                "'a_6e4_15k': 15000",
                "expected_counts = [76, 9, 9, 8, 8, 8, 4, 3, 2, 1]",
                "grid['update_count'] == 127",
                "grid['update_after_last_query'] is False",
                "'official_inverse_lut_gamma_to_alpha'"):
            self.assertIn(required, verifier)

    def test_gamma_bin_evaluation_defaults_to_512_sequences_per_bin(self):
        script = (SOURCE_ROOT / 'scripts' /
                  'eval_task1_a_gamma_bins.sh').read_text()
        for required in (
                'SAMPLES_PER_TAU_BIN:-',
                'samples_per_tau_bin="${samples_per_tau_bin:-512}"',
                'num_samples="$(( samples_per_tau_bin * tau_bin_count ))"',
                '+experiment.posterior_samples_per_tau_bin=',
                'validate_task1_gamma_bins.py',
                '--expected-samples-per-bin "$samples_per_tau_bin"'):
            self.assertIn(required, script)
        self.assertIn('Set only one of NUM_SAMPLES or SAMPLES_PER_TAU_BIN',
                      script)

    def test_task1_configs_name_the_only_a_c_difference(self):
        config_dir = SOURCE_ROOT / 'configs' / 'algo'
        variant_a = (config_dir / 'langflow_hybrid_task1_a.yaml').read_text()
        variant_c = (config_dir / 'langflow_hybrid_task1_c.yaml').read_text()
        self.assertIn('model_time_condition: tau', variant_a)
        self.assertIn('model_time_condition: log_nsr', variant_c)
        self.assertIn('token_bias_warmup_steps: 5000', variant_a)
        self.assertIn('optimization_diagnostic_interval_steps: 10', variant_a)
        comparable_a = [
            line for line in variant_a.splitlines()
            if not line.startswith('#')
            and not line.startswith('model_time_condition:')]
        comparable_c = [
            line for line in variant_c.splitlines()
            if not line.startswith('#')
            and not line.startswith('model_time_condition:')]
        self.assertEqual(comparable_a, comparable_c)
        for required in (
                'time_sampling: flm_decoding_error_rate_tau_inverse',
                'classification_prototype_mode: direct_vocab_state',
                'bias_interpretation: matched_flm_vocab_gaussian_direct',
                'optimization_scale: one_half'):
            self.assertIn(required, variant_a)
            self.assertIn(required, variant_c)

    def test_lr_selection_diagnostics_are_structured_and_low_overhead(self):
        model_source = (SOURCE_ROOT / 'langflow_hybrid' / 'model.py').read_text()
        diagnostics_source = (
            SOURCE_ROOT / 'langflow_hybrid' / 'diagnostics.py').read_text()
        manifest_source = (SOURCE_ROOT / 'run_manifest.py').read_text()
        self.assertIn('hybrid_probability_mse', model_source)
        for required in (
                'grad_global_norm_preclip', 'gradient_was_clipped',
                'actual_lr', 'lr_scaled_grad_to_weight_ratio_proxy',
                "'backbone_core', 'input_projection'",
                "'output_classifier', 'time_conditioning'",
                "f'diagnostics/grad_{group_name}_norm_preclip'",
                'nonfinite_loss_event_count',
                'nonfinite_gradient_event_count'):
            self.assertIn(required, diagnostics_source)
        self.assertIn("manifest['optimization_diagnostics']", manifest_source)
        self.assertIn("manifest['nonfinite_counts']", manifest_source)
        self.assertIn("manifest['resume_contract']", manifest_source)
        self.assertIn("model, '_task1_lr_graft_history'", manifest_source)

    def test_task1_uses_official_lut_direct_bias_and_untied_ddit(self):
        algo_source = (SOURCE_ROOT / 'algo.py').read_text()
        model_source = (SOURCE_ROOT / 'langflow_hybrid' / 'model.py').read_text()
        ops_source = (SOURCE_ROOT / 'langflow_hybrid' / 'ops.py').read_text()
        model_config = (SOURCE_ROOT / 'configs' / 'model' /
                        'small_128.yaml').read_text()
        self.assertIn('utils.build_luts(K=self.vocab_size)', algo_source)
        self.assertIn('return self._tau_to_t(u)', model_source)
        self.assertIn('return_target=False, return_noise=preserve_noise',
                      model_source)
        self.assertIn('flm_vocab_gaussian_bias(', model_source)
        self.assertIn('/ (1.0 - safe_t).square()', ops_source)
        self.assertIn('hidden_size: 768', model_config)
        self.assertIn('n_blocks: 12', model_config)
        self.assertIn('n_heads: 12', model_config)
        self.assertIn('tie_word_embeddings: False', model_config)

    def test_task1_entrypoint_uses_configurable_validated_batch(self):
        script = (SOURCE_ROOT / 'scripts' / 'train_task1_a.sh').read_text()
        for required in (
                'configs/task1_a_train.env',
                'task1_load_train_config "$config_path"',
                'task1_resolve_batch "$hardware_profile"',
                'export LOSS_VARIANT=task1_a',
                'export DATA_CONFIG=openwebtext_327m_packed',
                'MAX_STEPS must be in [1, 50000]',
                'train_owt_128_langflow_hybrid.sh'):
            self.assertIn(required, script)
        config = (SOURCE_ROOT / 'configs' / 'task1_a_train.env').read_text()
        for required in (
                'GLOBAL_BATCH_SIZE=64', 'MICRO_BATCH_SIZE=auto',
                'ACCUMULATE_GRAD_BATCHES=auto', 'HARDWARE_PROFILE=h800',
                'OPTIM_BETA2=0.999', 'EMA_DECAY=0.9999'):
            self.assertIn(required, config)
        self.assertNotIn('model=', script)
        self.assertNotIn('hidden_size', script)
        self.assertNotIn('n_blocks', script)

    def test_summary_orders_checkpoints_and_reports_mauve_direction(self):
        module = load_summary_module()
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for step, mauve in ((40000, 0.3), (20000, 0.2)):
                path = Path(directory) / f'{step}.json'
                path.write_text(json.dumps({
                    'checkpoint_global_step': step,
                    'generative_perplexity': 130.0,
                    'mean_sample_unigram_entropy_nats': 5.0,
                    'mauve': mauve,
                }))
                paths.append(path)
            summary = module.summarize(paths)
        self.assertEqual(
            [row['checkpoint_global_step'] for row in summary['metrics']],
            [20000, 40000])
        self.assertTrue(summary['mauve_increased'])
        self.assertEqual(
            summary['targets']['mean_sample_unigram_entropy_nats'],
            'increase')
        self.assertEqual(
            set(summary['metrics'][0]), {
                'checkpoint_global_step', 'generative_perplexity',
                'mean_sample_unigram_entropy_nats', 'mauve'})


if __name__ == '__main__':
    unittest.main()
