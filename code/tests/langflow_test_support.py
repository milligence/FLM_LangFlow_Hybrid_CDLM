"""Small harnesses shared by responsibility-focused LangFlow tests."""

from types import SimpleNamespace

import torch
import torch.nn.functional as F

import algo


class SamplingHarness(torch.nn.Module):
    generate_samples = algo.LangFlowFLMHybrid.generate_samples

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = SimpleNamespace(
            model=SimpleNamespace(hidden_size=3),
            sampling=SimpleNamespace(steps=4))
        self.num_tokens = 6
        self.vocab_size = 5
        self.gumbel_loc = 4.723
        self.gumbel_scale = 0.852
        self.gumbel_cutoff = 1e-5
        self.last_sampling_nfe = 0
        self.weight = torch.randn(self.vocab_size, 3)
        self.self_conditions = []

    @property
    def device(self):
        return self.anchor.device

    def embedding_weight(self):
        return self.weight

    def embed_probabilities(self, probabilities):
        return probabilities @ self.weight

    def _current_token_bias_weight(self):
        return 1.0

    def _forward_logits(self, z_gamma, gamma, x_self_cond=None,
                        bias_weight=None, physical_t=None):
        del gamma, bias_weight, physical_t
        self.self_conditions.append(x_self_cond)
        return torch.zeros(
            z_gamma.shape[0], z_gamma.shape[1], self.vocab_size,
            device=z_gamma.device)


class VocabSamplingHarness(SamplingHarness):
    _generate_vocab_state_samples = (
        algo.LangFlowFLMHybrid._generate_vocab_state_samples)
    _task1_vocab_sampling_grid = (
        algo.LangFlowFLMHybrid._task1_vocab_sampling_grid)
    _task1_tau_box_physical_equal_grid = (
        algo.LangFlowFLMHybrid._task1_tau_box_physical_equal_grid)
    _task1_first_interval_tail_balanced_grid = (
        algo.LangFlowFLMHybrid._task1_first_interval_tail_balanced_grid)
    _task1_first_interval64_tail_keep_grid = (
        algo.LangFlowFLMHybrid._task1_first_interval64_tail_keep_grid)
    _task1_physical_time_uniform_grid = (
        algo.LangFlowFLMHybrid._task1_physical_time_uniform_grid)
    _task1_initialize_sampling_diagnostics = (
        algo.LangFlowFLMHybrid._task1_initialize_sampling_diagnostics)
    _task1_record_sampling_diagnostic = (
        algo.LangFlowFLMHybrid._task1_record_sampling_diagnostic)
    task1_finalize_sampling_diagnostics = (
        algo.LangFlowFLMHybrid.task1_finalize_sampling_diagnostics)

    def __init__(self):
        super().__init__()
        self.state_space = 'vocab'
        self.flm_time_eps = 1e-3
        self.model_time_condition = 'tau'
        self.state_shapes = []
        self.time_records = []
        self.tokenizer = SimpleNamespace(
            convert_ids_to_tokens=lambda ids: [str(index) for index in ids],
            decode=lambda ids: str(ids[0]))

    def _task1_physical_time(self, u):
        return u.square()

    def _t_to_tau(self, t):
        return t.sqrt()

    def _forward_logits(self, state, gamma, x_self_cond=None,
                        bias_weight=None, physical_t=None):
        self.state_shapes.append(tuple(state.shape))
        self.time_records.append((gamma.detach().clone(),
                                  physical_t.detach().clone()))
        return super()._forward_logits(
            state, gamma, x_self_cond=x_self_cond,
            bias_weight=bias_weight, physical_t=physical_t)


class TokenBiasHarness:
    _current_token_bias_weight = (
        algo.LangFlowFLMHybrid._current_token_bias_weight)

    def __init__(self, mode, global_step, loaded_step=None):
        self.config = SimpleNamespace(mode=mode)
        self.global_step = global_step
        self.token_bias_warmup_steps = 5000
        self.token_bias_schedule = 'warmup'
        if loaded_step is not None:
            self._loaded_checkpoint_global_step = loaded_step


class TrainingTimeHarness(torch.nn.Module):
    _sample_task1_training_times = (
        algo.LangFlowFLMHybrid._sample_task1_training_times)
    _task1_training_time_spec = (
        algo.LangFlowFLMHybrid._task1_training_time_spec)

    def __init__(self, mode):
        super().__init__()
        self.anchor = torch.nn.Parameter(
            torch.zeros(()), requires_grad=False)
        self.training_time_sampling = mode
        self.global_step = 15000
        self.config = SimpleNamespace(
            seed=1,
            loader=SimpleNamespace(global_batch_size=256, batch_size=32),
            trainer=SimpleNamespace(
                accumulate_grad_batches=8, devices=1, num_nodes=1),
            experiment=SimpleNamespace(training_rng_seed=12345))
        self._task1_training_generators = {}
        self._task1_pending_training_rng_state = None
        self._task1_cached_global_time_plan = None
        self.flm_time_eps = 1e-5
        self.uniform_call = None

    @property
    def device(self):
        return self.anchor.device

    def _sample_t_interval(
            self, batch_size, accumulation_step, t_min=0.0, t_max=1.0):
        self.uniform_call = (
            batch_size, accumulation_step, t_min, t_max)
        return torch.linspace(
            0.0, 0.99, batch_size, device=self.device)

    def _task1_physical_time(self, tau):
        return tau.square()

    def _t_to_tau(self, physical_t):
        return physical_t.sqrt()

    _task1_training_generator = (
        algo.LangFlowFLMHybrid._task1_training_generator)


class DiagnosticHarness(torch.nn.Module):
    _accumulate_gamma_bins = algo.LangFlowFLMHybrid._accumulate_gamma_bins
    _sample_validation_q = algo.LangFlowFLMHybrid._sample_validation_q
    _validation_tau_edges = algo.LangFlowFLMHybrid._validation_tau_edges
    _frequency_group_ids = algo.LangFlowFLMHybrid._frequency_group_ids
    embedding_nearest_neighbor_diagnostic = (
        algo.LangFlowFLMHybrid.embedding_nearest_neighbor_diagnostic)
    _posterior_diagnostic_names = staticmethod(
        algo.LangFlowFLMHybrid._posterior_diagnostic_names)

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = SimpleNamespace(algo=SimpleNamespace(
            validation_tau_stratified=True,
            validation_tau_bin_count=10))
        embedding = torch.nn.Parameter(torch.randn(12, 4))
        self.backbone = SimpleNamespace(
            vocab_embed=SimpleNamespace(embedding=embedding))
        self.vocab_size = 12
        self._validation_gamma_bins = torch.zeros(
            10, len(self._posterior_diagnostic_names()) + 2,
            dtype=torch.float64)
        self._validation_posterior_rows = None
        self._geometry_sample_indices = torch.arange(12)
        self._initial_geometry_unit_vectors = F.normalize(
            embedding.detach().float(), dim=-1)
        self._training_token_counts = torch.zeros(12, dtype=torch.long)

    def embedding_weight(self):
        return F.normalize(
            self.backbone.vocab_embed.embedding.float(), dim=-1) * 2.0

    @property
    def device(self):
        return self.anchor.device


class NoBiasForwardHarness(torch.nn.Module):
    _forward_logits = algo.LangFlowFLMHybrid._forward_logits
    _state_embedding = algo.LangFlowFLMHybrid._state_embedding
    _record_validation_logit_stats = (
        algo.LangFlowFLMHybrid._record_validation_logit_stats)
    _update_tensor_moments = staticmethod(
        algo.LangFlowFLMHybrid._update_tensor_moments)

    def __init__(self):
        super().__init__()
        self.state_space = 'embedding'
        self.backbone = lambda z, gamma, **kwargs: z @ torch.ones(
            z.shape[-1], 7, device=z.device)
        self._validation_logit_stats = None

    def _process_sigma(self, gamma):
        return gamma

    def _current_token_bias_weight(self):
        return 0.0

    def classification_prototype_weight(self):
        raise AssertionError('No-bias arm touched the bias prototype.')


class DirectVocabForwardHarness(torch.nn.Module):
    _forward_logits = algo.LangFlowFLMHybrid._forward_logits
    _state_embedding = algo.LangFlowFLMHybrid._state_embedding
    _record_validation_logit_stats = (
        algo.LangFlowFLMHybrid._record_validation_logit_stats)
    _update_tensor_moments = staticmethod(
        algo.LangFlowFLMHybrid._update_tensor_moments)

    def __init__(self):
        super().__init__()
        self.state_space = 'vocab'
        self.flm_time_eps = 1e-5
        self.input_projection = torch.randn(5, 3)
        self.backbone = lambda hidden, model_time, **kwargs: torch.zeros(
            hidden.shape[0], hidden.shape[1], 5, device=hidden.device)
        self._validation_logit_stats = None

    def _process_sigma(self, model_time):
        return model_time

    def _task1_contract_smoke_enabled(self):
        return False

    def embed_probabilities(self, state):
        return state @ self.input_projection

    def classification_prototype_weight(self):
        raise AssertionError('Task1 direct bias touched a learned prototype.')
