"""Stable name-to-class lookup for training algorithms."""

import algo


_ALGORITHMS = {
    'ar': algo.AR,
    'mdlm': algo.MDLM,
    'duo_base': algo.DUO_BASE,
    'duo': algo.DUO,
    'flm': algo.FLM,
    'langflow_flm_hybrid': algo.LangFlowFLMHybrid,
    'fmlm_twomodel': algo.FMLM_TwoModel,
    'fmlm_twostage': algo.FMLM_TwoStage,
    'fmlm': algo.FMLM,
    'd3pm': algo.D3PMAbsorb,
    'sedd': algo.SEDDAbsorb,
    'distillation': algo.Distillation,
    'rectification': algo.Rectification,
}


def get_algorithm_class(name):
    """Return the existing algorithm class or raise the legacy error."""
    try:
        return _ALGORITHMS[name]
    except KeyError:
        raise ValueError(f'Invalid algorithm name: {name}') from None
