"""Stable name-to-class lookup for training algorithms."""

import algo
import task1_tvm_ce
import task1_posterior_tvm
import task1_no_sc_mse_teacher
import task1_tvm_50k_final


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
    'task1_tvm_ce': task1_tvm_ce.Task1TVMCE,
    'task1_tvm_endpoint500': task1_tvm_ce.Task1TVMEndpoint500,
    'task1_tvm_sc_repair': task1_tvm_ce.Task1TVMSCRepair,
    'task1_tvm_joint_j0': task1_tvm_ce.Task1TVMJointJ0,
    'task1_tvm_joint_j1': task1_tvm_ce.Task1TVMJointJ1,
    'task1_posterior_tvm': task1_posterior_tvm.Task1PosteriorTVM,
    'task1_no_sc_mse_teacher': (
        task1_no_sc_mse_teacher.Task1NoSCMSETeacher),
    'task1_tvm_50k_final': task1_tvm_50k_final.Task1TVM50KFinal,
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
