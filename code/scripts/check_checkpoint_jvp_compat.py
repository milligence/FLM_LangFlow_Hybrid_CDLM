#!/usr/bin/env python3
"""Probe torch.func JVP/backward compatibility with activation checkpointing."""

import json

import torch
from torch.utils.checkpoint import checkpoint


def probe(use_reentrant):
    x = torch.randn(4, requires_grad=True)
    direction = torch.randn_like(x)

    def function(value):
        return checkpoint(
            lambda tensor: tensor.sin().square(), value,
            use_reentrant=use_reentrant)

    try:
        primal, tangent = torch.func.jvp(
            function, (x,), (direction,))
        loss = (primal + tangent).sum()
        gradient, = torch.autograd.grad(loss, x)
        return {
            'use_reentrant': use_reentrant,
            'status': 'passed',
            'gradient_finite': bool(torch.isfinite(gradient).all()),
        }
    except Exception as error:  # compatibility evidence must retain type/text
        return {
            'use_reentrant': use_reentrant,
            'status': 'unsupported',
            'error_type': type(error).__name__,
            'error': str(error),
        }


print(json.dumps({
    'torch_version': torch.__version__,
    'results': [probe(True), probe(False)],
}, indent=2))
