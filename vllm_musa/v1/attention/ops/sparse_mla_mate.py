# SPDX-License-Identifier: Apache-2.0
"""GLM DSA sparse-MLA adapter backed by the pinned MATE stack."""

from __future__ import annotations

from mate.sparse_mla.tilelang.sparse_mla_prefill import sparse_mla_prefill_fwd

from vllm.logger import init_logger

from vllm_musa.optimization_contract import GLMSparseMLARoute

logger = init_logger(__name__)

try:
    from mate.sparse_mla.tilelang.sparse_mla_model1_fwd_pipelined import (
        sparse_mla_fwd_interface_model1 as _sparse_mla_fwd_model1,
    )
except ImportError:
    _sparse_mla_fwd_model1 = None


def sparse_mla_fwd_bf16(
    q,
    kv,
    indices,
    sm_scale,
    d_v=512,
    attn_sink=None,
    topk_length=None,
    glm_sparse_mla_route: int = int(GLMSparseMLARoute.DISABLED),
):
    """Dispatch the selected GLM sparse-MLA route through MATE.

    FLASH_MODEL1 uses the model1 interface directly and preserves its complete
    ``(output, max_logits, lse)`` return tuple.  The ordinary GLM prefill route
    retains the existing MATE prefill output-only contract.
    """
    if glm_sparse_mla_route == int(GLMSparseMLARoute.FLASH_MODEL1):
        if _sparse_mla_fwd_model1 is None:
            raise RuntimeError(
                "GLM-5.3-Flash model1 requires the pinned MATE model1 interface"
            )
        logger.info_once(
            "GLM-5.3-Flash MATE model1 CALL: "
            "q=%s dtype=%s kv=%s dtype=%s indices=%s dtype=%s "
            "sm_scale=%s topk_length=%s d_v=%s return_max_logits=%s "
            "is_persistence=%s attn_sink=%s",
            tuple(q.shape),
            q.dtype,
            tuple(kv.shape),
            kv.dtype,
            tuple(indices.shape),
            indices.dtype,
            sm_scale,
            (None if topk_length is None else (
                tuple(topk_length.shape), topk_length.dtype, topk_length.device
            )),
            512,
            True,
            False,
            attn_sink is not None,
        )
        return _sparse_mla_fwd_model1(
            q,
            kv,
            indices,
            topk_length=topk_length,
            sm_scale=sm_scale,
            attn_sink=attn_sink,
            d_v=512,
            return_max_logits=True,
            is_persistence=False,
        )

    return sparse_mla_prefill_fwd(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        d_v=d_v,
        attn_sink=attn_sink,
        topk_length=topk_length,
    )[0]
