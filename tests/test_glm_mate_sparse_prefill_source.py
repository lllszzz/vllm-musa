# SPDX-License-Identifier: Apache-2.0
"""Source contract for GLM MATE sparse-prefill routing."""

import ast
from pathlib import Path

import pytest

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "vllm_musa/v1/attention/backends/mla/flashmla_sparse.py"
)
PATCH_0056 = (
    Path(__file__).resolve().parents[1]
    / "vllm_musa/patches/series/0056-MUSA-pass-GLM-sparse-MLA-route-into-FlashMLA.patch"
)


def test_patch_0056_forwards_contract_route_into_backend() -> None:
    patch = PATCH_0056.read_text()
    assert "resolve_glm_sparse_mla_route" in patch
    assert "self._glm_sparse_mla_route = resolve_glm_sparse_mla_route(" in patch
    assert "glm_sparse_mla_route=self._glm_sparse_mla_route.value" in patch


def test_flashmla_route_is_musa_gated_and_non_musa_call_has_no_route_kwarg() -> None:
    patch = PATCH_0056.read_text()
    assert "if current_platform.is_musa():" in patch
    assert "glm_sparse_mla_route=self._glm_sparse_mla_route.value" in patch
    assert "            glm_sparse_mla_route=self._glm_sparse_mla_route.value," in patch
    assert "                out=out,\n+            )" in patch
    source = SOURCE.read_text()
    tree = ast.parse(source)
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert "_can_use_glm_mate_sparse_prefill" in functions
    assert "_is_glm_dsa_model" not in functions


def test_glm_detection_checks_scalar_route() -> None:
    source = SOURCE.read_text()
    assert "GLMSparseMLARoute" in source
    assert "glm_sparse_mla_route: int = int(GLMSparseMLARoute.DISABLED)" in source


def test_mate_dispatch_precedes_generic_tilelang_dispatch() -> None:
    source = SOURCE.read_text()
    mate = source.index("use_mate_adapter = _can_use_glm_mate_sparse_prefill(")
    generic = source.index("if _can_use_tilelang_sparse_prefill(", mate)
    assert mate < generic


def test_glm_mate_predicate_includes_h64_model1_shape() -> None:
    source = SOURCE.read_text()
    assert "q.shape[1] != 64" in source
    assert "q.shape[2] != 576 or indices.shape[-1] != 2048" in source
    assert "q.shape[2] != 512 or indices.shape[-1] not in (2048, 2176)" in source
    assert "kv.shape[2] != q.shape[2]" in source


def test_vllm_mate_adapter_requests_nonpersistent_model1() -> None:
    adapter = (
        SOURCE.parents[2] / "ops/sparse_mla_mate.py"
    ).read_text()
    assert "is_persistence=False" in adapter
    assert "GLMSparseMLARoute.FLASH_MODEL1" in adapter
    assert "MusaOptimizationContract" not in adapter
    assert "except ImportError:" in adapter
    assert "except (ImportError, AttributeError):" not in adapter


def test_musa_impl_accepts_scalar_route_before_runtime_dispatch() -> None:
    source = SOURCE.read_text()
    assert "def _musa_backend_sparse_fwd(" in source
    assert "glm_sparse_mla_route: int = int(GLMSparseMLARoute.DISABLED)" in source
    assert "MusaOptimizationContract" not in source


def test_route_scalar_ablation_controls_model1_dispatch(monkeypatch) -> None:
    try:
        import torch
        from vllm_musa.optimization_contract import GLMSparseMLARoute
        from vllm_musa.v1.attention.backends.mla import flashmla_sparse as backend
        from vllm_musa.v1.attention.ops import sparse_mla_mate as mate_adapter
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(f"MUSA runtime imports unavailable: {exc}")

    q = torch.empty((7, 64, 512), device="meta", dtype=torch.bfloat16)
    kv = torch.empty((123, 1, 512), device="meta", dtype=torch.bfloat16)
    indices = torch.empty((7, 1, 2176), device="meta", dtype=torch.int32)
    lengths = torch.empty((7,), device="meta", dtype=torch.int32)
    expected = (q, q, q)
    calls = []

    def fake_model1(*args, **kwargs):
        calls.append("model1")
        return expected

    def fake_flashmla(**kwargs):
        calls.append("flashmla")
        return expected

    monkeypatch.setenv("VLLM_MUSA_SPARSE_MLA_TILELANG", "0")
    monkeypatch.setattr(mate_adapter, "_sparse_mla_fwd_model1", fake_model1)
    monkeypatch.setattr(backend, "_musa_flash_mla_sparse_fwd", fake_flashmla)
    model1 = backend._musa_backend_sparse_fwd(
        q, kv, indices, 1.0, topk_length=lengths,
        glm_sparse_mla_route=int(GLMSparseMLARoute.FLASH_MODEL1),
    )
    assert model1 is expected
    assert calls == ["model1"]
    calls.clear()
    disabled = backend._musa_backend_sparse_fwd(
        q, kv, indices, 1.0, topk_length=lengths,
        glm_sparse_mla_route=int(GLMSparseMLARoute.DISABLED),
    )
    assert disabled is expected
    assert calls == ["flashmla"]


def test_generic_tilelang_sparse_predicate_remains_callable(monkeypatch) -> None:
    try:
        import torch
        from vllm_musa.v1.attention.backends.mla.flashmla_sparse import (
            _can_use_tilelang_sparse_prefill,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(f"MUSA runtime imports unavailable: {exc}")

    q = torch.empty((7, 64, 576), device="meta", dtype=torch.bfloat16)
    kv = torch.empty((123, 1, 576), device="meta", dtype=torch.bfloat16)
    indices = torch.empty((7, 1, 2048), device="meta", dtype=torch.int32)
    topk_length = torch.empty((7,), device="meta", dtype=torch.int32)
    monkeypatch.setenv("VLLM_MUSA_SPARSE_MLA_TILELANG", "1")
    assert _can_use_tilelang_sparse_prefill(
        q, kv, indices, 512, None, topk_length
    )
    monkeypatch.setenv("VLLM_MUSA_SPARSE_MLA_TILELANG", "0")
    assert not _can_use_tilelang_sparse_prefill(
        q, kv, indices, 512, None, topk_length
    )


def test_glm_mate_predicate_runtime_shape_guard(monkeypatch) -> None:
    try:
        import torch
        from vllm_musa.optimization_contract import GLMSparseMLARoute
        from vllm_musa.v1.attention.backends.mla.flashmla_sparse import (
            _can_use_glm_mate_sparse_prefill,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(f"MUSA runtime imports unavailable: {exc}")

    q = torch.empty((7, 64, 512), device="meta", dtype=torch.bfloat16)
    kv = torch.empty((123, 1, 512), device="meta", dtype=torch.bfloat16)
    indices = torch.empty((7, 1, 2048), device="meta", dtype=torch.int32)
    topk_length = torch.empty((7,), device="meta", dtype=torch.int32)
    route = int(GLMSparseMLARoute.FLASH_MODEL1)

    assert _can_use_glm_mate_sparse_prefill(
        q, kv, indices, 512, None, topk_length, route
    )
    monkeypatch.setenv("VLLM_MUSA_SPARSE_MLA_TILELANG", "0")
    assert _can_use_glm_mate_sparse_prefill(
        q, kv, indices, 512, None, topk_length, route
    )
    assert not _can_use_glm_mate_sparse_prefill(
        torch.empty((7, 32, 512), device="meta", dtype=torch.bfloat16),
        kv,
        indices,
        512,
        None,
        topk_length,
        route,
    )
    assert not _can_use_glm_mate_sparse_prefill(
        q,
        kv,
        torch.empty((7, 1, 1024), device="meta", dtype=torch.int32),
        512,
        None,
        topk_length,
        route,
    )
    assert _can_use_glm_mate_sparse_prefill(
        torch.empty((7, 64, 576), device="meta", dtype=torch.bfloat16),
        torch.empty((123, 1, 576), device="meta", dtype=torch.bfloat16),
        torch.empty((7, 1, 2048), device="meta", dtype=torch.int32),
        512,
        None,
        topk_length,
        int(GLMSparseMLARoute.MATE_PREFILL),
    )
    assert _can_use_glm_mate_sparse_prefill(
        q,
        kv,
        torch.empty((7, 1, 2176), device="meta", dtype=torch.int32),
        512,
        None,
        topk_length,
        route,
    )
    assert not _can_use_glm_mate_sparse_prefill(
        q,
        kv,
        indices,
        512,
        None,
        torch.empty((1, 7), device="meta", dtype=torch.int32),
        route,
    )
