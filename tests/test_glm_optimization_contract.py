from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_musa.optimization_contract import (
    GLMSparseMLARoute,
    OptimizationFeature,
    resolve_glm_contract,
    resolve_glm_sparse_mla_route,
    resolve_optimization_contract,
)


def _config(
    architecture="GlmMoeDsaForCausalLM",
    model_type="glm_moe_dsa",
    outer_architecture=None,
    outer_model_type=None,
    text_architecture=None,
    text_model_type=None,
    dtype="bfloat16",
    quantization="fp8",
    use_mla=True,
    index_topk=2048,
    num_experts=288,
    num_experts_per_tok=8,
):
    outer_architecture = outer_architecture or architecture
    outer_model_type = outer_model_type or model_type
    text_architecture = text_architecture or architecture
    text_model_type = text_model_type or model_type
    text = SimpleNamespace(
        architectures=[text_architecture],
        model_type=text_model_type,
        hidden_size=4096,
        intermediate_size=13696,
        num_hidden_layers=78,
        num_attention_heads=64,
        num_key_value_heads=1,
        head_dim=128,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        index_topk=index_topk,
        quantization_config={"quant_method": quantization},
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            architectures=[outer_architecture],
            model_type=outer_model_type,
            hf_text_config=text,
            dtype=dtype,
            quantization=quantization,
            use_mla=use_mla,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=8,
            pipeline_parallel_size=2,
            data_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=64),
        scheduler_config=SimpleNamespace(max_num_seqs=1, async_scheduling=True),
        speculative_config=None,
        quant_config=None,
    )


def test_glm_sparse_mla_route_without_contract_is_disabled() -> None:
    assert resolve_glm_sparse_mla_route(None) == GLMSparseMLARoute.DISABLED


def test_glm_contract_prefers_both_glm_features() -> None:
    contract = resolve_optimization_contract(_config())
    assert contract.prefers(OptimizationFeature.GLM5_EAGER_MCCL_INIT)
    assert contract.prefers(OptimizationFeature.GLM5_SPARSE_MLA_MATE_PREFILL)
    assert resolve_glm_sparse_mla_route(contract) == GLMSparseMLARoute.MATE_PREFILL


@pytest.mark.parametrize(
    "override",
    [
        {"dtype": "float16"},
        {"quantization": "compressed-tensors"},
        {"use_mla": False},
        {"index_topk": 1024},
    ],
)
def test_glm_mate_sparse_prefill_requires_full_signature(override) -> None:
    contract = resolve_optimization_contract(_config(**override))
    assert contract.prefers(OptimizationFeature.GLM5_EAGER_MCCL_INIT)
    assert not contract.prefers(OptimizationFeature.GLM5_SPARSE_MLA_MATE_PREFILL)


def test_glm53_flash_contract_prefers_model1_nonpersistent() -> None:
    contract = resolve_optimization_contract(
        _config(
            architecture="Glm5NextForConditionalGeneration",
            model_type="glm5_next",
            text_architecture="GlmMoeDsaForCausalLM",
            text_model_type="glm5_next_text",
        )
    )
    assert contract.prefers(OptimizationFeature.GLM5_SPARSE_MLA_MATE_PREFILL)
    assert contract.prefers(
        OptimizationFeature.GLM5_FLASH_SPARSE_MLA_MODEL1
    )
    assert contract.prefers(OptimizationFeature.GLM5_SPARSE_INDEXER_TOPK)
    assert contract.prefers(OptimizationFeature.GLM5_MOE_ROUTER_288)
    assert resolve_glm_sparse_mla_route(contract) == GLMSparseMLARoute.FLASH_MODEL1


def test_ordinary_glm_does_not_prefer_flash_model1_route() -> None:
    contract = resolve_optimization_contract(_config())
    assert not contract.prefers(
        OptimizationFeature.GLM5_FLASH_SPARSE_MLA_MODEL1
    )
    assert resolve_glm_sparse_mla_route(contract) == GLMSparseMLARoute.MATE_PREFILL


@pytest.mark.parametrize(
    "override",
    [
        {"dtype": "float16"},
        {"quantization": "compressed-tensors"},
        {"use_mla": False},
        {"index_topk": 1024},
    ],
)
def test_glm53_flash_model1_route_requires_full_signature(override) -> None:
    contract = resolve_optimization_contract(
        _config(
            architecture="Glm5NextForConditionalGeneration",
            model_type="glm5_next",
            text_architecture="GlmMoeDsaForCausalLM",
            text_model_type="glm5_next_text",
            **override,
        )
    )
    assert not contract.prefers(
        OptimizationFeature.GLM5_FLASH_SPARSE_MLA_MODEL1
    )


@pytest.mark.parametrize(
    "override, feature",
    [
        ({"num_experts": 256}, OptimizationFeature.GLM5_MOE_ROUTER_288),
        ({"num_experts_per_tok": 4}, OptimizationFeature.GLM5_MOE_ROUTER_288),
        ({"index_topk": 1024}, OptimizationFeature.GLM5_SPARSE_INDEXER_TOPK),
    ],
)
def test_glm53_flash_capabilities_require_exact_shape_metadata(override, feature) -> None:
    contract = resolve_optimization_contract(
        _config(
            architecture="Glm5NextForConditionalGeneration",
            model_type="glm5_next",
            text_architecture="GlmMoeDsaForCausalLM",
            text_model_type="glm5_next_text",
            **override,
        )
    )
    assert not contract.prefers(feature)


def test_glm53_router_capability_is_contract_gated() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "vllm_musa/model_executor/layers/fused_moe/router/grouped_topk_router.py"
    ).read_text()
    assert "GLM5_MOE_ROUTER_288" in source
    assert "use_glm5_moe_router" in source
    assert "gating_output.shape[1] == 288 and use_glm5_moe_router" in source


def test_glm53_flash_non_dsa_text_config_fails_closed() -> None:
    contract = resolve_optimization_contract(
        _config(
            architecture="Glm5NextForConditionalGeneration",
            model_type="glm5_next",
            text_architecture="SomeDenseTextModel",
            text_model_type="some_dense_text",
        )
    )
    assert resolve_glm_contract(contract.model, contract.execution) is None


def test_glm53_flash_mismatched_outer_metadata_fails_closed() -> None:
    contract = resolve_optimization_contract(
        _config(
            architecture="SomeOuterModel",
            model_type="some_outer_model",
            text_architecture="GlmMoeDsaForCausalLM",
            text_model_type="glm5_next_text",
        )
    )
    assert not contract.prefers(OptimizationFeature.GLM5_FLASH_SPARSE_MLA_MODEL1)
    assert not contract.prefers(OptimizationFeature.GLM5_SPARSE_INDEXER_TOPK)
    assert not contract.prefers(OptimizationFeature.GLM5_MOE_ROUTER_288)
    assert resolve_glm_sparse_mla_route(contract) == GLMSparseMLARoute.MATE_PREFILL


def test_standalone_glm5_next_text_fails_closed() -> None:
    contract = resolve_optimization_contract(
        _config(
            architecture="GlmMoeDsaForCausalLM",
            model_type="glm5_next_text",
        )
    )
    model = replace(
        contract.model,
        outer_architectures=(),
        outer_model_type=None,
    )
    resolved = resolve_glm_contract(model, contract.execution)
    assert resolved is not None
    assert not resolved.prefers(OptimizationFeature.GLM5_FLASH_SPARSE_MLA_MODEL1)
    assert not resolved.prefers(OptimizationFeature.GLM5_SPARSE_INDEXER_TOPK)
    assert not resolved.prefers(OptimizationFeature.GLM5_MOE_ROUTER_288)


def test_non_glm_contract_fails_closed() -> None:
    contract = resolve_optimization_contract(
        _config(architecture="Qwen3ForCausalLM", model_type="qwen3")
    )
    assert not contract.prefers(OptimizationFeature.GLM5_EAGER_MCCL_INIT)
    assert not contract.prefers(OptimizationFeature.GLM5_SPARSE_MLA_MATE_PREFILL)
    assert resolve_glm_sparse_mla_route(contract) == GLMSparseMLARoute.DISABLED
    assert resolve_glm_contract(contract.model, contract.execution) is None
