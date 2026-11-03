from __future__ import annotations

from dataclasses import replace

from .types import (
    ExecutionSignature,
    GLMSparseMLARoute,
    ModelFamily,
    ModelRole,
    ModelSignature,
    MusaOptimizationContract,
    OptimizationFeature,
)


def _is_glm5_flash(model: ModelSignature) -> bool:
    outer_architectures = set(model.outer_architectures)
    if outer_architectures or model.outer_model_type is not None:
        return (
            "Glm5NextForConditionalGeneration" in outer_architectures
            or "Glm5NextForCausalLM" in outer_architectures
            or model.outer_model_type in {"glm5_next", "glm5_next_text"}
        )
    # Some construction paths do not retain outer metadata.  The nested
    # architecture/model type can identify the outer model only in that case;
    # ``glm5_next_text`` alone is deliberately not sufficient.
    return (
        "Glm5NextForConditionalGeneration" in model.architectures
        or "Glm5NextForCausalLM" in model.architectures
        or model.model_type == "glm5_next"
    )


def resolve_glm_sparse_mla_route(
    contract: MusaOptimizationContract | None,
) -> GLMSparseMLARoute:
    if contract is None:
        return GLMSparseMLARoute.DISABLED
    if contract.prefers(OptimizationFeature.GLM5_FLASH_SPARSE_MLA_MODEL1):
        return GLMSparseMLARoute.FLASH_MODEL1
    if contract.prefers(OptimizationFeature.GLM5_SPARSE_MLA_MATE_PREFILL):
        return GLMSparseMLARoute.MATE_PREFILL
    return GLMSparseMLARoute.DISABLED


def _is_glm5_dsa(model: ModelSignature) -> bool:
    text_architectures = set(model.text_architectures or model.architectures)
    text_is_dsa = (
        "GlmMoeDsaForCausalLM" in text_architectures
        or model.model_type in {"glm_moe_dsa", "glm5_next_text"}
    )
    if not text_is_dsa:
        return False
    if _is_glm5_flash(model):
        return model.uses_mla is True and model.index_topk is not None
    return True


def resolve_glm_contract(
    model: ModelSignature,
    execution: ExecutionSignature,
) -> MusaOptimizationContract | None:
    if not _is_glm5_dsa(model):
        return None

    model = replace(model, family=ModelFamily.GLM5, role=ModelRole.TEXT)
    preferred = {
        OptimizationFeature.GLM5_EAGER_MCCL_INIT,
    }
    if _is_glm5_flash(model):
        if model.num_experts == 288 and model.num_experts_per_tok == 8:
            preferred.add(OptimizationFeature.GLM5_MOE_ROUTER_288)
        if model.index_topk == 2048:
            preferred.add(OptimizationFeature.GLM5_SPARSE_INDEXER_TOPK)
    if (
        model.dtype == "bfloat16"
        and model.quantization == "fp8"
        and model.uses_mla is True
        and model.index_topk == 2048
    ):
        preferred.add(OptimizationFeature.GLM5_SPARSE_MLA_MATE_PREFILL)
        if _is_glm5_flash(model):
            preferred.update(
                {
                    OptimizationFeature.GLM5_FLASH_SPARSE_MLA_MODEL1,
                }
            )

    contract = MusaOptimizationContract(
        model=model,
        execution=execution,
        profile="glm5",
        supported_features=frozenset(preferred),
        preferred_features=frozenset(preferred),
    )
    return contract
