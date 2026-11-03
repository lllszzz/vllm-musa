# SPDX-License-Identifier: Apache-2.0
"""Keep GLM graph top-k conversion isolated to the MUSA indexer path."""

from pathlib import Path


PATCH = (
    Path(__file__).resolve().parents[1]
    / "vllm_musa/patches/series/"
    / "0057-MUSA-run-GLM-sparse-index-conversion-on-compute-stre.patch"
)


def test_sparse_index_conversion_patch_touches_only_expected_source():
    source = PATCH.read_text()
    files = [line for line in source.splitlines() if line.startswith("diff --git ")]
    assert files == [
        "diff --git a/vllm/v1/attention/backends/mla/index_group.py "
        "b/vllm/v1/attention/backends/mla/index_group.py"
    ]


def test_glm_compute_stream_is_musa_and_indexer_only():
    source = PATCH.read_text()
    assert "+from vllm.platforms import current_platform" in source
    assert "+        if current_platform.is_musa() and self.has_indexer:" in source
    assert "+            if layer_index == 0:" in source
    assert "triton_convert_req_index_to_global_index(" in source
    # No extra graph-disable flag, global stream mutation or new kernel.
    assert "VLLM_USE_DEEP_GEMM_E8M0" not in source
    assert "set_stream(" not in source
    assert "cudagraph_utils.py" not in source


def test_non_musa_side_stream_dependency_remains():
    source = PATCH.read_text()
    assert "+        else:" in source
    assert "+                    self.side_stream.wait_event(self.logical_topk_ready)" in source
    assert "+                    self.side_stream.wait_stream(compute_stream)" in source
    assert "+                with self.side_stream:" in source
    assert "+                    self.physical_topk_ready.record(self.side_stream)" in source
    assert "+            compute_stream.wait_event(self.physical_topk_ready)" in source
