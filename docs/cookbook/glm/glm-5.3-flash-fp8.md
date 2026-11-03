# GLM-5.3-Flash FP8 on vLLM-MUSA v0.32.0

This recipe is for **GLM-5.3-Flash**, not the two-node GLM-5.3 FP8 model in
[the separate GLM-5.3 recipe](glm-5.3-fp8.md). Do not copy that model's
v0.28.0-dev prerequisite, PP2/EP topology, or no-thinking template override
into this single-node Flash deployment.

## Prerequisites

- vLLM-MUSA v0.32.0 at the pinned vLLM revision with the GLM-5.3-Flash source
  patches, including the MUSA-only sparse-index conversion fix in series entry
  `0211`. Use the matching vLLM-MUSA plugin and MUSA native extensions.
- The GLM-5.3-Flash checkpoint available locally, for example at
  `/models/GLM-5.3-Flash` (replace this example with your actual mounted path).
- One leased node with eight S5000 GPUs, a compatible MUSA driver/runtime and
  enough memory for the FP8 checkpoint and the 16,384-token context.

## Single-node TP8 graph profile

```bash
MODEL_PATH=/models/GLM-5.3-Flash
vllm serve "$MODEL_PATH" \
  --served-model-name glm53flash \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --attention-backend FLASHMLA_SPARSE \
  --block-size 128 \
  --max-model-len 16384 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.85 \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --skip-mm-profiling \
  --seed 0 \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --port 8000
```

The final TP8 experiment used the equivalent
`python -m vllm.entrypoints.openai.api_server` entry point, the mounted path
`/ipfs/models/GLM-5.3-Flash`, served name `glm53flash-no-fallback`, and port
`18083`; substitute these values as appropriate. Graph capture must remain
**enabled**. The MUSA indexer patch converts logical top-k indices to physical
indices on the compute stream; non-MUSA platforms keep their existing
side-stream behavior. Do not install an older `cudagraph_utils.py` overlay over
an image that already provides the MUSA graph-capture stream fix.

Do **not** set `VLLM_USE_DEEP_GEMM_E8M0=0` as a deployment prerequisite. The
v0.32.0 MUSA DeepGEMM capability guard automatically disables an unsupported
grouped-FP8 UE8M0 cast while retaining the legacy FP8 paged-MQA logits route.
The 8-rank graph capture and actual sparse-MLA model1 prefill must be checked
in the server log, not inferred from a successful startup alone.

## Minimal verification

1. Check `/health` and `/v1/models`; send a real chat request and require a
   semantically correct final answer and `finish_reason=stop`.
2. Send a multi-turn conversation, a streaming request, and two waves of eight
   simultaneous requests. Do not allow one request's content to leak to another.
3. Submit exactly 2,500 tokenizer IDs and check the **server's**
   `usage.prompt_tokens=2500`. A longer 16,000-token prompt should also fit
   within this profile, with an appropriately short output budget.
4. If vision is required, send an actual `image_url` rather than a textual
   description of an image. The tested image route returned a correct color
   description for a 32×32 red PNG even with `--skip-mm-profiling`.
5. Use the ticket's agreed accuracy and performance gates, including
   GSM8K-128 with zero invalid outputs and a comparable-hardware baseline.
   A startup log or an eager-only benchmark is not a substitute.

In the final single-node TP8 validation with the registry image
`sh-harbor.mthreads.com/mcctest/vllm:v0.32.0-ph1-5.2.0-torch2.11.0.post2-20261030`
(digest
`sha256:8f9f8fc900085368c3d0a710743e53f3f9d23ce16ef39425f627b59a23c1cdb1`),
the service reached `/health=200` and the regenerated `0209–0211` patch set was
applied on the pinned vLLM revision. Four identical GSM8K-128 runs produced
`121/128`, `122/128`, `121/128`, and `123/128` correct answers, with
`128/128` valid responses in every run. This confirms normal output variation
for this evaluation rather than a deterministic model-quality regression.

The same graph-enabled profile completed three serving-performance rounds with
`96/96` successful requests: output throughput averaged `197.46 tok/s`, total
throughput averaged `528.15 tok/s`, mean TTFT was `2431.90 ms`, mean TPOT was
`38.91 ms`, and mean ITL was `38.89 ms`. These figures are evidence for this
specific 20261030 registry image and TP8 hardware allocation; repeat the
benchmark on comparable hardware before making a general performance claim.

Return to the [cookbook index](../README.md).
