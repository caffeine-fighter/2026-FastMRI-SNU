# VESSL PID-domain attribution reprovision requirements

Status: `BLOCKED_NEEDS_VESSL_REPROVISION`

This is a minimum provider contract, not a claim that the fields below already exist in the public VESSL YAML or CLI schema. The current public workspace/run schemas expose GPU product/type, region or node candidate, and GPU count/resource preset, but no documented host/container PID bridge, host-PID mode, or trusted process telemetry identity API. Invented YAML fields are therefore prohibited.

## Unchanged workload requirements

- GPU product: NVIDIA GeForce GTX 1080
- GPU count: 1
- GPU UUID must be exposed and remain stable for the workload
- Inference-only scope
- No PromptMR+ training, backward, optimizer step, reconstruction, or official evaluation
- No privileged container
- No user-controlled hostPath mount
- No writable host procfs
- No host runtime socket or cluster credential access

## Required provider change

VESSL must supply exactly one documented, non-privileged authoritative PID bridge for the replacement workload.

### Acceptable option A: provider-managed read-only host procfs

Mount host procfs at the fixed path `/run/vessl-host-proc` using the VESSL control plane, not a user-supplied Kubernetes `hostPath` or privileged container.

Required mount properties:

- read-only
- `nosuid`
- `nodev`
- `noexec`
- unavailable for writes or remounts from the workload
- stable for the complete canary process lifetime

The following paths must be readable for only the NVIDIA-reported candidate PID:

- `/run/vessl-host-proc/<driver_pid>/status`
- `/run/vessl-host-proc/<driver_pid>/stat`
- `/run/vessl-host-proc/<driver_pid>/ns/pid`

The mapping is accepted only when:

1. `driver_pid > 0`.
2. Host-view `NSpid[0] == driver_pid`.
3. Host-view `NSpid[-1] == application_pid`.
4. The host-view nested PID namespace inode equals `/proc/self/ns/pid`.
5. Process start identity from host and local proc views matches.
6. The mapping is one-to-one and stable across two samples.
7. The process remains present through the sample and disappears after canary termination.

### Acceptable option B: trusted VESSL telemetry identity API

Expose a provider-signed or control-plane-authenticated read-only record for the current workload containing:

- VESSL workload ID
- VESSL workspace ID
- observation timestamp
- application/container PID
- complete `NSpid` vector
- NVIDIA driver/NVML PID
- nested PID namespace inode
- process start identity
- GPU index
- GPU UUID
- finite nonnegative process GPU memory
- allocation/lease identity proving the record belongs to this workload

The record must be retrievable without exposing cluster credentials or granting runtime-socket access. It must reject duplicate, ambiguous, stale, missing, or many-to-one mappings.

### Acceptable option C: trusted provider helper

Run a VESSL-managed helper in the NVIDIA driver PID domain. The helper must return the same typed fields and one-to-one guarantees as option B over a read-only, workload-bound channel. The user container must not receive host privileges, host runtime sockets, or arbitrary host filesystem access.

## GPU exclusivity requirement

The provider must attest that GPU UUID `GPU-3073d3e5-383c-775f-faca-904c38057c94` is dedicated to the replacement workload for the bounded canary interval, or provide an equivalent lease identity. Polling zero rows before and after cannot prove the absence of transient unrelated processes.

Canary preflight still requires:

- rows → GPU state → rows
- both process-row snapshots empty
- utilization exactly 0%
- baseline memory within the reviewed threshold
- exact GPU index and UUID match

During the canary, exactly one compute row may exist, and it must map authoritatively to the canary. Any additional row is `FAIL_GPU_BUSY`; a missing or unmappable own row is `FAIL_MEMORY_EVIDENCE`.

## VESSL configuration/command status

No exact public VESSL YAML key or CLI flag for any acceptable PID bridge was found in:

- <https://docs.vessl.ai/reference/yaml/run-yaml>
- <https://docs.cloud.vessl.ai/member/workspace/create>
- installed `vessl 0.1.199` workspace/run help

Therefore no launchable YAML or CLI command is provided. Adding an undocumented `hostPID`, `privileged`, or `hostPath: /proc` field would be an unsupported security bypass and is forbidden.

The required actionable provider request is:

> Reprovision VESSL workload `137439389349` / workspace `85899410676` as a one-GTX-1080 inference-only workload with a documented non-privileged PID identity bridge satisfying option A, B, or C, plus workload-bound exclusive GPU attestation. Do not start the replacement workload until the exact provider configuration is returned for review and separately approved.

## Evidence and launch gate

Current diagnosis:

`reports/promptmr_plus/PID_DOMAIN_DIAGNOSIS_20260717T050532Z.json`

Existing blocker manifest, preserved without regeneration:

`reports/promptmr_plus/PROMPTMR_PLUS_INFERENCE_PREFLIGHT_8GB_V2_BLOCKED_20260717.json`

Until the provider returns a documented configuration and a fresh independent exact-byte review returns PASS:

- telemetry canary: forbidden
- CUDA initialization: forbidden
- H5 preflight: forbidden
- model import: forbidden
- PromptMR+ training on GTX 1080: forbidden
- EXP036 registry/output creation: forbidden
