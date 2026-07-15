# FastMRI 우승 전략 요약

> 기준일: 2026-07-15 KST
> 이 문서는 현재 기본 브랜치의 기록과 `EXP054` 전략 로드맵을 한 장으로 정리한 실행 기준이다. 실험·공식 평가·Git 병합은 각각 별도 승인 범위로 취급한다.

## 목표

평가 환경의 8GB GPU 제약과 공식 평가 규칙을 지키면서 최종 점수를 올린다. 단순히 검증 점수가 높은 모델이 아니라, **품질·추론 시간·재현성·패키징 가능성**을 모두 만족하는 한 개의 최종 후보와 한 개의 fallback을 확보한다. 최종 서버의 CPU·RAM·GTX 1080·driver/runtime 계약과 합격 기준은 [`final_evaluation_server.md`](final_evaluation_server.md)를 단일 기준으로 사용한다.
로컬 **RTX 3090 24 GB**는 VESSL 8 GB 환경에서 오래 걸리거나 메모리 여유가 부족한 학습 screen·seed 검증·긴 recipe 실험을 빠르게 수행하는 주력 학습 환경으로 쓴다. 다만 최종 후보는 반드시 공식 8 GB 추론 계약과 시간을 별도로 통과해야 한다.

## 현재 기준선

| 역할 | 후보 | 근거 |
|---|---|---|
| 보호할 공식 리더 | `EXP035_epoch30` | one-shot quality `0.92055`, total `0.92146109375` |
| 확정 fallback | `EXP030` | 30회 공식 timing cohort 완료 |
| 이전 one-shot 리더 | `EXP033R_epoch32` | quality `0.91595`, total `0.91690125`; 더 빠른 보존 후보 |

- LOCAL 평가는 방향을 찾는 증거일 뿐, 공식 후보 체크포인트로 승격하지 않는다.
- LOCAL 승격 기준은 acc4와 acc8을 동등하게 평균내는 leaderboard-faithful quality다. 완료된 EXP035 gate는 EXP033R `0.9156824558941089`를 사용했으며, 새 matched recipe의 보호 기준은 EXP035 epoch 30 `0.9199788092310326`이다.
- 공식 평가는 자동 실행하지 않는다. 독립 검증과 별도 승인을 통과한 후보만 one-shot 평가한다.

## 핵심 판단

1. **현재 vanilla VarNet의 용량 탐색은 EXP035로 끝낸다.**
   `c8/ch12`가 장기 학습에서도 기준선을 이기면 이후 recipe 실험의 baseline으로 쓰고, 실패하면 채널·cascade를 더 키우는 탐색을 중단한다.
2. **학습 VRAM과 추론 VRAM을 분리해서 판단한다.**
   24 GB 이상이 필요한 학습도 `eval` + no-grad + batch 1 추론에서는 8 GB에 그대로 들어갈 수 있다. 큰 모델을 압축하기 전에 무작위 초기화 상태로 최대 입력 8 GB forward preflight를 먼저 수행하고, 통과한 가장 큰 구조는 RTX 3090에서 그대로 학습해 무압축 배포를 우선한다.
3. **압축보다 무손실 배포 최적화를 먼저 한다.**
   직접 추론이 실패할 때만 per-coil sensitivity, coil chunking, tensor lifetime 정리 같은 출력 동등 메모리 제어를 적용한다. 선택적 FP16은 parity를 통과한 뒤에만 허용하고, 지식 증류·구조적 pruning은 그 뒤의 손실 가능 fallback이다.
4. **VRAM당 품질을 먼저 개선한다.**
   8 GB에서 기존 stack은 여유가 거의 없으므로, vanilla 폭을 무작정 키우기보다 PromptMR+의 정보 흐름·memory-efficient sensitivity와 같은 효율적인 구조를 먼저 screen한다.
5. **한 번에 큰 변수 하나만 바꾼다.**
   architecture, optimizer/scheduler, loss, augmentation을 동시에 바꾸지 않는다. 그래야 이득과 회귀의 원인을 추적할 수 있다.
6. **공식 리더와 fallback을 보호한다.**
   실험 실패가 기존 제출 가능 상태를 훼손하면 안 된다. 체크포인트 hash, 명령, commit, 평가 범위를 항상 남긴다.

## 로컬 RTX 3090 24 GB 활용 원칙

1. **더 빠르게 학습하고 더 많이 검증한다.**
   RTX 3090에서 short screen, matched e5/e15 run, seed confirmation, optimizer/loss 비교를 수행해 VESSL GPU 시간을 강한 후보에만 쓴다.
2. **큰 모델은 탐색 대상으로만 허용한다.**
   24 GB에서 학습이 되는 모델도 8 GB 공식 추론에서 동작한다는 보장은 없다. VESSL 승격 전에는 최대 허용 입력으로 추론 VRAM, 시간, 출력 parity를 측정한다.
3. **gradient checkpointing의 역할을 구분한다.**
   gradient checkpointing은 backward 때 activation을 재계산하여 **학습 VRAM을 줄이는** 기법이다. 학습 후에 적용해 파라미터 수, checkpoint 크기, 추론 VRAM 또는 추론 시간을 줄이는 방법은 아니다. 따라서 큰 모델을 RTX 3090에서 학습한 뒤 마지막에 checkpointing만 켜서 8 GB 모델로 바꾸는 전략은 성립하지 않는다.
4. **최종 8 GB 적합성은 구조적으로 확보한다.**
   최종 모델은 checkpointing 유무와 독립적으로 8 GB 추론에 맞아야 한다. 필요하면 coil chunking, sensitivity-map 메모리 절감, 더 작은/효율적인 model family, batch=1 추론 경로를 사용하며, 모든 변경은 출력 parity와 공식 시간 측정을 통과해야 한다.

## 대형 LOCAL 모델에서 8 GB로 가는 배포 사다리

대형 모델을 먼저 완주한 뒤 압축 가능성을 묻지 않는다. 가중치 값과 무관하게 결정되는 architecture-level 추론 메모리를 학습 전에 측정하여, 아래 순서를 고정한다.

### A. 8 GB 계약을 먼저 등록

- 후보 config, commit, 입력 shape·coil 수, dtype과 [`final_evaluation_server.md`](final_evaluation_server.md)의 환경 fingerprint를 고정한다.
- 학습 전 무작위 가중치로 공식 `recon_eval.py` 호출 경로의 최대 허용 slice를 batch 1로 warm-up 후 반복 실행한다.
- `max_memory_allocated`와 `max_memory_reserved`, process peak RSS, system available RAM, OOM 여부, ms/slice를 모두 기록한다. 단 한 번의 forward 성공이 아니라 반복 실행과 GPU/host headroom을 확인한다.
- 현재 고정 harness는 `model.eval()`과 `torch.no_grad()`를 이미 사용한다. `torch.inference_mode()`는 더 낮은 overhead 가능성이 있지만 별도 opt-in으로 출력 parity와 호출 호환성을 증명한 뒤에만 사용한다.

### B. 통과하면 압축 없이 직접 학습·배포

- 8 GB forward contract를 통과한 후보 중 품질 잠재력이 가장 큰 구조를 RTX 3090에서 학습한다.
- activation checkpointing, gradient accumulation, clipping은 LOCAL 학습을 가능하게 하는 수단일 뿐 최종 모델 변환 단계가 아니다.
- 학습된 checkpoint로 같은 8 GB probe를 다시 통과하고, full/bbox × acc4/acc8과 공식 시간을 확인한다.

### C. 실패하면 무손실 메모리 제어부터 적용

1. sensitivity map per-coil 또는 coil chunking
2. cascade 순차 실행 중 불필요한 tensor lifetime·복사 제거
3. batch 1, eval/no-grad 계약과 출력 저장 경로 점검
4. 선택적 FP16/autocast: GTX 1080에는 Tensor Core가 없으므로 속도 향상을 가정하지 않는다. complex FFT 안정성, SSIM parity, peak VRAM과 실제 총점이 FP32 control을 이길 때만 채택한다.

각 변경은 FP32 기준 출력·지표 parity와 peak VRAM을 함께 비교한다. 단순히 `empty_cache()` 호출이나 checkpoint 파일 압축을 VRAM 개선으로 인정하지 않는다.

### D. 그래도 안 들어갈 때만 teacher–student 증류

- teacher는 현재 최선 8 GB baseline보다 equal-acc quality가 최소 `0.001` 높고 네 보호 지표가 모두 건강할 때만 만든다. 이 간격이 없으면 압축 중 보존할 이득 자체가 부족하다.
- student 구조와 8 GB contract를 증류 시작 전에 고정한다.
- 기본 supervised SSIM/L1에 teacher 최종 출력 imitation을 더하고, architecture가 허용하면 cascade별 복원 또는 attention/feature transfer를 별도 ablation으로 추가한다.
- student는 teacher 이득 보존율이 아니라 최종 절대 gate로 판정한다: 현재 리더 대비 `>= 0.0005`, 완전한 coverage, 8 GB 반복 안정성, 공식 시간 이득을 모두 요구한다.
- 사후 unstructured pruning과 INT8 양자화는 실제 GTX 1080 kernel·메모리 이득이 증명되지 않으면 사용하지 않는다. pruning을 연구할 경우에는 사후 pruning보다 initialization-time structured/sparse 후보를 별도 실험으로 취급한다.

즉 기본 전략은 **큰 teacher를 만든 뒤 줄이기**가 아니라 **8 GB에 직접 배포 가능한 가장 큰 모델을 먼저 찾아 LOCAL에서 크게 학습하기**다. 증류는 direct-deploy 가능한 구조가 품질 목표를 못 채우고, 별도의 큰 teacher가 충분한 oracle gap을 증명했을 때만 사용한다.

## 실행 순서

### 1. EXP035 종료 및 판정 — 완료

- tracked terminal이 exit code 0으로 끝났고 30개 retained checkpoint가 모두 exact coverage와 finite-output gate를 통과했다.
- epoch 30 LOCAL quality는 `0.9199788092310326`으로 EXP033R 기준보다 `+0.004296353336923686` 높다.
- 승인된 공식 one-shot은 full `0.9234`, bbox `0.9177`, quality `0.92055`, total `0.92146109375`를 기록했다.
- EXP035가 명확한 PASS이므로 c8/ch12/s8을 vanilla recipe baseline으로 보호하고, c9/c10/c12 unmodified scaling은 시작하지 않는다.

판정 기준:

| 결과 | 조치 |
|---|---|
| quality `<= 0.9156824558941089` | c8을 기각하고 vanilla depth scaling을 종료 |
| gain `0 ~ 0.0005` | seed 또는 더 긴 matched run으로 견고성 확인 |
| quality `>= 0.9161824558941088` | 강한 LOCAL 승격 신호; c8을 recipe 실험 baseline으로 사용 |

### 2. 메모리 제어를 opt-in으로 추가

다음 기능은 baseline 출력과 checkpoint 의미를 바꾸지 않는 독립 옵션으로 구현·검증한다. RTX 3090에서는 더 큰 학습 screen을 가능하게 하고, 공식 환경에서는 추론 계약을 별도로 증명한다.

- cascade-level activation checkpointing (학습 전용; 최종 추론 VRAM 절감 수단이 아님)
- sensitivity-map coil chunking / per-coil 실행
- gradient accumulation과 effective batch 기록
- gradient norm telemetry, 그리고 별도 실험으로 clipping
- 최대 입력 forward probe의 peak allocated/reserved와 반복 안정성 기록
- 공식 harness의 기존 no-grad 경로를 control로 둔 `inference_mode` opt-in parity probe

각 옵션은 고정 입력의 출력·loss·gradient parity, resume 호환성, 최대 허용 입력에서의 학습/추론 peak VRAM을 통과해야 한다. `recon_eval.py`는 바꾸지 않는다.

### 3. 학습 recipe를 통제된 사다리로 비교

1. 같은 seed·데이터·구조·epoch에서 Adam과 AdamW를 비교한다.
2. AdamW가 이기면 warmup + cosine 또는 사전 등록한 decay schedule 하나만 추가 비교한다.
3. accumulation과 clipping은 optimizer/schedule 승자가 정해진 뒤 따로 본다.
4. 강한 gain(`>= 0.0005`), 네 보호 지표 건강성, 완전한 coverage, 두 번째 seed 또는 matched long run을 만족해야 VESSL로 올린다.

### 4. winner-style masked SSIM + L1을 별도 검증

과거의 실패한 sparse score-aligned loss를 이름만 바꿔 재시도하지 않는다. training-only foreground mask 기반 SSIM + L1을 표준 loss control과 동일 조건에서 비교한다.

- full-image 손실보다 bbox만 좋아지는 후보는 승격하지 않는다.
- annotation은 추론에 절대 유입하지 않는다.
- empty mask, 복수 box, crop/window, finite gradient를 테스트한다.

### 5. 다음 모델 계열은 deployment-first feasibility race로 제한

고정된 논문 순위를 먼저 정하지 않는다. **PromptMR+**는 upside와 공식 fastMRI 구현이 강하지만 license/integration risk가 있고, **Feature/FI-VarNet**은 현재 stack과 가까워 구현 위험이 낮다. PromptMR+ license 확인을 가장 먼저 끝낸 뒤 두 계열의 CPU shape/schema와 동일한 GTX 1080 최대입력 probe를 싸게 비교하여, direct-deploy·속도·구현 비용 gate를 먼저 통과한 계열을 학습한다. SDUM의 progressive cascade expansion과 sampling-aware DC는 근거 있는 후속 아이디어이지만 새 전체 stack을 바로 이식하지 않고 작은 ablation으로만 다룬다.

1. 논문/참조 구현 commit과 라이선스를 고정한다.
2. CPU shape test와 checkpoint schema test를 만든다.
3. 학습 전에 최대 입력 8 GB forward-only VRAM·runtime probe를 수행한다.
4. direct-deploy가 되면 그 구조를 LOCAL에서 학습하고, 안 되면 C 단계의 무손실 메모리 제어까지 적용한다.
5. vanilla baseline과 architecture만 다른 two-seed 1-epoch LOCAL screen을 한다.
6. seed-robust 방향만 e5 → e15/e30 또는 VESSL로 승격한다.
7. direct-deploy가 끝내 불가능하지만 큰 모델의 oracle gap이 `>= 0.001`이면 그때만 고정된 8 GB student로 증류한다.

PromptMR+ 참조 구현은 non-commercial research license이므로 challenge 사용·파생 코드 배포 가능성을 확인하기 전에는 코드를 복사하거나 제출물에 포함하지 않는다.

정해진 screen 예산에서 확실한 신호가 없으면 rewrite를 중단하고 최선의 vanilla recipe로 복귀한다.

### 6. 조건부 아이디어: mask, MRAugment, MoE, averaging

- mask/ACS 분석은 k-space에서 얻은 합법적 feature만 사용한다.
- MRAugment는 base model이 수렴한 뒤에만 검토한다.
- MoE는 고정 expert의 volume-disjoint oracle upper bound가 충분할 때만 구현한다. filename, annotation, target, exact mask hash, leaderboard 결과로 routing하지 않는다.
- 같은 basin의 checkpoint averaging/EMA/SWA는 추론 비용이 없으므로 성공한 장기 run 뒤 우선 screen한다.
- two-forward ensemble은 품질 이득이 실제 시간 패널티를 확실히 넘을 때만 허용한다.

## 승격·중단·동결 규칙

### 승격

- equal-acc quality가 현재 리더를 넘고, full/bbox 및 acc4/acc8의 설명되지 않는 붕괴가 없어야 한다.
- coverage, checkpoint provenance, clean completion을 확인한다.
- LOCAL 이득은 두 번째 seed 또는 matched long run으로 확인한다.
- 그 뒤에도 공식 one-shot은 별도 승인 사항이다.

### 중단

- 메모리 공학 없이 `chans > 12` 또는 c9/c10/c12을 추가로 시도하지 않는다.
- c8이 실패하면 unmodified vanilla capacity track을 살리려 여러 변수를 동시에 바꾸지 않는다.
- broad random mask augmentation, 4-way MoE, 약한 output ensemble은 근거가 생길 때까지 하지 않는다.
- LOCAL checkpoint를 공식 후보로 취급하지 않는다.
- 8 GB architecture preflight 없이 24 GB LOCAL 장기 teacher 학습을 시작하지 않는다.
- direct-deploy와 무손실 메모리 제어를 건너뛰고 곧바로 사후 pruning·양자화·증류로 가지 않는다.

### 최종 동결

1. 최대 두 후보만 seed confirmation으로 보낸다.
2. 성공한 장기 run에서 no-cost averaging을 먼저 본다.
3. 최종 후보와 fallback의 model family, config, checkpoint hash, preprocessing, inference code, 환경을 동결한다.
4. 동결 뒤에만 승인된 30회 공식 timing cohort, fresh-clone package 검증, 업로드를 실행한다.

## 운영 안전선

- `recon_eval.py`, mounted `Data`를 수정하지 않는다.
- data, H5, checkpoint, result directory, `.env`, credential을 Git에 넣지 않는다.
- VESSL training, official evaluation, Git 병합/정리는 각각 명시적으로 승인받는다.
- 활성 GPU 작업의 terminal evidence를 확인하고, branch 이동이나 PID만으로 완료를 판단하지 않는다.

## 한 문장 전략

`EXP035` epoch 30을 새 one-shot 공식 리더로 보호하고 vanilla capacity scaling을 끝내며, 다음 recipe 또는 upstream model-family 후보는 최종 GTX 1080 최대입력 계약과 thin-adapter 원칙을 먼저 통과시킨 뒤 재현 가능한 최종 후보만 별도 승인된 timing freeze와 제출로 올린다.

## 추가 근거

- PromptMR+는 gradient/information flow와 memory-efficient sensitivity estimation을 함께 개선하고 fastMRI knee/brain 결과와 공식 코드를 제공한다: <https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09565.pdf>, <https://github.com/hellopipu/PromptMR-plus>
- KD-MRI는 attention transfer와 output imitation으로 작은 MRI reconstruction student가 독립 학습 student보다 개선될 수 있음을 보였지만, 이는 증류가 direct deployment보다 우선이라는 뜻은 아니다: <https://arxiv.org/abs/2004.05319>
- PUN은 MoDL에서 사후 pruning보다 initialization-time pruning이 나았다고 보고하므로, pruning을 쓸 경우에도 마지막 순간의 비구조적 압축을 기본값으로 두지 않는다: <https://arxiv.org/abs/2412.18668>
- SDUM은 progressive cascade expansion과 sampling-aware weighted DC의 가능성을 보고하지만 2025 preprint이므로 현재 codebase에서는 제한된 ablation 근거로만 사용한다: <https://arxiv.org/abs/2512.17137>
- PyTorch는 `no_grad`가 gradient 기록을 꺼 메모리를 줄이고, `inference_mode`가 view tracking과 version counter overhead도 제거한다고 문서화한다: <https://docs.pytorch.org/docs/stable/generated/torch.no_grad>, <https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html>
