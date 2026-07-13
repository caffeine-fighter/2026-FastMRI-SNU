# FastMRI 우승 전략 요약

> 기준일: 2026-07-13 KST  
> 이 문서는 현재 기본 브랜치의 기록과 `EXP054` 전략 로드맵을 한 장으로 정리한 실행 기준이다. 실험·공식 평가·Git 병합은 각각 별도 승인 범위로 취급한다.

## 목표

평가 환경의 8GB GPU 제약과 공식 평가 규칙을 지키면서 최종 점수를 올린다. 단순히 검증 점수가 높은 모델이 아니라, **품질·추론 시간·재현성·패키징 가능성**을 모두 만족하는 한 개의 최종 후보와 한 개의 fallback을 확보한다.
로컬 **RTX 3090 24 GB**는 VESSL 8 GB 환경에서 오래 걸리거나 메모리 여유가 부족한 학습 screen·seed 검증·긴 recipe 실험을 빠르게 수행하는 주력 학습 환경으로 쓴다. 다만 최종 후보는 반드시 공식 8 GB 추론 계약과 시간을 별도로 통과해야 한다.

## 현재 기준선

| 역할 | 후보 | 근거 |
|---|---|---|
| 보호할 공식 리더 | `EXP033R_epoch32` | one-shot quality `0.91595`, total `0.91690125` |
| 확정 fallback | `EXP030` | 30회 공식 timing cohort 완료 |
| 진행 중인 마지막 vanilla capacity 검증 | `EXP035_varnet_c8_ch12_s8_e30` | c6/ch12 대비 c8/ch12의 장기 효과를 판단 |

- LOCAL 평가는 방향을 찾는 증거일 뿐, 공식 후보 체크포인트로 승격하지 않는다.
- LOCAL 승격 기준은 acc4와 acc8을 동등하게 평균내는 leaderboard-faithful quality다. 현재 `EXP033R`의 비교 기준은 `0.9156824558941089`다.
- 공식 평가는 자동 실행하지 않는다. 독립 검증과 별도 승인을 통과한 후보만 one-shot 평가한다.

## 핵심 판단

1. **현재 vanilla VarNet의 용량 탐색은 EXP035로 끝낸다.**
   `c8/ch12`가 장기 학습에서도 기준선을 이기면 이후 recipe 실험의 baseline으로 쓰고, 실패하면 채널·cascade를 더 키우는 탐색을 중단한다.
2. **VRAM당 품질을 먼저 개선한다.**
   8 GB에서 unmodified stack은 여유가 거의 없으므로, 무작정 폭을 키우기보다 memory engineering과 더 효율적인 모델 계열을 우선한다.
3. **한 번에 큰 변수 하나만 바꾼다.**
   architecture, optimizer/scheduler, loss, augmentation을 동시에 바꾸지 않는다. 그래야 이득과 회귀의 원인을 추적할 수 있다.
4. **공식 리더와 fallback을 보호한다.**
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

## 실행 순서

### 1. EXP035 종료 및 판정

- 학습이 끝날 때까지 경쟁 VESSL GPU 작업을 추가하지 않는다.
- retained checkpoint마다 30 volumes, 791 slices, 161 boxes, `skipped=[]`를 독립 확인한다.
- full/bbox × acc4/acc8 지표와 equal-acc quality를 원본 산출물에서 다시 계산한다.
- 종료 뒤에만 실제 추론 VRAM과 공식 경로 시간을 측정한다.

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

### 5. 다음 모델 계열은 feasibility race로 제한

우선순위는 **Feature/FI-VarNet**, 그 다음 **reduced PromptMR 계열**이다.

1. 논문/참조 구현 commit과 라이선스를 고정한다.
2. CPU shape test와 checkpoint schema test를 만든다.
3. 최대 입력 forward-only VRAM·runtime probe를 통과한 구성만 남긴다.
4. vanilla baseline과 architecture만 다른 two-seed 1-epoch LOCAL screen을 한다.
5. seed-robust 방향만 e5 → e15/e30 또는 VESSL로 승격한다.

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

`EXP035`로 현재 vanilla capacity 가설을 끝까지 판정하고, 그 사이 8 GB-safe 메모리 제어와 Feature/FI-VarNet 대 reduced PromptMR feasibility를 준비한 뒤, AdamW/schedule과 masked SSIM + L1을 하나씩 검증하여 재현 가능한 최종 후보만 공식 평가와 제출로 올린다.
