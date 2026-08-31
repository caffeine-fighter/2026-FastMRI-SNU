# FastMRI next-challenge award playbook

이 문서는 2026년 SNU FastMRI Challenge에서 얻은 실측 결과를 다음 대회의
초기 의사결정에 바로 사용할 수 있도록 정리한 실행 계획이다. 목표는 완주가
아니라 **수상권 진입을 우선하고, 그 위에서 우승 가능성을 남기는 것**이다.

올해 결과는 다음 대회의 데이터와 규칙에 대한 보장이 아니다. 따라서 아래의
모델 선택은 강한 사전 가설(prior)이며, 데이터 공개 직후 짧은 matched screen으로
재확인한다. 단, 명확한 반증이 나오기 전까지는 다른 대형 architecture를 탐색하느라
최종 학습 시작을 늦추지 않는다.

## 1. 2026년에서 확정된 교훈

### C8이 기본 incumbent다

- 최종적으로 C8이 C10보다 좋은 결과를 냈다.
- cascade 증가는 표현력을 늘리지만 최적화 난도, 학습시간, 추론시간을 함께 늘린다.
- GTX1080에서는 절약한 VRAM과 시간을 mask conditioning, specialist, 후처리 같은
  고수익 축에 배분하는 편이 유리하다.
- C10은 기본 모델이 아니라 C8을 이겨야 하는 challenger다.

### GTX1080은 마지막 배포 gate가 아니라 architecture 조건이다

- 최종 learned lineage는 VESSL GTX1080에서 허용된 initialization부터 end-to-end로
  학습한다.
- A6000/RTX3090에서만 가능한 architecture는 최종 후보가 아니다.
- 모든 후보는 최대 coil/H/W의 acc4·acc8에서 forward, backward, optimizer step,
  checkpoint reload, 공식 추론 경로를 GTX1080으로 통과해야 한다.
- PyTorch peak 통계뿐 아니라 실제 process VRAM과 다음 allocation 가능 여부를 본다.

### TTA는 기본적으로 기각한다

- TTA2→TTA4에서 품질은 올랐지만 GTX1080 시간이 거의 두 배가 되어 최종 score가
  하락했다.
- 다음 대회의 scoring formula가 달라지지 않는 한 submission TTA에 연구 예산을
  쓰지 않는다.
- 변환 민감도 분석에는 사용할 수 있지만, 최종 경로는 single-pass여야 한다.

### 학습 recipe가 모델 크기보다 중요했다

- legal mask family의 acceleration, native width, ACS width/start, periodic residue를
  학습에 포함하는 MRAugment가 핵심 축이었다.
- 이미 수렴한 checkpoint에 큰 누적 LR을 오래 적용하면 train loss가 내려가도
  four-cell quality는 하락했다.
- 3e-4 부근에서는 loss saturation과 all-zero gradient collapse가 발생했다.
- 수렴 후 fine-tuning은 1e-5 부근의 짧은 warmup과 즉시 decay를 우선 검증한다.
- ACC4 specialist는 유망했지만 ACC8 specialist는 generalist를 이기지 못한 경우가
  많았다. specialist는 acceleration마다 자동 채택하지 않는다.

## 2. 다음 대회의 기본 제출 가설

```text
C8 PromptMR+ single-pass generalist
  + train-only legal-mask MRAugment
  + small mask-conditioned prompt/adapter
  + only proven single-route acceleration specialist
  + VESSL-only checkpoint averaging when beneficial
  + optional very small single-pass residual refiner
```

기본적으로 비활성화할 항목:

- inference TTA
- 두 모델을 항상 실행하는 ensemble/MoE
- C10 이상 cascade의 장기 학습
- 외부 GPU learned state의 VESSL 반입
- leaderboard sample을 training loader에 넣는 경로

라우팅이 필요하면 입력 mask에서 acceleration과 mask family를 결정해 **한 번에
한 경로만** 실행한다. unknown/mismatched mask는 C8 generalist로 fail-safe한다.

## 3. 데이터 공개 직후 72시간

### 0–6시간: 규칙과 데이터 계약 봉인

1. 올해 규칙을 그대로 가정하지 말고 다음 항목을 운영진 문서에서 재확인한다.
   - 외부 GPU의 architecture/scalar search 허용 범위
   - 최종 VESSL scratch/end-to-end 요구
   - public leaderboard 데이터의 허용 용도
   - `prep_volume()`과 timed reconstruction 경계
   - ensemble/MoE/TTA와 시간 점수
2. train/val 파일 inventory, SHA-256, slice/coil/H/W 분포를 만든다.
3. 제공 mask에서 acceleration, width, ACS, residue family를 분석한다.
4. public과 private이 같은 generator라는 보장이 없다면 public 빈도를 학습
   sampling weight로 사용하지 않는다.

### 6–24시간: 실제 장비 기준선

1. C8 PromptMR+를 GTX1080, RTX3090, A6000에서 동일 recipe로 smoke한다.
2. GTX1080에서 다음 전체 sequence를 측정한다.
   - acc4/acc8 최대 shape
   - first/steady optimizer step
   - save/reload/next step
   - 공식 single-pass inference
   - process VRAM, ms/slice, finite output, shape parity
3. C8 baseline의 1 epoch 실측시간으로 전체 VESSL ETA를 계산한다.
4. GPU 중단 gate는 OOM, NaN/nonfinite, 데이터 손상, checkpoint 불능, 장시간
   무진전으로 제한한다. telemetry와 부가 검증 문제는 기록하고 학습을 계속한다.

### 24–72시간: 방법론 matched screen

외부 GPU에서 동일 seed/order/step budget으로 다음만 비교한다.

1. C8 plain vs legal-mask MRAugment
2. augmentation onset: immediate vs short delay
3. LR/schedule: 기존 baseline과 low-LR short-decay
4. mask-conditioned lightweight adapter
5. C10 matched challenger 한 번

평가는 반드시 다음 four-cell을 모두 기록한다.

- acc4 full
- acc4 bbox
- acc8 full
- acc8 bbox

72시간 안에 backbone을 C8로 동결하는 것이 기본이다. C10은 quality와 실제
GTX1080 final score가 모두 명확히 이길 때만 승격한다.

## 4. 연구와 VESSL 최종 학습의 병렬화

최종 VESSL 학습 시작을 모든 연구가 끝날 때까지 기다리지 않는다.

```text
External RTX3090/A6000
  ├─ mask/LR/adapter matched screens
  ├─ ACC4 specialist
  ├─ ACC8 generalist-vs-specialist adjudication
  └─ report-only checkpoint evaluation

VESSL GTX1080
  ├─ frozen C8 architecture from fresh initialization
  ├─ sparse immutable milestones
  ├─ late specialist only if external evidence is positive
  └─ final single-pass 8GB candidate
```

- 4주 대회라면 데이터 공개 후 3–5일 안, 늦어도 마감 20일 전에 VESSL C8
  generalist를 시작한다.
- 외부 GPU에서는 weight가 아니라 architecture와 scalar recipe만 결정한다.
- VESSL이 이미 지나간 학습 구간을 변경하는 결과는 다음 lineage의 연구로 남기고,
  실행 중인 최종 lineage를 갈아엎지 않는다.
- 후반 fine-tuning은 VESSL에서 학습된 checkpoint로 VESSL 안에서만 이어간다.

## 5. Fine-tuning과 specialist 정책

### 수렴 후 LR

- 모든 epoch checkpoint를 후보로 유지하되 rolling checkpoint는 2–3개만 남긴다.
- selection은 마지막 epoch가 아니라 완료된 전체 milestone의 four-cell quality로 한다.
- 1e-5 부근에서 1 epoch 개선 후 하락하면 즉시 early stop한다.
- fine-tuning schedule은 한 epoch warmup 뒤 바로 decay하는 짧은 horizon을 사용한다.
- train loss 감소만으로 continuation을 승인하지 않는다.

### Specialist

- ACC4: generalist보다 해당 full/bbox가 모두 좋아질 때 단일-route specialist를 허용한다.
- ACC8: 기본값은 generalist다. specialist가 matched evaluation에서 이긴 경우에만
  활성화한다.
- complete model 두 개를 동시에 실행해 평균내지 않는다.
- 가능하면 shared C8 trunk와 작은 acceleration/mask adapter를 사용한다.

### 비용 없는 또는 저비용 전술

우선순위는 다음과 같다.

1. VESSL-trained checkpoint averaging/soup
2. legal-mask coverage 개선
3. lightweight mask conditioning
4. single-route specialist
5. 매우 작은 single-pass refiner

여러 checkpoint를 실행하는 ensemble과 TTA는 이 목록에 포함하지 않는다.

## 6. 승격과 중단 기준

모든 비교는 동일 evaluator, 동일 데이터, 동일 장비, 동일 timing 범위에서 한다.

| 결과 | 결정 |
|---|---|
| quality +0.0003 이상, 네 cell의 큰 후퇴 없음 | 승격 |
| +0.0001–0.0003 | 독립 checkpoint/seed로 한 번 확인 |
| ±0.0001 | 기각 |
| 한 cell -0.0003 이하 | 단일-route로 격리하지 못하면 기각 |
| quality 상승, GTX1080 final score 하락 | 기각 |
| C10이 C8 대비 명확한 이득 없음 | C8 유지 |
| ACC8 specialist가 generalist 이하 | generalist 유지 |

공개 1위 점수는 목표가 아니라 하한이다. 숨겨진 SOTA와 private 60%를 고려해 공개
최고점보다 최소 +0.001의 내부 목표를 세운다. public mask에만 맞춘 미세한 이득보다
generator family 전체에서 재현되는 이득을 우선한다.

## 7. 저장공간과 자동화

- 매 step checkpoint 저장 금지
- rolling latest 2개 + 사전 등록 milestone만 보존
- 평가를 위해 checkpoint를 복제하지 않고 동일 immutable 파일을 read-only로 사용
- 평가 결과는 작은 JSON/CSV로 누적
- GPU가 비면 다음 우선순위의 bounded screen을 자동 시작
- verifier/reviewer 실패는 학습 중단과 분리

매일 다음 일곱 항목만 보고한다.

1. 최고 quality와 final score
2. four-cell 점수
3. GTX1080 ms/slice와 process VRAM
4. 현재 VESSL epoch/ETA
5. 지난 24시간에 새로 확인된 사실
6. 기각한 가설과 사용 GPU-hours
7. 다음 24시간의 단일 핵심 가설

## 8. 마감 운영

- D-3: architecture, router, refiner, TTA-off 정책 동결
- D-2: GTX1080 최대-shape admission과 재현성 dry run 완료
- D-1: package와 안전한 fallback 완성
- 제출 당일: 새 학습 금지, 후보 선택·최종 평가·업로드만 수행
- 마지막 1시간을 목표로 제출하더라도 검증된 archive는 최소 24시간 전에 만든다.

## 9. Day-0 성공 조건

다음 항목이 모두 충족되면 대회가 시작된 당일 준비가 완료된 것이다.

- [ ] 올해 규칙 변경점 기록
- [ ] dataset inventory와 mask-family report 생성
- [ ] GTX1080 C8 exact admission PASS
- [ ] C8 1-epoch ETA 측정
- [ ] RTX3090/A6000 matched-screen queue 실행
- [ ] four-cell report-only evaluation 작동
- [ ] VESSL scratch launch package 준비
- [ ] rolling retention과 resume recovery 작동
- [ ] TTA submission path 비활성화 확인

`experiment-plan.json`은 이 문서의 핵심 gate와 순서를 자동화 도구가 읽을 수 있는
형태로 고정한다.
