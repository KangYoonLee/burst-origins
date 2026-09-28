# CPU Gloo AllToAll: first controlled experiment

## 이번 수정

- `--counter` 생략 시 `--ifname`에서 counter 경로 자동 생성. 직접 지정한 경로는 그대로 유지.
- probe 요약의 `total_delta_bytes`를 `total_delta_raw`로 변경. 단위 변환을 수행하지 않는다.
  기본 C2(`tx_bytes`)에서는 raw 1이 1 byte이다. 다른 counter는 단위와 의미를 별도로 검증해야 한다.
- CPU 병목 판정, timeout/buffer 관계, ethtool/probe 측정 창 설명을 명확히 했다.
- 기존 probe source 및 기본 CPU/통신 조건은 변경하지 않았다.

## 목적

동일한 CPU tensor 교환을 probe off/on에서 비교한다. `all_to_all_single`을 사용한다.
MoE training 전체나 자연 발생 burst의 원인을 재현하는 실험은 아니다.
관측 그래프의 큰 반복 주기는 여기서 명시적으로 넣은 barrier/guard/gap의 영향을 받는다.

## 파일과 역할

- `coll_bench.py`: node마다 하나의 rank. CPU tensor 준비, Gloo 통신, 데이터 확인, 시각 기록.
- `run_experiment.py`: node0에서 SSH로 worker 배포/실행, warmup 대기, probe 실행/종료, 결과 수집.
- `run.sh`: Python controller를 실행하는 짧은 launcher.
- 기존 `~/burst/counter_probe`: 사용자가 검증한 실행 파일을 그대로 사용. 이 archive에 포함하지 않음.

## 기본 조건

| 항목 | 기본값 |
|---|---|
| Nodes | node0, node1 |
| Rendezvous | 10.10.1.1:29531 |
| Interface | enp65s0f0np0, 양쪽 node 동일 이름이라는 전제 |
| Backend / operation | Gloo / all_to_all_single, blocking |
| Device / dtype | CPU / float32 |
| Benchmark CPU | logical CPU 10, worker가 생성하는 thread도 해당 affinity를 상속 |
| Probe CPU | logical CPU 6, node0에서만 실행 |
| Intra/inter-op threads | 각각 1 |
| Peer당 tensor | 32 MiB, 1 MiB = 1,048,576 bytes |
| Warmup | 3회, probe 시작 전 |
| 측정 | 20회 |
| 반복 | barrier → guard 20 ms → collective → gap 200 ms |

`env.sh`를 자동으로 읽지 않는다. 위 기본값은 이전에 확인한 환경을 바탕으로 정했다.
변경은 CLI 인자로 지정한다. `--counter`를 생략하면 `/sys/class/net/<ifname>/statistics/tx_bytes`를
사용한다. 직접 지정한 `--counter`는 interface 변경으로 덮어쓰지 않는다.
이 baseline은 C2 byte counter로 검증한 것이며, 다른 경로를 지정한다고 해당 counter의
집계 범위와 단위까지 검증되는 것은 아니다. 예: `bash run.sh on --bench-cpu 10 --probe-cpu 6`.
PyTorch는 모든 node에 이미 설치되어 있어야 한다. node0에서 각 host로 passwordless SSH,
scp, remote timeout, ethtool을 사용할 수 있어야 한다. controller 전체를 sudo로 실행하지 않는다.
기존 probe가 읽는 sysfs 파일은 사용자 권한으로 읽을 수 있어야 한다.

## 전송량을 어떻게 정의했나?

2 ranks에서 각 input은 64 MiB이며, destination별 32 MiB block 두 개다.
자기 자신으로 향하는 block 32 MiB는 remote payload에서 제외한다.
따라서 측정 20회의 rank당 remote payload = 32 MiB × 20 = 640 MiB
= 671,088,640 bytes. 양쪽 rank가 동시에 같은 양을 보내고 받는다.

N ranks일 때 rank당 input = N × peer_bytes,
rank당 remote payload/iteration = (N-1) × peer_bytes.
Warmup 3회는 별도이며 metadata에 별도로 기록된다.
Counter에는 barrier, rendezvous 이후의 control, TCP header/retransmission,
background traffic 등이 포함될 수 있다. 640 MiB와 전체 counter 증가량이 정확히 같아야
한다고 판정하지 않는다. RX나 wire 전송 시점을 직접 측정하는 코드도 아니다.

## 실제 실행 과정

1. node0 controller가 환경 정보를 저장한다.
2. worker source를 node1의 고유 /tmp directory에 복사한다.
3. rank0은 node0에서, rank1은 node1에서 실행한다.
4. 각 worker가 tensor를 한 번 준비하고 warmup 3회 및 전체 수신값 검증을 한다.
5. 각 worker가 READY를 만들고 GO를 기다린다.
6. controller가 ethtool before를 저장하고, on이면 node0 probe를 실행한다.
7. off/on 모두 동일하게 3초 기다린 뒤 GO를 전달한다.
8. barrier/collective 시작·끝을 memory에 기록하며 20회 통신한다.
9. 전체 수신값을 다시 검사하고 timing.csv와 metadata.json을 저장한다.
10. controller는 2초 tail 여유 후 probe에 SIGINT를 보내고 CSV 저장 완료를 기다린다.
11. ethtool after와 remote rank 결과를 수집하고 summary.json을 만든다.
12. on에서는 probe CSV를 순서대로 읽어 측정 전체 coverage/실패/감소/buffer_full을 검사한다.

기존 probe에는 READY가 없으므로 3초 sleep을 정확한 준비 완료 증거로 취급하지 않는다.
결과에서 probe의 첫 valid read가 첫 barrier보다 앞이고 마지막 valid read가 마지막
collective 완료보다 뒤인지 확인한다. 이 검사에 실패하면 run은 실패로 표시한다.
probe가 5천만 sample buffer에 먼저 도달하는 경우도 실패다.

`--run-timeout=240`은 정상 run을 240초 동안 polling하라는 뜻이 아니다. 정상적으로는
worker 완료와 tail 대기 후 SIGINT로 중단한다. 관측한 초당 약 64만 sample 속도라면
5천만 sample buffer는 대략 78초 분량이며, 고정된 60초 제한은 적용하지 않았다.
긴 실험으로 바꿀 때에는 실험 길이, 실제 sample 속도, buffer와 memory 예산을 함께
검토해야 한다. 시간 제한이나 buffer 종료로 coverage가 부족해도 성공 처리하지 않는다.

ethtool before는 probe 시작 전이고 after는 probe 종료 및 CSV 저장 후이다.
3초 lead와 2초 tail은 probe에도 포함되도록 의도한 구간이며 ethtool만의 추가 구간이 아니다.
두 측정 창은 정확히 같지 않지만, 차이나는 구간에 counter 증가가 없으면 총량은
정확히 같을 수 있다. 항상 같아야 한다거나 절대로 같을 수 없다고 가정하지 않는다.

## 왜 barrier와 gap을 넣었나?

첫 실험에서는 반복된 collective를 그래프에서 구분하기 위해 넣었다.
barrier는 rank의 진행을 맞출 뿐 clock을 동기화하지 않는다. barrier traffic도 존재한다.
barrier start/end를 별도 기록하고 20 ms guard를 둔다. guard가 모든 관련 traffic을
완전히 분리한다고 보장하지는 않는다. time.sleep은 실제 wakeup 시각에 오차가 있어도
실제 collective start/end는 RAW clock으로 다시 기록한다.

이 설정에서 보이는 200 ms 주기를 Gloo 고유 burst 주기라고 주장해서는 안 된다.
후속 실험에서 gap/guard를 바꿀 수 있지만 첫 off/on 비교에서는 고정한다.

## 설치 및 첫 실행 (node0)

압축 파일을 node0의 ~/burst/gloo-main-kit.zip으로 전송한 뒤:

```bash
cd ~/burst
unzip gloo-main-kit.zip
cd gloo-main-kit
bash run.sh smoke
```

node1에 수동 실행 명령을 넣을 필요가 없다. node2/3은 사용하지 않는다.
다른 iperf client나 benchmark는 동시에 실행하지 않는다. 대기 중인 iperf server는
collective를 받는 프로그램이 아니며 Gloo worker는 이 launcher가 별도로 시작한다.

smoke에서는 peer당 1 MiB, warmup 3회, 측정 2회, probe off로 기능을 확인한다.
`summary.json`의 success=true, 두 rank의 metadata success=true가 통과 기준이다.
해당 torch build에서 all_to_all_single이 지원되지 않으면 rank log에 오류를 남기고
멈춘다. 다른 collective나 P2P 구현으로 조용히 대체하지 않는다.

## Main off/on (smoke 통과 후)

```bash
bash run.sh off
bash run.sh on
```

이는 첫 한 쌍이다. 한 쌍으로 observer effect를 확정하지 않는다.
같은 설정의 순서 균형을 잡은 반복 예:

```bash
bash run.sh off
bash run.sh on
bash run.sh on
bash run.sh off
bash run.sh off
bash run.sh on
```

처음 한 쌍을 한 뒤 위 여섯 run을 또 해야 한다는 뜻은 아니다. 필요하면 첫 한 쌍에
이어 나머지 네 run을 수행한다. 각 run은 고유 결과 directory를 생성한다.

## 결과 파일

`~/results/YYYYMMDD_HHMMSS_MODE_ID/`

- config.json, preflight.txt: 실행 조건과 환경
- rank0.log, rank1.log: 오류/상태 출력
- rank0/timing.csv, rank1/timing.csv: iteration별 local RAW timestamp (절대 ns)
- rank*/metadata.json: version, tensor 크기, 예상 payload, validation 결과, clock anchor
- ethtool_before.txt, ethtool_after.txt: 측정 창을 감싼 port statistics
- on일 때 counter.csv, probe.log
- summary.json: 성공 여부, rank별 duration 요약, on의 probe coverage와 raw 총 증가량 (`total_delta_raw`)

`duration_ns = end_raw_ns - start_raw_ns`는 해당 rank의 blocking collective 호출
경과 시간이다. 순수 wire latency나 전체 job의 synchronized duration은 아니다.
다른 node의 RAW clock끼리 빼거나 그대로 시간 정렬하지 않는다.
node0 timing과 node0 probe는 같은 CLOCK_MONOTONIC_RAW를 사용하므로 다음으로 연결한다:

```
collective_start_relative_ns = start_raw_ns - probe_anchor_start.mono_raw_ns
```

REALTIME은 참고 anchor이며 정밀한 cross-node synchronization 수단이 아니다.
Counter CSV는 총량/coverage까지만 자동 검사한다. 아직 bin 크기나 신뢰구간을 가정해
traffic 그래프를 그리지 않는다. 실제 run 데이터를 본 뒤 별도로 분석한다.

## 한계와 현재 검증 범위

생성 환경에는 PyTorch가 없어 실제 Gloo 통신을 실행하지 못했다.
Python compile, shell syntax, CLI help, probe coverage 검사(정상/늦은 시작/조기 종료/
buffer_full/invalid sample)를 테스트했다. CloudLab의 torch 2.14.0+cpu와 실제 NIC에서의
통신 성공은 smoke로 확인해야 한다. 공식 문서의 list all_to_all 지원 표와 tensor
all_to_all_single을 혼동하지 않도록 tensor API를 명시적으로 사용했다.

Main의 probe on/off 성능 비교는 필수다. 이전 iperf 결과에 observer effect 의심이
있으므로 측정한 pattern을 probe 없는 자연 상태와 같다고 단정하지 않는다.
기본 CPU 10 하나에서 Gloo worker/thread가 실행된다. 이는 재현 가능한 첫 baseline을
위한 설정이지 최대 성능을 보장하는 설정이 아니다.
PyTorch intra/inter-op thread 설정이 1이라고 Gloo 내부 thread가 총 하나라는 뜻은 아니다.
Benchmark의 affinity 설정은 NIC IRQ affinity를 변경하지 않는다.
32 MiB를 25 Gbps로 보내는 payload-only 계산은 약 10.74 ms지만, 실제 link speed를
확인해야 하며 collective duration에는 software 처리/대기 등이 포함된다.
11–15 ms이면 CPU 병목이 아니거나 30 ms이면 CPU 병목이라고 판정하지 않는다.
CPU 병목 가설은 이후 다른 조건을 고정하고 허용 CPU 수 등을 바꾸어 검증해야 한다.

## 실패/중단

오류 시 전체 launcher는 nonzero로 종료하며 summary.json에 error를 남긴다.
노드별 log를 먼저 읽는다. local process는 해당 run이 시작한 PID에만 signal을 보낸다.
SSH 연결이 끊겨 remote worker가 남더라도 remote timeout이 실행 시간을 제한한다.
remote /tmp 경로는 성공 후에도 남긴다(증거 보존). 결과는 node0로 복사된다.
SIGKILL/서버 장애/강제 예약 종료는 저장 완료를 보장하지 않는다.

## 참고

PyTorch distributed API와 interface 선택:
https://docs.pytorch.org/docs/stable/distributed
