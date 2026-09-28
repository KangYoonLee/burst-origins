# Context: 내 개발 환경 셋업 현황

## 나에 대해
- Johns Hopkins University에서 Soudeh Ghorbani 교수님 아래 PhD 시작
- 연구 목표: AI datacenter network에서 Traffic Burst의 origin을 밝혀내는 것
  (network stack 전반에서 burst 발생 원인을 찾고 domain knowledge를 구축)
- 영어에 익숙해지는 중 — DCN/networking 전문 용어는 반드시 영어로,
  설명은 한국어에 영어 단어를 섞은 교포 말투로 (전문 용어는 옆에 뜻 병기)
- 터미널/CLI에 익숙하지 않음 — 가능하면 GUI(클릭) 방식으로 안내
- 코드 수정 요청 시: 전체 파일 재작성 말고 diff 스타일 최소 snippet + 위치 맥락

## 하드웨어 / OS
- Windows 데스크탑 (호스트명 Uni-verse)
- Intel i7-9700, RAM 32GB, SSD(SATA)
- 보조: M1 MacBook (RAM 8GB)

## 개발 환경 구조
VS Code 하나를 editor로 쓰고, Remote Development extension으로
backend를 전환하는 방식:

  [VS Code (Windows)]
       ├─ Connect to WSL  → 로컬 Ubuntu (코드 작성, compile, logic test)
       └─ Connect to Host → CloudLab (실제 측정)
              ↕ GitHub가 둘 사이의 다리

역할 분리 원칙:
- WSL = correctness용 (컴파일 통과, logic 검증)
- CloudLab = measurement용 (bare-metal, 실제 숫자)
- Windows = SSH key 관리만 (Remote-SSH가 Windows의 ssh.exe를 쓰므로)

## WSL2 환경 (설치 완료)
- Ubuntu 22.04 LTS, WSL2 (Hyper-V VM)
- 사용자: klee, 작업 경로 /home/klee/research/
- BIOS에서 Intel VT-x 켜야 설치 가능했음 (HCS_E_HYPERV_NOT_INSTALLED 에러)
- gcc 11.4.0 / Python 3.10.12 / git 2.34.1
- 설치 패키지: build-essential, gdb, cmake, git, python3-venv
- venv: ~/research/.venv (numpy 2.2.6, pandas 2.3.3, matplotlib 3.10.9)
  → requirements.txt로 고정해둠

## VS Code
- Extensions (WSL 쪽에 설치): Python, C/C++, C/C++ Extension Pack
- Remote Development extension 설치됨
- pre-release 버전은 쓰지 않음 (stable만)
- C 실행 설정: "C/C++: gcc build and debug active file" (/usr/bin/gcc)
  — 버전 고정(gcc-11) 대신 일반 경로를 써서 CloudLab과 config 공유 가능하게

## GitHub
- SSH key로 인증 (~/.ssh/id_ed25519, comment "klee-wsl")
- Repo: git@github.com:KangYoonLee/burst-origins.git (Private)
- 올라간 파일: counter_probe.c, coll_bench.py, run_experiment.py,
  run.sh, README.md
- .gitignore: __pycache__/, *.pyc, .venv/, *.o, counter_probe,
  *.csv, results/, data/

## WSL2 measurement baseline (2026-09-28 측정)
CloudLab bare-metal과 비교하기 위한 기준값:
- clocksource: hyperv_clocksource_tsc_page (hypervisor 중계)
- clock_getres 보고값: 1 ns
- 실측 tick granularity: 108 ns
- 두 번 연속 clock_gettime 최소 gap: 0 ns
  (호출이 tick보다 빨라서 같은 값을 두 번 읽음)
→ WSL의 timestamp는 hypervisor를 거치므로 CloudLab bare-metal(tsc)보다
  거칠 것으로 예상. 아직 CloudLab에서 같은 측정 안 했음.

## 겪었던 이슈와 해결
1. wsl --install 실패 (HCS_E_HYPERV_NOT_INSTALLED)
   → 작업관리자 성능탭에서 "가상화: 사용 안 함" 확인
   → BIOS에서 Intel VT-x 활성화로 해결
2. Windows 탐색기로 \\wsl$\ 경로에 폴더 생성 시 소유자가 root가 됨
   → Permission denied. sudo chown -R klee:klee 로 해결
   → 교훈: 폴더는 WSL에서 mkdir, 파일만 탐색기로 드래그
3. Windows에서 복사한 파일에 :Zone.Identifier 파일이 딸려옴
   → NTFS Alternative Data Stream. rm -f *:Zone.Identifier

## 다음에 할 것
- CloudLab Remote-SSH 연결 설정 (Windows에서 ssh-keygen →
  CloudLab Manage SSH Keys 등록 → ~/.ssh/config 작성)
- CloudLab node에서 GitHub SSH key 설정 후 clone
- CloudLab에서 clock resolution 측정해서 WSL과 비교
- 실험 결과 폴더마다 commit hash / requirements.txt / env 정보 같이 남기기

## 내가 지키기로 한 원칙
- 측정 전에 도구의 resolution / overhead / granularity를 먼저 잰다
- "측정할 수 없다"와 "존재하지 않는다"는 다른 명제
- 실험 노트에 signal path(신호가 거친 layer)와 observation path(내가 본 도구)를 구분해 기록
- 커밋 메시지는 "무엇을 왜"로 작성