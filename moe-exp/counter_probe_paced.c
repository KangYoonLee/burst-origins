// =============================================================================
// counter_probe_paced.c — 100us target paced counter sampling, no per-sample disk I/O
//
// 사용법:  counter_probe_paced <counter_path> <cpu> <max_sec> <out.csv> [interval_us] [max_samples]
//   예)    counter_probe_paced /sys/class/net/enp65s0f0np0/statistics/tx_bytes 6 120 A_tx.csv 100
//
// 종료:    max_sec가 지나거나, SIGINT(Ctrl+C)/SIGTERM을 받으면 기록을 CSV로 저장하고 끝남
// 원칙:    probe는 raw 값만 정직하게 남긴다. 단위 변환(x4 등)과 rate 계산은 분석 단계에서.
// Build:   gcc -O2 -Wall -Wextra -o counter_probe counter_probe.c -lm
// =============================================================================
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <sched.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

// sample 하나. ta는 시작 기준 상대시간(ns), dt = tb - ta (ns). 크기는 sizeof(sample_t)로 계산
typedef struct {
    uint64_t ta;    // 읽기 직전 시각
    uint64_t val;   // 읽은 counter 값 (raw, 단위 변환 안 함)
    uint64_t dt;    // 읽는 데 걸린 시간 (tb - ta). 긴 지연도 자르지 않고 그대로 저장
    uint32_t ok;    // 1 = 읽기 + 숫자 변환 성공, 0 = 실패
} sample_t;

#ifdef __linux__
#define CPU_SETSIZE_SAFE CPU_SETSIZE
#else
#define CPU_SETSIZE_SAFE 1024
#endif

// signal handler는 "멈춰" 표시만 한다. (handler 안에서 파일 쓰기 같은 복잡한 일은 위험)
static volatile sig_atomic_t stop_flag = 0;
static void on_signal(int sig) { (void)sig; stop_flag = 1; }

// counter 문자열 → 숫자. 성공하면 1, 실패하면 0.
// 허용: 숫자만 + (선택) 끝의 newline 하나. 거부: 빈 값, 음수, 공백/기호 시작, overflow, 숫자 뒤 이상한 문자
static int parse_counter(const char *buf, uint64_t *out) {
    if (buf[0] < '0' || buf[0] > '9') return 0;       // "-5", " 5", "abc" 등 거부
    errno = 0;
    char *end;
    unsigned long long v = strtoull(buf, &end, 10);
    if (errno == ERANGE) return 0;                      // 64-bit를 넘는 값
    if (*end == '\n') end++;                            // 정상적인 끝 newline 허용
    if (*end != '\0') return 0;                         // "123abc" 같은 경우 거부
    *out = (uint64_t)v;
    return 1;
}

// 인자 검사용: 문자열 전체가 올바른 숫자일 때만 성공
static int parse_long(const char *str, long *out) {
    errno = 0;
    char *end;
    long v = strtol(str, &end, 10);
    if (errno || end == str || *end != '\0') return 0;
    *out = v;
    return 1;
}
static int parse_double(const char *str, double *out) {
    errno = 0;
    char *end;
    double v = strtod(str, &end);
    if (errno || end == str || *end != '\0' || !isfinite(v)) return 0;
    *out = v;
    return 1;
}
static int parse_u64(const char *str, uint64_t *out) {
    if (str[0] < '0' || str[0] > '9') return 0;
    errno = 0;
    char *end;
    unsigned long long v = strtoull(str, &end, 10);
    if (errno || *end != '\0') return 0;
    *out = (uint64_t)v;
    return 1;
}

static inline uint64_t now_ns(clockid_t clk) {
    struct timespec t;
    clock_gettime(clk, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

int main(int argc, char **argv) {
    // ---------- [1] 인자 읽기 ----------
    if (argc < 5) {
        fprintf(stderr, "usage: %s <counter_path> <cpu> <max_sec> <out.csv> [interval_us] [max_samples]\n", argv[0]);
        return 1;
    }
    const char *path = argv[1];
    const char *out  = argv[4];
    long cpu_l;
    double max_sec;
    // 기본 buffer: 2백만 sample (~64 MB). Target 100us이면 최대 200초 분량.
    // 측정 중에 buffer를 늘리면(realloc) 그 순간 지연이 생기므로, 처음에 크게 잡고 꽉 차면 경고한다.
    uint64_t max_n = 2000000ull;  // 64MB cap per probe; 100us for 120s needs ~1.2M samples
    long interval_us = 100;

    long ncpu = sysconf(_SC_NPROCESSORS_CONF);
    if (!parse_long(argv[2], &cpu_l) || cpu_l < 0 || cpu_l >= ncpu || cpu_l >= CPU_SETSIZE_SAFE) {
        fprintf(stderr, "error: cpu must be an integer in [0, %ld)\n", ncpu); return 1;
    }
    if (!parse_double(argv[3], &max_sec) || max_sec <= 0) {
        fprintf(stderr, "error: max_sec must be a positive number\n"); return 1;
    }
    if (argc > 5 && (!parse_long(argv[5], &interval_us) || interval_us < 1 || interval_us > 1000000)) {
        fprintf(stderr, "error: interval_us must be in [1, 1000000]\n"); return 1;
    }
    if (argc > 6 && (!parse_u64(argv[6], &max_n) || max_n == 0)) {
        fprintf(stderr, "error: max_samples must be a positive integer\n"); return 1;
    }
    if (max_n > SIZE_MAX / sizeof(sample_t)) {           // 곱셈 overflow 방지
        fprintf(stderr, "error: max_samples too large\n"); return 1;
    }
    int cpu = (int)cpu_l;

    // ---------- [1b] SIGINT/SIGTERM이 오면 stop_flag만 세운다 ----------
    // 준비(큰 memory 초기화) 중에 Ctrl+C가 와도 CSV(0 sample)를 남기고 정상 종료하도록 가장 먼저 등록
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_signal;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    // ---------- [2] CPU core 고정 (affinity). priority(chrt)는 건드리지 않음 ----------
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) { perror("sched_setaffinity"); return 1; }
#else
    fprintf(stderr, "[counter_probe] note: CPU affinity는 Linux에서만 적용됨 (지금은 연습용 실행)\n");
#endif

    // ---------- [3] counter 파일은 처음에 한 번만 연다 (예전: 매번 open/close) ----------
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); return 1; }

    // ---------- [4] 기록할 memory를 미리 확보하고 미리 한 번 건드려 둔다 (prefault) ----------
    // 이유: 처음 쓰는 memory 페이지는 OS가 그 순간 준비하느라 느려짐(page fault).
    //       측정 도중에 그 지연이 끼지 않도록, 측정 전에 전부 0으로 한 번 써둔다.
    size_t bytes = (size_t)max_n * sizeof(sample_t);
    sample_t *s = malloc(bytes);
    if (!s) { fprintf(stderr, "malloc failed (%" PRIu64 " samples, %.2f GB)\n", max_n, bytes / 1e9); return 1; }
    memset(s, 0, bytes);

    // ---------- [6] 시계 기준점(anchor): 두 시계를 연달아 찍어둔다 (같은 순간은 아님, 수십 ns 차이) ----------
    // MONOTONIC_RAW: 간격 측정용 (NTP 보정 영향 없음)
    // REALTIME:      다른 프로그램(coll_bench.py) 로그와 시각을 맞추기 위한 연결고리
    uint64_t rt0   = now_ns(CLOCK_REALTIME);
    uint64_t mono0 = now_ns(CLOCK_MONOTONIC_RAW);
    uint64_t deadline = mono0 + (uint64_t)(max_sec * 1e9);

    // ---------- [7] 측정 loop: MONOTONIC for sleeping, RAW for trace timestamps ----------
    // Linux CLOCK_MONOTONIC is used only as the sleep scheduler; trace stays RAW.
    uint64_t period_ns = (uint64_t)interval_us * 1000ull;
    uint64_t next_wake_mono = now_ns(CLOCK_MONOTONIC);
    char buf[64];
    uint64_t n = 0, nfail = 0;
    const char *reason = "max_sec";
    for (;;) {
        if (stop_flag)  { reason = "signal";      break; }
        if (n >= max_n) { reason = "buffer_full"; break; }

        uint64_t ta = now_ns(CLOCK_MONOTONIC_RAW);          // 읽기 직전
        if (ta >= deadline) break;
        ssize_t r = pread(fd, buf, sizeof(buf) - 1, 0);     // 파일 처음(offset 0)부터 다시 읽기
        uint64_t tb = now_ns(CLOCK_MONOTONIC_RAW);          // 읽기 직후

        if (r < 0 && errno == EINTR && stop_flag) { reason = "signal"; break; }  // 종료 신호로 끊긴 read는 기록 안 함

        sample_t *p = &s[n++];
        p->ta = ta - mono0;
        p->dt = tb - ta;
        // 문자열 → 숫자 변환은 tb 이후 (bracket 밖). buffer를 꽉 채웠으면 잘렸을 수 있으니 실패 처리
        if (r > 0 && (size_t)r < sizeof(buf) - 1) {
            buf[r] = '\0';
            p->ok = parse_counter(buf, &p->val);
        } else {
            p->ok = 0;
        }
        if (!p->ok) { p->val = 0; nfail++; }
        // Avoid CPU-saturating busy polling. Missed schedules are not "caught up".
        next_wake_mono += period_ns;
        uint64_t cur_mono = now_ns(CLOCK_MONOTONIC);
        if (cur_mono > next_wake_mono) next_wake_mono = cur_mono;
        struct timespec wake = { .tv_sec = (time_t)(next_wake_mono / 1000000000ull),
                                 .tv_nsec = (long)(next_wake_mono % 1000000000ull) };
        while (!stop_flag && clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &wake, NULL) == EINTR) {}
    }
    uint64_t rt1   = now_ns(CLOCK_REALTIME);
    uint64_t mono1 = now_ns(CLOCK_MONOTONIC_RAW);
    close(fd);

    // ---------- [8] 측정이 끝난 뒤에야 파일에 쓴다 ----------
    FILE *f = fopen(out, "w");
    if (!f) { perror(out); return 1; }
    static char iobuf[1 << 20];
    setvbuf(f, iobuf, _IOFBF, sizeof(iobuf));
    // '#'로 시작하는 줄 = metadata (분석 코드가 읽어서 사용)
    fprintf(f, "# counter_path=%s\n", path);
    fprintf(f, "# cpu=%d\n", cpu);
    fprintf(f, "# target_interval_us=%ld\n", interval_us);
    fprintf(f, "# clock=CLOCK_MONOTONIC_RAW, ta_ns/tb_ns are relative to anchor_start.mono_raw_ns\n");
    fprintf(f, "# anchor_start realtime_ns=%" PRIu64 " mono_raw_ns=%" PRIu64 "\n", rt0, mono0);
    fprintf(f, "# anchor_end realtime_ns=%" PRIu64 " mono_raw_ns=%" PRIu64 "\n", rt1, mono1);
    fprintf(f, "# stop_reason=%s samples=%" PRIu64 " read_failures=%" PRIu64 "\n", reason, n, nfail);
    fprintf(f, "ta_ns,tb_ns,value,ok\n");
    for (uint64_t i = 0; i < n; i++)
        fprintf(f, "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu32 "\n",
                s[i].ta, s[i].ta + s[i].dt, s[i].val, s[i].ok);
    int save_failed = ferror(f);
    if (fclose(f) == EOF) save_failed = 1;
    free(s);
    if (save_failed) {
        fprintf(stderr, "[counter_probe] ERROR: CSV 저장 실패 (%s)\n", out);
        return 1;
    }

    // ---------- [9] 화면에 한 줄 요약 ----------
    fprintf(stderr, "[counter_probe] %s: %" PRIu64 " samples, %" PRIu64 " read failures, %.3f s, stop=%s\n",
            out, n, nfail, (double)(mono1 - mono0) / 1e9, reason);
    if (strcmp(reason, "buffer_full") == 0)
        fprintf(stderr, "[counter_probe] WARNING: buffer가 max_sec 전에 가득 참 → max_samples를 늘릴 것\n");
    return 0;
}
