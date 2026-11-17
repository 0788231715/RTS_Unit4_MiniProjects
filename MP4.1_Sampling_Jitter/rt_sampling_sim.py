#!/usr/bin/env python3
"""
Mini Project 4.1 - Simulating sampling jitter and processing overruns
in an embedded sensing-and-control loop (Sections 4.1 and 4.5).

Model (documented so the results can be audited)
------------------------------------------------
* Physical process : true_signal(t) = A*sin(2*pi*t/P) + small deterministic "noise".
* Ideal loop       : sample instants t_k = k*T (zero jitter, fixed nominal computation
                     budget of 0.25*T before actuation) - the
                     idealised fixed-period PLC scan cycle of Section 4.5.
* Real loop        : cycle k is *scheduled* at k*T but cannot start before the previous
                     cycle has finished (clock).  Sample time = max(start + jitter_k, clock).
                     Computation c_k is short normally, but with probability p_over it is
                     drawn from an overrun range (> T).  The actuation command is issued
                     when computation ends (sample_t + c_k).  An overrun therefore pushes
                     the NEXT cycle's start later - exactly like an overrunning PLC scan.
* Controllers      : (a) threshold controller: command on every upward crossing of THR;
                     (b) P-type controller: u = Kp*(setpoint - y) held (zero-order hold)
                     until the next actuation.
"""
import random, statistics, math, csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)

AMP, PERIOD, THR = 1.0, 2.0, 0.5       # signal amplitude, signal period (s), threshold
RUN_LEN = 20.0                         # simulated seconds
T_NOM = 0.05                           # nominal sampling period (20 Hz)
C_NOM = 0.25                           # nominal computation budget as a fraction of T (ideal loop)


def true_signal(t):
    """Slow sine (temperature/position stand-in) + small continuous noise."""
    noise = 0.010 * math.sin(2 * math.pi * 13.7 * t) + 0.006 * math.sin(2 * math.pi * 31.1 * t + 1)
    return AMP * math.sin(2 * math.pi * t / PERIOD) + noise


def ideal_run(T=T_NOM, run_len=RUN_LEN):
    n = int(run_len / T)
    ts = [k * T for k in range(n)]
    # sample times, values, actuation times (ideal actuation = sample + nominal computation budget)
    return ts, [true_signal(t) for t in ts], [t + C_NOM * T for t in ts]


def real_run(T=T_NOM, run_len=RUN_LEN, J=0.0, p_over=0.0, seed=1,
             over_range=(1.2, 3.0), norm_range=(0.10, 0.40)):
    """Jitter J is an absolute bound in seconds.  Returns dict of per-cycle series."""
    rnd = random.Random(seed)
    n = int(run_len / T)
    clock = 0.0
    S, V, ACT, JIT, C, OVR = [], [], [], [], [], []
    for k in range(n):
        start = max(clock, k * T)                         # cannot start before previous cycle ends
        jit = rnd.uniform(-J, J) if J > 0 else 0.0
        t_s = max(start + jit, clock)                     # sample instant
        if rnd.random() < p_over:                         # processing overrun (WCET violation)
            c = rnd.uniform(*over_range) * T
        else:
            c = rnd.uniform(*norm_range) * T
        S.append(t_s); V.append(true_signal(t_s)); JIT.append(jit); C.append(c)
        OVR.append(c > T)
        ACT.append(t_s + c)                               # command leaves when computation ends
        clock = t_s + c
    return dict(S=S, V=V, ACT=ACT, JIT=JIT, C=C, OVR=OVR, T=T)


def crossings(values, thr=THR):
    """Indices of samples where an upward threshold crossing is detected."""
    return [i for i in range(1, len(values)) if values[i - 1] < thr <= values[i]]


def timing_stats(r):
    S, T = r["S"], r["T"]
    iv = [S[i] - S[i - 1] for i in range(1, len(S))]
    miss = sum(1 for k in range(len(S)) if r["ACT"][k] > (k + 1) * T + 1e-12) / len(S) * 100
    over15 = sum(1 for x in iv if x > 1.5 * T) / len(iv) * 100
    return dict(mean_iv=statistics.mean(iv), sd_iv=statistics.pstdev(iv),
                max_over=max(0.0, max(c - T for c in r["C"])),
                pct_missed=miss, pct_gt15=over15)


def threshold_delays(ideal, real):
    """Match i-th ideal crossing with i-th real crossing; return actuation delays."""
    _, vi, ai = ideal
    ci = crossings(vi)
    cr = crossings(real["V"])
    n = min(len(ci), len(cr))
    delays = [real["ACT"][cr[i]] - ai[ci[i]] for i in range(n)]
    return delays, len(ci), len(cr)


def p_controller_error(real, T, Kp=2.0, sp=0.0, run_len=RUN_LEN, dt=0.005):
    """RMS difference between zero-order-held P outputs of the real and ideal loops."""
    ti = [k * T for k in range(int(run_len / T))]
    ideal_u = [Kp * (sp - true_signal(t)) for t in ti]

    def held(times, u):
        out, j = [], -1
        for g in [i * dt for i in range(int(run_len / dt))]:
            while j + 1 < len(times) and times[j + 1] <= g:
                j += 1
            out.append(u[j] if j >= 0 else 0.0)
        return out
    uh_i = held([t + C_NOM * T for t in ti], ideal_u)
    uh_r = held(real["ACT"], [Kp * (sp - v) for v in real["V"]])
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(uh_i, uh_r)) / len(uh_i))


def evaluate(J, p, seed=1, T=T_NOM):
    ideal = ideal_run(T)
    r = real_run(T=T, J=J, p_over=p, seed=seed)
    st = timing_stats(r)
    d, n_i, n_r = threshold_delays(ideal, r)
    st.update(mean_delay=statistics.mean(d) if d else float("nan"),
              max_delay=max(d) if d else float("nan"),
              crossings=f"{n_r}/{n_i}", p_err=p_controller_error(r, T))
    return ideal, r, st, d


def plot_cfg(name, ideal, r, d, J, p):
    ti, vi, ai = ideal
    tt = [i * 0.002 for i in range(int(RUN_LEN / 0.002))]
    fig, ax = plt.subplots(2, 1, figsize=(10, 7))
    ax[0].plot(tt, [true_signal(t) for t in tt], color="0.6", lw=1, label="true signal")
    ax[0].plot(ti, vi, "o", ms=3, color="tab:blue", label="ideal samples")
    ax[0].plot(r["S"], r["V"], "x", ms=4, color="tab:red", label="jittered/overrun samples")
    ax[0].axhline(THR, color="k", ls=":", lw=0.8, label="threshold")
    ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("signal"); ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set_title(f"Figure 1 - {name}: J={J/T_NOM*100:.0f}% of T, overrun prob={p:.2f}")
    iv = [r["S"][i] - r["S"][i - 1] - T_NOM for i in range(1, len(r["S"]))]
    ax[1].plot(range(1, len(r["S"])), iv, lw=0.9, color="tab:purple")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xlabel("cycle index k"); ax[1].set_ylabel("interval - T_nominal (s)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_{name}.png", dpi=110); plt.close(fig)

    ci, cr = crossings(vi), crossings(r["V"])
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.vlines([ai[i] for i in ci], 0.55, 1.0, color="tab:blue", lw=2, label="ideal actuation")
    ax.vlines([r["ACT"][i] for i in cr], 0.0, 0.45, color="tab:red", lw=2, label="jittered/overrun actuation")
    ax.set_yticks([]); ax.set_xlabel("time (s)"); ax.legend(loc="upper right", fontsize=8)
    md = statistics.mean(d) * 1000 if d else float("nan")
    xd = max(d) * 1000 if d else float("nan")
    ax.set_title(f"Figure 2 - {name}: mean actuation delay = {md:.1f} ms, max = {xd:.1f} ms")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig2_{name}.png", dpi=110); plt.close(fig)


def row(label, st):
    return (f"{label:<26}{st['mean_iv']*1000:9.2f}{st['sd_iv']*1000:9.2f}{st['max_over']*1000:10.1f}"
            f"{st['pct_missed']:9.1f}{st['pct_gt15']:9.1f}{st['mean_delay']*1000:10.2f}"
            f"{st['max_delay']*1000:10.2f}{st['crossings']:>10}{st['p_err']:9.3f}")


HDR = (f"{'configuration':<26}{'mean_iv':>9}{'sd_iv':>9}{'max_over':>10}{'%miss':>9}{'%>1.5T':>9}"
       f"{'mean_dly':>10}{'max_dly':>10}{'cross':>10}{'P_rms':>9}\n"
       f"{'':<26}{'(ms)':>9}{'(ms)':>9}{'(ms)':>10}{'':>9}{'':>9}{'(ms)':>10}{'(ms)':>10}{'det/ideal':>10}{'':>9}")


def main():
    lines = []

    def P(s=""):
        print(s); lines.append(s)

    P("=== Sanity check: J=0, p=0 must reproduce the ideal baseline ===")
    ideal, r, st, d = evaluate(0.0, 0.0)
    P(HDR); P(row("J=0, p=0", st)); P()

    P("=== Full sweep: 3 jitter bounds x 3 overrun probabilities (seed=1) ===")
    P(HDR)
    csv_rows = []
    for jf in (0.0, 0.10, 0.30):
        for p in (0.0, 0.05, 0.20):
            _, _, st, _ = evaluate(jf * T_NOM, p)
            P(row(f"J={jf*100:.0f}%T, p={p:.2f}", st))
            csv_rows.append([jf, p] + [st[k] for k in ("mean_iv", "sd_iv", "max_over", "pct_missed",
                                                       "pct_gt15", "mean_delay", "max_delay", "p_err")])
    with open(f"{OUT}/sweep_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["J_frac", "p_over", "mean_iv", "sd_iv", "max_over", "pct_missed", "pct_gt15",
                    "mean_delay", "max_delay", "p_rms_err"])
        w.writerows(csv_rows)

    P("\n=== Lab task 1: jitter-only vs overrun-only (threshold controller delay) ===")
    P(HDR)
    for jf in (0.10, 0.30, 0.60):
        _, _, st, _ = evaluate(jf * T_NOM, 0.0); P(row(f"jitter only {jf*100:.0f}%T", st))
    for p in (0.05, 0.10, 0.20):
        _, _, st, _ = evaluate(0.0, p); P(row(f"overrun only p={p:.2f}", st))

    P("\n=== Lab task 3: compliance metric (% cycles with interval > 1.5 T) vs sd of interval ===")
    for jf, p in ((0.30, 0.0), (0.0, 0.10), (0.30, 0.10)):
        _, _, st, _ = evaluate(jf * T_NOM, p)
        P(f"J={jf*100:.0f}%T p={p:.2f}: sd_iv = {st['sd_iv']*1000:6.2f} ms | "
          f"%>1.5T = {st['pct_gt15']:5.1f}% | max_over = {st['max_over']*1000:6.1f} ms")

    P("\n=== Lab task 4: halve T_nominal, hold absolute jitter = 7.5 ms ===")
    for T in (0.05, 0.025):
        ideal = ideal_run(T)
        rr = real_run(T=T, J=0.0075, p_over=0.0, seed=1)
        d, ni, nr = threshold_delays(ideal, rr)
        st = timing_stats(rr)
        P(f"T={T*1000:5.1f} ms: relative jitter = {0.0075/T*100:5.1f}% of T | "
          f"sd_iv = {st['sd_iv']*1000:5.2f} ms | mean delay = {statistics.mean(d)*1000:6.2f} ms")

    P("\n=== Figures for three configurations ===")
    cfgs = {"A_mild_jitter": (0.10, 0.0), "B_heavy_jitter": (0.30, 0.0),
            "C_jitter_plus_overruns": (0.30, 0.10)}
    for name, (jf, p) in cfgs.items():
        ideal, r, st, d = evaluate(jf * T_NOM, p)
        plot_cfg(name, ideal, r, d, jf * T_NOM, p)
        P(f"saved fig1_{name}.png and fig2_{name}.png  (mean delay {statistics.mean(d)*1000:.1f} ms)")
    with open(f"{OUT}/results_log.txt", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
