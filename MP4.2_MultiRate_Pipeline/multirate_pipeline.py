#!/usr/bin/env python3
"""
Mini Project 4.2 - Multi-rate environmental monitoring pipeline with
timing-compliance reporting (Sections 4.1 and 4.2).

One gateway loop services several sensor streams that share a single processor.

Release / classification model
------------------------------
* Stream i releases sample k at r = k * P_i (its required period).
* A release is SERVICED when the gateway starts reading the sensor at time s >= r.
    on-time : s - r <= TOL * P_i        (TOL = 25 % of the stream's period)
    late    : TOL * P_i < s - r < P_i   (still serviced before the next release)
    missed  : not started before the next release r + P_i  (sample dropped / stale)
* Each read takes a random processing time c (light-tailed normally, occasionally a long
  overrun).  The CPU is busy for c seconds (the gateway cannot do two reads at once).
* Period overrun of a serviced sample = max(0, s - r).

Scheduling rules (documented, selectable)
-----------------------------------------
EDF : earliest-deadline-first among pending releases (deadline = r + P_i).
FP  : fixed priority, fastest stream (shortest period) = highest priority, like
      lowest-ID-wins arbitration on CAN (Section 4.2): bounded latency for the top
      priority, possible starvation for the slowest stream.
RR  : round-robin over streams that have a pending release (no urgency information).
"""
import heapq, random, statistics, math, os
from dataclasses import dataclass, field
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)
TOL = 0.25
RUN_LEN = 120.0


@dataclass
class Stream:
    name: str
    period: float
    thr: float
    base: float
    amp: float
    sig_period: float
    k: int = 0                       # next release index
    on_time: int = 0
    late: int = 0
    missed: int = 0
    overruns: list = field(default_factory=list)
    samples: list = field(default_factory=list)      # (time, value, cls)
    alerts: list = field(default_factory=list)       # (true_cross, detect_time)
    alert_active: bool = False
    last_cross: float = None

    def signal(self, t):
        return self.base + self.amp * math.sin(2 * math.pi * t / self.sig_period) \
            + 0.02 * self.amp * math.sin(2 * math.pi * 7.3 * t)

    def true_crossings(self):
        out, prev, dt = [], self.signal(0), 0.001
        n = int(RUN_LEN / dt)
        for i in range(1, n):
            v = self.signal(i * dt)
            if prev < self.thr <= v:
                out.append(i * dt)
            prev = v
        return out


def make_streams(load):
    s = [Stream("temperature", 0.50, 28.0, 22.0, 8.0, 40.0),
         Stream("humidity",    1.00, 65.0, 50.0, 20.0, 50.0),
         Stream("air_quality", 2.00, 120.0, 80.0, 60.0, 60.0)]
    if load == 4:
        s.insert(0, Stream("vibration", 0.20, 0.8, 0.3, 0.6, 30.0))
    return s


def proc_time(rnd):
    """Light-tailed read time with an occasional long tail (overrun)."""
    if rnd.random() < 0.05:
        return rnd.uniform(0.20, 0.60)
    return rnd.uniform(0.020, 0.080)


def simulate(rule, load, seed=7):
    rnd = random.Random(seed)
    streams = make_streams(load)
    truth = {s.name: s.true_crossings() for s in streams}
    heap = [(0.0, i) for i in range(len(streams))]   # (next release time, stream idx) via heapq
    heapq.heapify(heap)
    pending = {}                                     # idx -> release time of oldest pending release
    t, rr = 0.0, 0
    busy_time = 0.0
    while True:
        # move all releases <= t into 'pending' (dropping stale ones = missed)
        while heap and heap[0][0] <= t:
            r, i = heapq.heappop(heap)
            st = streams[i]
            if i in pending:                         # previous release never serviced -> missed
                st.missed += 1
            pending[i] = r
            st.k += 1
            if r + st.period < RUN_LEN:
                heapq.heappush(heap, (r + st.period, i))
        if not pending:
            if not heap:
                break
            t = heap[0][0]
            continue
        # choose a stream to service
        if rule == "EDF":
            i = min(pending, key=lambda j: pending[j] + streams[j].period)
        elif rule == "FP":
            i = min(pending, key=lambda j: streams[j].period)
        else:  # RR
            order = [(rr + n) % len(streams) for n in range(len(streams))]
            i = next(j for j in order if j in pending)
            rr = (i + 1) % len(streams)
        st, r = streams[i], pending.pop(i)
        lateness = t - r
        if lateness <= TOL * st.period:
            cls = "on"; st.on_time += 1
        else:
            cls = "late"; st.late += 1
        st.overruns.append(max(0.0, lateness))
        val = st.signal(t)
        st.samples.append((t, val, cls))
        c = proc_time(rnd)
        t += c
        busy_time += c
        # alert evaluation right after the sample (detection time = end of processing)
        if val >= st.thr and not st.alert_active:
            st.alert_active = True
            tc = [x for x in truth[st.name] if x <= t]
            st.alerts.append((tc[-1] if tc else float("nan"), t))
        elif val < st.thr:
            st.alert_active = False
    for i in pending:                                # releases still pending at the end
        streams[i].missed += 1
    return streams, busy_time / RUN_LEN, truth


def report(streams, util, title, lines):
    def P(s=""):
        print(s); lines.append(s)
    P(f"--- {title}  (CPU utilisation {util*100:.1f}%) ---")
    P(f"{'stream':<12}{'period':>7}{'required':>9}{'on-time':>8}{'late':>6}{'missed':>7}{'on-time%':>9}"
      f"{'mean ovr(ms)':>13}{'max ovr(ms)':>12}{'alerts':>7}{'mean alert lat(ms)':>20}")
    for s in streams:
        req = s.on_time + s.late + s.missed
        mo = statistics.mean(s.overruns) * 1000 if s.overruns else 0
        xo = max(s.overruns) * 1000 if s.overruns else 0
        lat = [d - c for c, d in s.alerts if not math.isnan(c)]
        ml = statistics.mean(lat) * 1000 if lat else float("nan")
        P(f"{s.name:<12}{s.period:7.2f}{req:9d}{s.on_time:8d}{s.late:6d}{s.missed:7d}"
          f"{s.on_time/req*100:9.1f}{mo:13.1f}{xo:12.1f}{len(s.alerts):7d}{ml:20.1f}")
    P()


def plots(streams, cfg):
    n = len(streams)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.3 * n), sharex=True)
    if n == 1: axes = [axes]
    col = {"on": "tab:green", "late": "tab:orange"}
    for ax, s in zip(axes, streams):
        tt = [i * 0.05 for i in range(int(RUN_LEN / 0.05))]
        ax.plot(tt, [s.signal(t) for t in tt], color="0.7", lw=1)
        ax.axhline(s.thr, color="k", ls=":", lw=0.8)
        for c in ("on", "late"):
            pts = [(a, b) for a, b, k in s.samples if k == c]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], ".", ms=3, color=col[c], label=c)
        ax.set_ylabel(s.name, fontsize=8)
        ax.legend(loc="upper right", fontsize=6, ncol=2)
    axes[0].set_title(f"{cfg}: true signal, threshold, serviced samples (green on-time / orange late)")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout(); fig.savefig(f"{OUT}/signals_{cfg}.png", dpi=100); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = range(n); w = 0.27
    ax.bar([i - w for i in x], [s.on_time for s in streams], w, label="on-time", color="tab:green")
    ax.bar(list(x), [s.late for s in streams], w, label="late", color="tab:orange")
    ax.bar([i + w for i in x], [s.missed for s in streams], w, label="missed", color="tab:red")
    ax.set_xticks(list(x)); ax.set_xticklabels([s.name for s in streams], fontsize=8)
    ax.set_ylabel("samples"); ax.set_title(f"{cfg}: on-time / late / missed per stream"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/compliance_{cfg}.png", dpi=100); plt.close(fig)


def main():
    lines = []
    summary = []
    for load in (3, 4):
        for rule in ("EDF", "FP", "RR"):
            streams, util, _ = simulate(rule, load)
            cfg = f"{rule}_{load}streams"
            report(streams, util, f"{rule} scheduling, {load} streams", lines)
            plots(streams, cfg)
            tot_m = sum(s.missed for s in streams); tot = sum(s.on_time + s.late + s.missed for s in streams)
            summary.append((cfg, util, tot_m, tot, min(streams, key=lambda s: s.on_time / (s.on_time + s.late + s.missed)).name))
    print("=== Summary: worst-served stream per configuration ===")
    lines.append("=== Summary: worst-served stream per configuration ===")
    for cfg, u, m, t, w in summary:
        s = f"{cfg:<16} util={u*100:5.1f}%  missed={m:4d}/{t:<5d} ({m/t*100:4.1f}%)  worst stream (lowest on-time %): {w}"
        print(s); lines.append(s)
    with open(f"{OUT}/compliance_report.txt", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
