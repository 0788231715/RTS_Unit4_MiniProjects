#!/usr/bin/env python3
"""
Mini Project 4.3 - Edge-versus-cloud latency simulator for a real-time IoT decision task
(Sections 4.8 URLLC and 4.9 edge computing / digital twin).

A vibration/temperature reading arrives every 20 ms; each needs one 'decide' computation
and the decision must be available within DEADLINE = 100 ms (end-to-end).

Latency models (all in milliseconds)
------------------------------------
EDGE   : transport ~ 1 ms; compute on limited local hardware: easy decision ~ N(20, 4),
         hard decision ~ N(70, 10).  Low variance, no heavy tail.
CLOUD  : upload leg + remote compute + download leg.
         legs ~ lognormal(median 15 ms, sigma 0.25); remote compute ~ N(8, 1.5) (powerful server);
         with probability p_cong each leg additionally suffers a congestion spike drawn from a
         heavy-tailed Pareto distribution (scale 40 ms, shape 1.6) - a genuinely long tail.
         Higher cong level => higher spike probability AND a 1.0+cong multiplier on leg median.
HYBRID : every event first passes a 10 ms edge screening step.
         easy decisions (70 %) are decided locally (screening + easy edge compute);
         hard decisions (30 %) are HEDGED: the request is sent to the cloud AND the edge starts
         the full local computation in parallel; the first answer to arrive wins.
         => worst case is bounded by  screen + local hard compute  (~10 + ~71 ms), because the
         local path never depends on the network, while a fast cloud answer can still win.
         (A simple timeout-then-fallback policy was tried first: its fallback path,
         screen + timeout + local compute, itself exceeds the 100 ms deadline - see report.)

Common random numbers: each event carries pre-drawn uniform/normal variates so that edge,
cloud and hybrid all face the identical event sequence and congestion sweeps are comparable.
"""
import math, random, statistics, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)
DEADLINE = 100.0
N = 20000
HARD_FRACTION = 0.30
BASE_CONG = 0.10


def make_events(n=N, seed=2026):
    rnd = random.Random(seed)
    ev = []
    for _ in range(n):
        ev.append(dict(hard=rnd.random() < HARD_FRACTION,
                       g_edge=rnd.gauss(0, 1), g_cloud=rnd.gauss(0, 1),
                       g_up=rnd.gauss(0, 1), g_dn=rnd.gauss(0, 1),
                       u_up=rnd.random(), u_dn=rnd.random(),
                       u_p_up=rnd.random(), u_p_dn=rnd.random(),
                       g_screen=rnd.gauss(0, 1)))
    return ev


def pareto(u, scale=40.0, shape=1.6):
    """Inverse-CDF Pareto sample from uniform u (heavy tail)."""
    return min(3000.0, scale / ((1.0 - min(u, 0.999999)) ** (1.0 / shape)) - scale)   # capped at 3 s (timeout)


def leg(g, u_spike, u_mag, cong):
    median = 15.0 * (1.0 + cong)
    lat = median * math.exp(0.25 * g)
    if u_spike < cong:                       # congestion spike on this leg
        lat += pareto(u_mag)
    return lat


def edge_latency(e):
    comp = (70.0 + 10.0 * e["g_edge"]) if e["hard"] else (20.0 + 4.0 * e["g_edge"])
    return max(1.0, comp) + 1.0              # + ~1 ms transport


def cloud_latency(e, congestion):
    up = leg(e["g_up"], e["u_up"], e["u_p_up"], congestion)
    dn = leg(e["g_dn"], e["u_dn"], e["u_p_dn"], congestion)
    comp = max(1.0, 8.0 + 1.5 * e["g_cloud"])
    return up + comp + dn


def hybrid_latency(e, congestion):
    screen = max(2.0, 10.0 + 2.0 * e["g_screen"])
    if not e["hard"]:
        return screen + max(1.0, 20.0 + 4.0 * e["g_edge"]) + 1.0
    local = max(1.0, 70.0 + 10.0 * e["g_edge"]) + 1.0
    return screen + min(cloud_latency(e, congestion), local)     # hedged: first answer wins


def pct(sorted_x, p):
    k = min(len(sorted_x) - 1, int(math.ceil(p / 100 * len(sorted_x))) - 1)
    return sorted_x[k]


def stats(x):
    s = sorted(x)
    return dict(mean=statistics.mean(x), p95=pct(s, 95), p99=pct(s, 99), worst=s[-1],
                comp=sum(1 for v in x if v <= DEADLINE) / len(x) * 100)


def main():
    ev = make_events()
    lines = []

    def P(s=""):
        print(s); lines.append(s)

    edge = [edge_latency(e) for e in ev]
    cloud = [cloud_latency(e, BASE_CONG) for e in ev]
    hyb = [hybrid_latency(e, BASE_CONG) for e in ev]

    P(f"Events: {N}   deadline: {DEADLINE:.0f} ms   hard-decision fraction: {HARD_FRACTION:.0%}   "
      f"baseline cloud congestion: {BASE_CONG:.2f}\n")
    P(f"{'architecture':<22}{'mean':>8}{'p95':>8}{'p99':>8}{'worst':>9}{'compliance':>12}")
    for name, x in (("Edge only", edge), (f"Cloud (cong={BASE_CONG:.2f})", cloud),
                    (f"Hybrid (cong={BASE_CONG:.2f})", hyb)):
        s = stats(x)
        P(f"{name:<22}{s['mean']:8.1f}{s['p95']:8.1f}{s['p99']:8.1f}{s['worst']:9.1f}{s['comp']:11.2f}%")
    P("(all latencies in ms)\n")

    # compliance versus congestion
    levels = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    P("Compliance (%) versus cloud congestion level")
    P(f"{'cong':>6}{'cloud':>9}{'hybrid':>9}{'edge':>9}{'cloud p99':>11}{'cloud worst':>13}")
    ec = stats(edge)["comp"]
    cc, hc = [], []
    for L in levels:
        c = [cloud_latency(e, L) for e in ev]
        h = [hybrid_latency(e, L) for e in ev]
        sc, sh = stats(c), stats(h)
        cc.append(sc["comp"]); hc.append(sh["comp"])
        P(f"{L:6.2f}{sc['comp']:9.2f}{sh['comp']:9.2f}{ec:9.2f}{sc['p99']:11.1f}{sc['worst']:13.1f}")

    # Figure 1: overlaid histograms + deadline
    fig, ax = plt.subplots(figsize=(10, 4.6))
    bins = [i * 5 for i in range(0, 80)]
    ax.hist(edge, bins=bins, alpha=0.55, label="Edge", color="tab:green", density=True)
    ax.hist(cloud, bins=bins, alpha=0.55, label=f"Cloud (cong={BASE_CONG})", color="tab:red", density=True)
    ax.hist(hyb, bins=bins, alpha=0.45, label="Hybrid", color="tab:blue", density=True)
    ax.axvline(DEADLINE, color="k", ls="--", label=f"deadline {DEADLINE:.0f} ms")
    ax.set_xlim(0, 300); ax.set_yscale("log")
    ax.set_xlabel("end-to-end latency (ms)"); ax.set_ylabel("density (log scale)")
    ax.set_title("Figure 1 - latency distributions (axis clipped at 300 ms; cloud tail extends further)")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/fig1_latency_distributions.png", dpi=110); plt.close(fig)

    # Figure 2: compliance vs congestion
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(levels, cc, "o-", color="tab:red", label="Cloud only")
    ax.plot(levels, hc, "s-", color="tab:blue", label="Hybrid")
    ax.axhline(ec, color="tab:green", ls="--", label="Edge only (flat reference)")
    ax.set_xlabel("congestion level"); ax.set_ylabel("deadline compliance (%)")
    ax.set_ylim(min(cc) - 3, 101); ax.set_title("Figure 2 - compliance vs congestion")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig2_compliance_vs_congestion.png", dpi=110); plt.close(fig)

    # Figure 3 (bonus): empirical CDF
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for name, x, c in (("Edge", edge, "tab:green"), ("Cloud", cloud, "tab:red"), ("Hybrid", hyb, "tab:blue")):
        s = sorted(x); ax.plot(s, [(i + 1) / len(s) for i in range(len(s))], label=name, color=c)
    ax.axvline(DEADLINE, color="k", ls="--"); ax.set_xlim(0, 400)
    ax.set_xlabel("latency (ms)"); ax.set_ylabel("empirical CDF"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Figure 3 - empirical CDF of end-to-end latency")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_cdf.png", dpi=110); plt.close(fig)

    with open(f"{OUT}/results_log.txt", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
