"""
aere_simulation.py -- Monte Carlo generation of Table 5, Figure 3 (Systemic Error
Rate vs kappa), the sensitivity suite (Table 6), and
Experiment IV (Table 11, Figure 7), for
"Earning the Benefit of the Doubt: An Auditable Authorial-Effort
 Mechanism for Resolving Peer Review Disagreements"
(submitted to Research Evaluation).

MODEL (paper, Secs. 5.1-5.2)
  * Latent quality Q ~ Beta(a, a) on [0, 1].
  * Dyad reviewers perceive X_i = Q + N(0, sigma^2); vote accept iff X_i > c.
  * Tie-breaking third reviewer (R3) is a colder, less experienced draw
    (Kovanis et al. 2016; Adam 2025): X_3 = Q + N(0, (m3*sigma)^2),
    accept iff X_3 > c3.
  * kappa is Fleiss' kappa over the three-rater benchmark, traced by varying the
    ambient review noise sigma; the dyadic Cohen's kappa is reported alongside.
    The baseline sigma_R = 0.25 yields kappa ~= 0.30 (Table 5 baseline).
  * Error rates are conditional on an initial 1-1 split -- the deadlock
    population the mechanism is designed to resolve.
      - R3 regime: the split is broken by S3.
      - AERE regime: the author participates iff V >= C(e*, Q), where
        e* = (exp(tau/eta) - 1)/Q  is the minimum effort with
        R_e = eta*ln(1 + e*Q) = tau, and C(e, Q) = k*e^2/Q. Participation is
        therefore Q >= Q_w with Q_w = (k*(e^{tau/eta}-1)^2/V)^{1/3}.
        Participants face the dual audit Psi = A1 ^ A2, modeled as two noisy
        quality reads Y_j = Q + N(0, sigma_a^2), pass iff Y_j > c_a, with
        sigma_a = s0*(sigma/0.25)^alpha (auditors retain contextual knowledge
        of M0, hence are more accurate than a cold-start referee, but degrade
        with the ambient evaluative noise).
  * Type I: rejection given Q > 0.7. Type II: acceptance given Q < 0.4.

SAMPLING DESIGN (stratified)
  Every reported operating point accumulates deadlocked manuscripts until BOTH
  (i) at least N_SPLITS_MIN deadlocks are collected AND (ii) each conditioning
  tail (Q > 0.7 and Q < 0.4) holds at least the requested minimum number of
  deadlocked manuscripts. Rates therefore carry controlled binomial Monte Carlo
  standard errors, reported as +-1 SE, at every kappa -- including the
  high-agreement end where quality-tail deadlocks are rare events.

CALIBRATION (transparency)
  The free parameters (c, a, c3, m3 | c_a, s0) are pinned down by a two-stage
  grid search that minimizes the distance, at sigma_R = 0.25, to a stipulated
  target error profile (R3: 29.1/16.2; AERE: 11.4/3.9) under the constraint
  kappa ~= 0.30. These targets are CALIBRATION DEVICES, not empirical
  estimates: the paper reports only the rates the calibrated model actually
  attains (printed below as Table 5), and Figure 3 shows the calibrated
  model's implied behavior across kappa. Q_w is set so that 65% of sub-par
  manuscripts (Q < 0.4) self-select out, which back-solves the cost scale k.

Deterministic given the fixed seeds below (developed under numpy 2.x; the
Generator/PCG64 bit streams are stable across recent numpy releases).
Runtime: ~3-4 minutes end to end. Outputs: console (LaTeX-ready blocks) and
aere_figure3_data.csv in the working directory.
"""
import csv
import numpy as np

# ----------------------------- configuration --------------------------------
TAU, ETA, V = 0.18, 0.40, 10.0   # contest primitives (illustrative units)
SIGMA_BASE = 0.25                # ambient noise at the kappa ~= 0.30 baseline
ALPHA = 0.5                      # audit-noise scaling exponent (Sec. 5.5 varies it)
WITHDRAW_FRAC = 0.65             # P(Q < Q_w | Q < 0.4): 65% self-selection
N_CAL = 300_000                  # manuscripts per calibration evaluation
N_SPLITS_MIN = 10_000            # minimum deadlocks per plotted point
N_TAIL_FIG = 5_000               # minimum tail counts per Figure-3 point
N_TAIL_TABLE = 20_000            # minimum tail counts for the Table-4 baseline
N_TAIL_SENS = 4_000              # minimum tail counts for sensitivity cells
BATCH = 400_000                  # manuscripts per rejection-sampling batch
MAX_BATCHES = 2_000              # hard cap per operating point
SEED = 20260713
Q_HI, Q_LO = 0.7, 0.4            # quality-tail definitions (Type I / Type II)

# Calibration targets (devices only -- see header note).
TARGETS_R3 = (0.291, 0.162)
TARGETS_AERE = (0.114, 0.039)

EXPM1 = np.expm1(TAU / ETA)


# ------------------------------- primitives ---------------------------------
def qw_from_withdrawal(a, frac=WITHDRAW_FRAC, n=2_000_000):
    """Q_w s.t. P(Q < Q_w | Q < Q_LO) = frac (self-selection of sub-par work)."""
    q = np.random.default_rng(1).beta(a, a, n)
    return np.quantile(q[q < Q_LO], frac)


def draw(sigma, c, a, c3, m3, ca, s0, qw, N, rng,
         alpha=ALPHA, author_sigma=0.0, leniency=0.0, cold_dyad=False):
    """One batch of N manuscripts.

    Returns Q, n_acc (triad votes for Fleiss), dyad votes (S1, S2), split flag,
    the R3 tie-break vote S3 (with optional leniency: auto-accept w.p.
    `leniency`), the AERE decision (with optional authorial self-assessment
    noise and optional leniency of the proponent auditor A1), and -- if
    requested -- a capacity-matched cold-dyad tie-break (two fresh referees,
    unanimity to accept).
    """
    Q = rng.beta(a, a, N)
    S1 = Q + rng.normal(0, sigma, N) > c
    S2 = Q + rng.normal(0, sigma, N) > c
    S3 = Q + rng.normal(0, m3 * sigma, N) > c3

    n_acc = S1.astype(int) + S2.astype(int) + S3.astype(int)
    split = S1 != S2

    if leniency > 0.0:
        lenient3 = rng.random(N) < leniency
        S3_dec = S3 | lenient3          # negligent tie-breaker defaults to accept
    else:
        S3_dec = S3

    if author_sigma > 0.0:
        participate = Q + rng.normal(0, author_sigma, N) >= qw
    else:
        participate = Q >= qw
    sa = s0 * (sigma / SIGMA_BASE) ** alpha
    pass1 = Q + rng.normal(0, sa, N) > ca
    pass2 = Q + rng.normal(0, sa, N) > ca
    if leniency > 0.0:
        lenient1 = rng.random(N) < leniency
        pass1 = pass1 | lenient1        # captured proponent auditor waves A1 through
    accept_A = participate & pass1 & pass2

    cold2 = None
    if cold_dyad:
        C1 = Q + rng.normal(0, sigma, N) > c
        C2 = Q + rng.normal(0, sigma, N) > c
        cold2 = C1 & C2                 # unanimity of two fresh referees
    return Q, n_acc, S1, S2, split, S3_dec, accept_A, cold2


def fleiss(n_acc):
    """Fleiss' kappa, binary categories, three raters."""
    Pi = (n_acc * (n_acc - 1) + (3 - n_acc) * (2 - n_acc)) / 6.0
    p = n_acc.mean() / 3.0
    Pe = p**2 + (1 - p)**2
    return (Pi.mean() - Pe) / (1 - Pe)


def cohen(S1, S2):
    """Cohen's kappa for the initial dyad (identically distributed raters)."""
    po = np.mean(S1 == S2)
    p1, p2 = S1.mean(), S2.mean()
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    return (po - pe) / (1 - pe)


def safe_rate(x):
    """Proportion with binomial SE; returns (nan, nan, 0) on empty input."""
    n = x.size
    if n == 0:
        return np.nan, np.nan, 0
    p = float(np.mean(x))
    return p, float(np.sqrt(max(p * (1 - p), 0.0) / n)), n


def evaluate(sigma, params, N, seed):
    """Single-batch evaluation (calibration only; no tail stratification)."""
    rng = np.random.default_rng(seed)
    Q, n_acc, S1, S2, split, S3, accept_A, _ = draw(sigma, *params, N, rng)
    hi, lo = Q > Q_HI, Q < Q_LO
    t1 = np.mean(~S3[split & hi]); t2 = np.mean(S3[split & lo])
    ta = np.mean(~accept_A[split & hi]); tb = np.mean(accept_A[split & lo])
    return fleiss(n_acc), t1, t2, ta, tb


def evaluate_on_splits(sigma, params, seed, min_splits=N_SPLITS_MIN,
                       min_tail=N_TAIL_FIG, **variant):
    """Stratified rejection sampling of the deadlock population.

    Accumulates batches until min_splits deadlocks are collected AND each
    quality tail holds at least min_tail deadlocked manuscripts. Returns a
    dict with realized kappas, rates, SEs, and effective sample sizes.
    """
    rng = np.random.default_rng(seed)
    Qs, S3s, As, C2s = [], [], [], []
    kap_chunks, dyad_chunks = [], []
    got = n_hi = n_lo = 0
    for _ in range(MAX_BATCHES):
        Q, n_acc, S1, S2, split, S3, accept_A, cold2 = draw(
            sigma, *params, BATCH, rng, **variant)
        kap_chunks.append(n_acc); dyad_chunks.append((S1, S2))
        Qs.append(Q[split]); S3s.append(S3[split]); As.append(accept_A[split])
        if cold2 is not None:
            C2s.append(cold2[split])
        got += int(split.sum())
        qs = Q[split]
        n_hi += int((qs > Q_HI).sum()); n_lo += int((qs < Q_LO).sum())
        if got >= min_splits and n_hi >= min_tail and n_lo >= min_tail:
            break
    else:
        raise RuntimeError(
            f"tail stratification not reached at sigma={sigma:.3f}; "
            f"raise MAX_BATCHES (got={got}, n_hi={n_hi}, n_lo={n_lo})")

    Q = np.concatenate(Qs); S3 = np.concatenate(S3s); A = np.concatenate(As)
    hi, lo = Q > Q_HI, Q < Q_LO
    out = {"sigma": sigma, "n_splits": int(Q.size),
           "kappa_fleiss": float(fleiss(np.concatenate(kap_chunks))),
           "kappa_cohen": float(cohen(np.concatenate([a for a, _ in dyad_chunks]),
                                      np.concatenate([b for _, b in dyad_chunks])))}
    out["t1_r3"], out["se_t1_r3"], out["n_hi"] = safe_rate(~S3[hi])
    out["t2_r3"], out["se_t2_r3"], out["n_lo"] = safe_rate(S3[lo])
    out["t1_ae"], out["se_t1_ae"], _ = safe_rate(~A[hi])
    out["t2_ae"], out["se_t2_ae"], _ = safe_rate(A[lo])
    if C2s:
        C2 = np.concatenate(C2s)
        out["t1_c2"], out["se_t1_c2"], _ = safe_rate(~C2[hi])
        out["t2_c2"], out["se_t2_c2"], _ = safe_rate(C2[lo])
    return out


def agg(res, tag):
    """Aggregate rate (Type I + Type II, %) and its SE for regime tag."""
    a = (res[f"t1_{tag}"] + res[f"t2_{tag}"]) * 100
    se = np.hypot(res[f"se_t1_{tag}"], res[f"se_t2_{tag}"]) * 100
    return a, se


# ------------------------------- calibration --------------------------------
def calibrate():
    """Two-stage grid search; identical loops/seeds as the released baseline."""
    best, bloss = None, np.inf
    for c in np.arange(0.56, 0.70, 0.02):
        for a in (1.4, 1.8, 2.2, 2.6):
            for c3 in np.arange(0.44, 0.62, 0.02):
                for m3 in (1.0, 1.3, 1.6):
                    k, t1, t2, _, _ = evaluate(SIGMA_BASE, (c, a, c3, m3, 0.4, 0.2, 0.0), N_CAL, 7)
                    loss = ((k - .30) / .01)**2 + ((t1 - TARGETS_R3[0]) / .005)**2 \
                         + ((t2 - TARGETS_R3[1]) / .005)**2
                    if loss < bloss:
                        bloss, best = loss, (c, a, c3, m3)
    c, a, c3, m3 = best
    for _ in range(2):
        for cc in np.arange(c - .015, c + .016, .005):
            for c33 in np.arange(c3 - .015, c3 + .016, .005):
                for mm in np.arange(max(1.0, m3 - .2), m3 + .21, .1):
                    k, t1, t2, _, _ = evaluate(SIGMA_BASE, (cc, a, c33, mm, .4, .2, 0.), N_CAL, 7)
                    loss = ((k - .30) / .01)**2 + ((t1 - TARGETS_R3[0]) / .005)**2 \
                         + ((t2 - TARGETS_R3[1]) / .005)**2
                    if loss < bloss:
                        bloss, best = loss, (cc, a, c33, mm)
        c, a, c3, m3 = best
    k0, t1_0, t2_0, _, _ = evaluate(SIGMA_BASE, (c, a, c3, m3, .4, .2, 0.), N_CAL * 2, 8)
    print(f"[A] c={c:.3f} Beta({a},{a}) c3={c3:.3f} m3={m3:.2f} | kappa={k0:.3f} "
          f"TI_R3={t1_0*100:.1f} TII_R3={t2_0*100:.1f} "
          f"(calibration targets {TARGETS_R3[0]*100:.1f}/{TARGETS_R3[1]*100:.1f}; "
          f"achieved values are the ones reported) loss={bloss:.1f}")

    qw = qw_from_withdrawal(a)
    print(f"[A] Q_w={qw:.3f}  => k/V={qw**3/EXPM1**2:.4f}  "
          f"(k={qw**3/EXPM1**2*V:.3f}, V={V})")

    best, bloss = None, np.inf
    for ca in np.arange(0.30, 0.58, 0.01):
        for s0 in np.arange(0.10, 0.36, 0.01):
            _, _, _, ta, tb = evaluate(SIGMA_BASE, (c, a, c3, m3, ca, s0, qw), N_CAL, 11)
            loss = ((ta - TARGETS_AERE[0]) / .003)**2 + ((tb - TARGETS_AERE[1]) / .003)**2
            if loss < bloss:
                bloss, best = loss, (ca, s0)
    ca, s0 = best
    _, _, _, ta0, tb0 = evaluate(SIGMA_BASE, (c, a, c3, m3, ca, s0, qw), N_CAL * 2, 12)
    print(f"[B] c_a={ca:.2f} s0={s0:.2f} | TI_A={ta0*100:.1f} TII_A={tb0*100:.1f} "
          f"(calibration targets {TARGETS_AERE[0]*100:.1f}/{TARGETS_AERE[1]*100:.1f})")
    return (c, a, c3, m3, ca, s0, qw)


# ------------------------------ kappa mapping -------------------------------
def kappa_map(params):
    """Monotone (non-increasing) kappa(sigma) map for target inversion."""
    sigmas = np.geomspace(0.06, 1.4, 70)
    kmap = np.array([evaluate(s, params, 150_000, 3)[0] for s in sigmas])
    kmono = np.minimum.accumulate(kmap)          # enforce monotone decrease
    if np.max(np.abs(kmono - kmap)) > 0.01:
        print("  [warn] kappa(sigma) map required monotonicity repair > 0.01")
    return sigmas, kmono


def sigma_at(kt, sigmas, kmono):
    if not (kmono[-1] - 1e-9 <= kt <= kmono[0] + 1e-9):
        raise RuntimeError(f"kappa target {kt} outside achieved range "
                           f"[{kmono[-1]:.3f}, {kmono[0]:.3f}]")
    return float(np.interp(-kt, -kmono, sigmas))


# ---------------------------------- stages ----------------------------------
def stage_table4(params, sigmas, kmono):
    st = sigma_at(0.30, sigmas, kmono)
    r = evaluate_on_splits(st, params, SEED + 300, min_tail=N_TAIL_TABLE)
    print("\n=== TABLE 5 (baseline kappa = 0.30; split-conditional; +-1 MC SE) ===")
    print(f"sigma={st:.3f}  kappa_Fleiss={r['kappa_fleiss']:.3f}  "
          f"kappa_Cohen(dyad)={r['kappa_cohen']:.3f}  "
          f"n_splits={r['n_splits']:,}  n_hi={r['n_hi']:,}  n_lo={r['n_lo']:,}")
    t1r, t2r = r['t1_r3'] * 100, r['t2_r3'] * 100
    t1a, t2a = r['t1_ae'] * 100, r['t2_ae'] * 100
    print(f"Type I : R3 {t1r:5.1f} (+-{r['se_t1_r3']*100:.1f})   "
          f"AERE {t1a:5.1f} (+-{r['se_t1_ae']*100:.1f})   "
          f"gain {100*(t1a-t1r)/t1r:+.1f}%")
    print(f"Type II: R3 {t2r:5.1f} (+-{r['se_t2_r3']*100:.1f})   "
          f"AERE {t2a:5.1f} (+-{r['se_t2_ae']*100:.1f})   "
          f"gain {100*(t2a-t2r)/t2r:+.1f}%")
    ar, sr = agg(r, "r3"); aa, sa_ = agg(r, "ae")
    print(f"Aggreg.: R3 {ar:5.1f} (+-{sr:.1f})   AERE {aa:5.1f} (+-{sa_:.1f})")
    return r


def stage_figure3(params, sigmas, kmono):
    targets = np.round(np.arange(0.10, 0.701, 0.05), 2)
    rows = []
    print("\n=== FIGURE 3 (stratified: >=10,000 deadlocks and >=5,000 per tail) ===")
    print("kappa  sigma   n_splits   n_hi   n_lo   R3agg(+-SE)   AEREagg(+-SE)")
    for kt in targets:
        st = sigma_at(kt, sigmas, kmono)
        r = evaluate_on_splits(st, params, SEED + int(kt * 1000))
        ar, sr = agg(r, "r3"); aa, sa_ = agg(r, "ae")
        rows.append((kt, r, ar, sr, aa, sa_))
        print(f"{kt:.2f}  {st:.3f}  {r['n_splits']:8,d}  {r['n_hi']:5,d}  "
              f"{r['n_lo']:5,d}   {ar:5.1f} (+-{sr:.1f})   {aa:5.1f} (+-{sa_:.1f})")

    print("\n%% pgfplots -- Traditional R3 (y explicit error bars, +-1 SE):")
    print("    " + " ".join(f"({k:.2f}, {a:.1f}) +- (0, {s:.1f})"
                            for k, _, a, s, _, _ in rows))
    print("%% pgfplots -- AERE (y explicit error bars, +-1 SE):")
    print("    " + " ".join(f"({k:.2f}, {a:.1f}) +- (0, {s:.1f})"
                            for k, _, _, _, a, s in rows))

    with open("aere_figure3_data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kappa_target", "sigma", "kappa_fleiss", "kappa_cohen",
                    "n_splits", "n_hi", "n_lo",
                    "t1_r3", "se_t1_r3", "t2_r3", "se_t2_r3",
                    "t1_aere", "se_t1_aere", "t2_aere", "se_t2_aere"])
        for kt, r, *_ in rows:
            w.writerow([kt, f"{r['sigma']:.4f}", f"{r['kappa_fleiss']:.4f}",
                        f"{r['kappa_cohen']:.4f}", r['n_splits'], r['n_hi'], r['n_lo'],
                        f"{r['t1_r3']:.4f}", f"{r['se_t1_r3']:.4f}",
                        f"{r['t2_r3']:.4f}", f"{r['se_t2_r3']:.4f}",
                        f"{r['t1_ae']:.4f}", f"{r['se_t1_ae']:.4f}",
                        f"{r['t2_ae']:.4f}", f"{r['se_t2_ae']:.4f}"])
    print("[csv] aere_figure3_data.csv written")
    return rows


def stage_sensitivity(params, sigmas, kmono):
    c, a, c3, m3, ca, s0, qw = params
    sb = sigma_at(0.30, sigmas, kmono)
    s10 = sigma_at(0.10, sigmas, kmono)
    kw = dict(min_tail=N_TAIL_SENS)
    print("\n=== TABLE 6 SENSITIVITY SUITE (split-conditional, %, +-1 SE) ===")

    print("\n(a) Audit-degradation exponent alpha (gap = R3agg - AEREagg):")
    for kt, s in ((0.30, sb), (0.10, s10)):
        for al in (0.5, 1.0, 1.5):
            r = evaluate_on_splits(s, params, SEED + 91, alpha=al, **kw)
            ar, sr = agg(r, "r3"); aa, sa_ = agg(r, "ae")
            print(f"  kappa={kt:.2f} alpha={al}: R3 {ar:5.1f} (+-{sr:.1f})  "
                  f"AERE {aa:5.1f} (+-{sa_:.1f})  gap {ar-aa:5.1f} pp")

    print("\n(b) Tie-breaker profile (baseline kappa=0.30):")
    for mm, cc3, lbl in ((m3, c3, "calibrated (noisier + stricter)"),
                         (1.0, c3, "same noise as dyad, stricter"),
                         (1.0, c, "statistically identical to dyad")):
        p2 = (c, a, cc3, mm, ca, s0, qw)
        r = evaluate_on_splits(sb, p2, SEED + 92, **kw)
        ar, sr = agg(r, "r3"); aa, sa_ = agg(r, "ae")
        print(f"  m3={mm:.2f}, c3={cc3:.3f} [{lbl}]: R3 {ar:5.1f} (+-{sr:.1f})  "
              f"AERE {aa:5.1f} (+-{sa_:.1f})")

    print("\n(c) Capacity-matched cold-dyad tie-break (two fresh referees, unanimity):")
    r = evaluate_on_splits(sb, params, SEED + 93, cold_dyad=True, **kw)
    a2, s2 = agg(r, "c2"); ar, sr = agg(r, "r3"); aa, sa_ = agg(r, "ae")
    print(f"  TI {r['t1_c2']*100:5.1f} (+-{r['se_t1_c2']*100:.1f})  "
          f"TII {r['t2_c2']*100:5.1f} (+-{r['se_t2_c2']*100:.1f})  "
          f"agg {a2:5.1f} (+-{s2:.1f})   [vs R3 {ar:.1f}, AERE {aa:.1f}]")

    print("\n(d) Withdrawal fraction of sub-par work (baseline kappa=0.30):")
    for frac in (0.40, 0.65, 0.80):
        qw2 = qw_from_withdrawal(a, frac=frac)
        p2 = (c, a, c3, m3, ca, s0, qw2)
        r = evaluate_on_splits(sb, p2, SEED + 94, **kw)
        print(f"  frac={frac:.2f} (Q_w={qw2:.3f}, k={qw2**3/EXPM1**2*V:.2f}): "
              f"AERE TI {r['t1_ae']*100:5.1f} (+-{r['se_t1_ae']*100:.1f})  "
              f"TII {r['t2_ae']*100:5.1f} (+-{r['se_t2_ae']*100:.1f})")

    print("\n(e) Authorial self-assessment noise sigma_A (baseline kappa=0.30):")
    for sA in (0.0, 0.05, 0.10):
        r = evaluate_on_splits(sb, params, SEED + 95, author_sigma=sA, **kw)
        print(f"  sigma_A={sA:.2f}: AERE TI {r['t1_ae']*100:5.1f} "
              f"(+-{r['se_t1_ae']*100:.1f})  TII {r['t2_ae']*100:5.1f} "
              f"(+-{r['se_t2_ae']*100:.1f})")


def stage_experiment_IV(params, sigmas, kmono):
    """Type II under tie-breaker / proponent-auditor leniency delta."""
    sb = sigma_at(0.30, sigmas, kmono)
    print("\n=== EXPERIMENT IV -- TABLE 11 (leniency delta; Type II, %, +-1 SE) ===")
    rows = []
    for d in (0.0, 0.3, 0.5, 0.8):
        r = evaluate_on_splits(sb, params, SEED + 96 + int(d * 10),
                               leniency=d, min_tail=N_TAIL_SENS)
        t2r, t2a = r['t2_r3'] * 100, r['t2_ae'] * 100
        rows.append((d, t2r, r['se_t2_r3'] * 100, t2a, r['se_t2_ae'] * 100))
        print(f"  delta={d:.1f}: R3 {t2r:5.1f} (+-{r['se_t2_r3']*100:.1f})   "
              f"AERE {t2a:5.1f} (+-{r['se_t2_ae']*100:.1f})   "
              f"improvement {100*(t2a-t2r)/t2r:+.1f}%")
    print("%% pgfplots -- Exp IV R3:   "
          + " ".join(f"({d:.1f}, {a:.1f})" for d, a, _, _, _ in rows))
    print("%% pgfplots -- Exp IV AERE: "
          + " ".join(f"({d:.1f}, {a:.1f})" for d, _, _, a, _ in rows))


def main():
    params = calibrate()
    sigmas, kmono = kappa_map(params)
    stage_table4(params, sigmas, kmono)
    stage_figure3(params, sigmas, kmono)
    stage_sensitivity(params, sigmas, kmono)
    stage_experiment_IV(params, sigmas, kmono)
    c, a, c3, m3, ca, s0, qw = params
    print(f"\nCalibrated parameters: c={c:.3f}, Beta({a},{a}), c3={c3:.3f}, "
          f"m3={m3:.2f}, Q_w={qw:.3f}, c_a={ca:.2f}, s0={s0:.2f}, alpha={ALPHA}")


if __name__ == "__main__":
    main()
