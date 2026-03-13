import numpy as np
from scipy.sparse import dok_matrix
from scipy.linalg import expm
import mdptoolbox.mdp as mdp
import os
import itertools
from scipy.optimize import curve_fit
import multiprocessing as mp

# ---------------------------
# global-ish constants
# ---------------------------
N_pop = 10000
mutation_rate = 0.00005  # weak mutation Nmu=0.5
tau = 10000
dt = 1
L = 25
a = 1 / L
N_states = L**3

# for reproducibility on HPC
SEED = int(os.environ.get("SEED", "12345"))
np.random.seed(SEED)

# make grid once
states = [(a*(0.5+i), a*(0.5+j), a*(0.5+k))
          for k in range(L) for j in range(L) for i in range(L)]
actions = [0, 1, 2, 3]
num_genotypes = 4
genotypes = [format(i, '02b') for i in range(num_genotypes)]  # 4 -> 2 bits
t_step = int(tau / dt)

LAST_K = 100

_CALL_SS = np.random.SeedSequence(SEED)

# ---------------------------
# helpers that now take drug_lists
# ---------------------------
def get_f(action, drug_lists):
    return drug_lists[action]

def get_s(action, drug_lists):
    f = get_f(action, drug_lists)
    s1 = f[1] / f[0] - 1
    s2 = f[2] / f[0] - 1
    s3 = f[3] / f[0] - 1
    return s1, s2, s3

def unpack(u):
    x1 = u[0]
    x2 = u[1] * (1 - x1)
    x3 = u[2] * (1 - u[0]) * (1 - u[1])
    x0 = 1 - x1 - x2 - x3
    return x0, x1, x2, x3

def A1(u, action, drug_lists):
    s1, s2, s3 = get_s(action, drug_lists)
    return mutation_rate*((1-u[0])*(1-u[1])-2*u[0]) + u[0]*(1-u[0])*(s1 - u[1]*s2 - (1-u[1])*u[2]*s3)

def A2(u, action, drug_lists):
    s1, s2, s3 = get_s(action, drug_lists)
    return mutation_rate/(1-u[0])*((1-u[0])*(1-u[1])*(1+u[1]) - 2*(1-u[0])*u[1] - 2*u[0]*u[1]) + u[1]*(1-u[1])*(s2 - u[2]*s3)

def A3(u, action, drug_lists):
    s1, s2, s3 = get_s(action, drug_lists)
    return mutation_rate/((1-u[0])*(1-u[1]))*(u[0]+(1-u[0])*u[1])*(1-2*u[2]) + u[2]*(1-u[2])*s3

def D1(u):
    return u[0] * (1 - u[0]) / (2 * N_pop)

def D2(u):
    return u[1] * (1 - u[1]) / (2 * N_pop * (1 - u[0]))

def D3(u):
    return u[2] * (1 - u[2]) / (2 * N_pop * (1 - u[0]) * (1 - u[1]))


def build_transition_rate_matrix(action, states, drug_lists):
    Omega = dok_matrix((N_states, N_states), dtype=np.float64)

    for n in range(N_states):
        u = states[n]
        i = n % L
        j = (n // L) % L
        k = n // (L * L)

        if i < L - 1:
            m = n + 1
            rate = D1(u)/a**2 + A1(u, action, drug_lists)/(2*a)
            Omega[n, m] = rate
        if i > 0:
            m = n - 1
            rate = D1(u)/a**2 - A1(u, action, drug_lists)/(2*a)
            Omega[n, m] = rate
        if j < L - 1:
            m = n + L
            rate = D2(u)/a**2 + A2(u, action, drug_lists)/(2*a)
            Omega[n, m] = rate
        if j > 0:
            m = n - L
            rate = D2(u)/a**2 - A2(u, action, drug_lists)/(2*a)
            Omega[n, m] = rate
        if k < L - 1:
            m = n + L*L
            rate = D3(u)/a**2 + A3(u, action, drug_lists)/(2*a)
            Omega[n, m] = rate
        if k > 0:
            m = n - L*L
            rate = D3(u)/a**2 - A3(u, action, drug_lists)/(2*a)
            Omega[n, m] = rate

    for n in range(N_states):
        row_sum = Omega[n, :].sum()
        Omega[n, n] = -row_sum

    return Omega.tocsr()


def build_transition_matrix(Omega, dt):
    # convert to dense and exponentiate
    W = expm(Omega.toarray() * dt)
    W[W < 0] = 0.0
    W = W / W.sum(axis=1, keepdims=True)
    return W


def compute_reward(state, action, drug_lists):
    x0, x1, x2, x3 = unpack(state)
    f0, f1, f2, f3 = get_f(action, drug_lists)
    fitness = x0*f0 + x1*f1 + x2*f2 + x3*f3
    return -fitness


def build_W_and_R(actions, states, dt, drug_lists):
    P = []
    R = np.zeros((len(states), len(actions)), dtype=np.float64)
    for a in actions:
        Omega = build_transition_rate_matrix(a, states, drug_lists)
        W = build_transition_matrix(Omega, dt)
        P.append(W)
        for i, s in enumerate(states):
            R[i, a] = compute_reward(s, a, drug_lists)
    return P, R

# freq <-> index
def pack(freq):
    x0, x1, x2, x3 = freq
    u1 = x1
    den1 = 1 - u1
    u2 = 0 if den1 == 0 else x2 / den1
    den2 = den1 * (1 - u2)
    u3 = 0 if den2 == 0 else x3 / den2
    return u1, u2, u3

def freq_to_state_idx(freq):
    u1, u2, u3 = pack(freq)
    i = min(max(int(np.floor(u1 * L)), 0), L-1)
    j = min(max(int(np.floor(u2 * L)), 0), L-1)
    k = min(max(int(np.floor(u3 * L)), 0), L-1)
    return i + L*j + (L**2)*k

def _wf_worker(args):
    """
    args = (n_chunk, drug_lists, picker_kind, picker_data, seed_int)

    picker_kind:
      - "constant": picker_data = int action (0-3)
      - "mdp":      picker_data = policy array (len N_states)
      - "cycle":    picker_data = (d1, d2) where schedule is ABAB... (your current pattern)
    """
    (n_chunk, drug_lists, picker_kind, picker_data, seed_int) = args
    rng = np.random.default_rng(seed_int)

    # mutation matrix once per worker
    Q = np.zeros((num_genotypes, num_genotypes))
    for i in range(num_genotypes):
        for j in range(num_genotypes):
            if i != j:
                hamming_dist = sum(a != b for a, b in zip(genotypes[i], genotypes[j]))
                if hamming_dist == 1:
                    Q[i, j] = mutation_rate
    for i in range(num_genotypes):
        Q[i, i] = 1 - Q[i].sum()

    out = np.zeros((n_chunk, t_step + 1), dtype=float)

    for r in range(n_chunk):
        counts = np.array([N_pop] + [0] * (num_genotypes - 1), dtype=int)
        freq = counts / N_pop
        fit_traj = np.zeros(t_step + 1, dtype=float)

        # t=0 action
        if picker_kind == "constant":
            a0 = int(picker_data)
        elif picker_kind == "mdp":
            policy = picker_data
            a0 = int(policy[freq_to_state_idx(freq)])
        elif picker_kind == "cycle":
            d1, d2 = picker_data
            a0 = int(d2)  # gen=0 => d2 in your original schedule
        else:
            raise ValueError(f"Unknown picker_kind: {picker_kind}")

        f0 = np.array(get_f(a0, drug_lists), dtype=float)
        fit_traj[0] = freq.dot(f0)

        for gen in range(1, t_step + 1):
            if picker_kind == "constant":
                a = int(picker_data)
            elif picker_kind == "mdp":
                policy = picker_data
                a = int(policy[freq_to_state_idx(freq)])
            elif picker_kind == "cycle":
                d1, d2 = picker_data
                a = int(d1) if (gen % 2 == 1) else int(d2)
            else:
                raise ValueError(f"Unknown picker_kind: {picker_kind}")

            f_vec = np.array(get_f(a, drug_lists), dtype=float)
            w_bar = freq.dot(f_vec)
            freq_sel = (freq * f_vec) / w_bar
            freq_mut = Q.T @ freq_sel

            counts = rng.multinomial(N_pop, freq_mut)
            freq = counts / N_pop
            fit_traj[gen] = freq.dot(f_vec)

        out[r] = fit_traj

    return out

def run_wf(drug_lists, n_reps=1, picker_kind="constant", picker_data=0):
    n_jobs_env = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("N_JOBS")
    n_jobs = int(n_jobs_env) if n_jobs_env is not None else 1
    n_jobs = max(1, min(n_jobs, n_reps))

    if n_jobs == 1:
        ss = _CALL_SS.spawn(1)[0]
        seed_int = int(ss.generate_state(1, dtype=np.uint32)[0])
        return _wf_worker((n_reps, drug_lists, picker_kind, picker_data, seed_int))

    chunks = [n_reps // n_jobs] * n_jobs
    for i in range(n_reps % n_jobs):
        chunks[i] += 1

    ss = _CALL_SS.spawn(1)[0]
    child_seeds = ss.spawn(n_jobs)
    seed_ints = [int(s.generate_state(1, dtype=np.uint32)[0]) for s in child_seeds]

    args = [(chunks[i], drug_lists, picker_kind, picker_data, seed_ints[i]) for i in range(n_jobs)]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_jobs) as pool:
        parts = pool.map(_wf_worker, args)

    return np.vstack(parts)

# ---------------------------
# balanced landscape pools: exactly n_sets weak + n_sets strong
# ---------------------------
LANDSCAPE_MU = 1.0
LANDSCAPE_SIGMA = 0.00004
WEAK_THRESHOLD = 0.0001

def build_balanced_pools(target_per_class, mu, sigma, weak_threshold):
    weak_pool = []
    strong_pool = []
    draws = 0

    while (len(weak_pool) < target_per_class) or (len(strong_pool) < target_per_class):
        draws += 1
        fitness = np.random.normal(loc=mu, scale=sigma, size=(4, 4))
        f00 = fitness[:, [0]]
        selection = fitness / f00 - 1.0
        max_sel = float(np.max(np.abs(selection)))

        # Convert to your expected drug_lists format: {0:[f00,f01,f10,f11], ...}
        drug_lists = {d: [float(fitness[d, j]) for j in range(4)] for d in range(4)}

        if max_sel < weak_threshold:
            if len(weak_pool) < target_per_class:
                weak_pool.append(drug_lists)
        else:
            if len(strong_pool) < target_per_class:
                strong_pool.append(drug_lists)

    return weak_pool, strong_pool

# ---------------------------
# env helper
# ---------------------------
def get_env_minmax(drug_lists):
    """Return (global_min, global_max) across all drugs/genotypes for this env."""
    all_fits = np.concatenate([np.array(v) for v in drug_lists.values()])
    return all_fits.min(), all_fits.max()

# ---------------------------
# Check stationary distribution
# ---------------------------
def normalize_trajs(fit_trajs, gmin, den):
    """Normalize trajectories to (fit - gmin)/den. fit_trajs shape (n_reps, T)."""
    if den <= 0:
        return np.zeros_like(fit_trajs, dtype=float)
    return (fit_trajs - gmin) / den

def stationary_delta(trajs_norm, last=50, prev=50):
    """
    delta per replicate:
        mean(last 'last' gens) - mean(previous 'prev' gens)
    Using the END of the trajectory.
    trajs_norm shape (n_reps, T).
    Returns (mean_delta, se_delta, deltas_per_rep)
    """
    T = trajs_norm.shape[1]
    if last + prev > T:
        raise ValueError(f"Need last+prev <= T, got {last}+{prev} > {T}")

    last_block = trajs_norm[:, T-last:T]                 # last 50
    prev_block = trajs_norm[:, T-(last+prev):T-last]     # gens -100:-50 (50 gens)

    deltas = last_block.mean(axis=1) - prev_block.mean(axis=1)  # per replicate
    mean_delta = float(deltas.mean())
    se_delta = float(deltas.std(ddof=0) / np.sqrt(len(deltas)))
    return mean_delta, se_delta, deltas

def exp_saturation(t, A, B, C):
    # A(1 - B e^{-Ct})
    return A * (1.0 - B * np.exp(-C * t))

def fit_saturation_params(trajs_norm, dt=1.0):
    """
    Fit mean normalized trajectory (across reps) to A(1 - B e^{-Ct}).
    Returns (A,B,C) and a success flag.
    """
    y = trajs_norm.mean(axis=0)          # length T
    T = y.shape[0]
    t = np.arange(T, dtype=float) * dt

    # sensible initial guesses
    A0 = float(y[-1])
    if abs(A0) < 1e-12:
        A0 = float(np.clip(y.max(), 1e-6, 10.0))
    B0 = float(np.clip(1.0 - (y[0] / A0), 0.0, 2.0)) if abs(A0) > 1e-12 else 1.0
    C0 = 1e-3  # slow-ish relaxation default

    # bounds: keep it stable
    # A can be slightly outside [0,1] due to normalization + noise, so allow [-1, 2]
    # B in [0, 3], C in [1e-8, 10]
    bounds = ([-1.0, 0.0, 1e-8], [2.0, 3.0, 10.0])

    try:
        popt, _ = curve_fit(
            exp_saturation, t, y,
            p0=[A0, B0, C0],
            bounds=bounds,
            maxfev=20000
        )
        A, B, C = map(float, popt)
        return (A, B, C), True
    except Exception:
        return (np.nan, np.nan, np.nan), False
    
# ---------------------------
# main sweep
# ---------------------------
if __name__ == "__main__":
    n_sets = int(os.environ.get("N_SETS", "300"))
    n_reps = int(os.environ.get("N_REPS", "1000"))

    # build exactly 300 weak and 300 strong landscapes
    weak_pool, strong_pool = build_balanced_pools(
        target_per_class=n_sets,
        mu=LANDSCAPE_MU,
        sigma=LANDSCAPE_SIGMA,
        weak_threshold=WEAK_THRESHOLD
    )

    ratios_weak = []
    ratios_strong = []
    ratios_weak_se = []
    ratios_strong_se = []

    weak_mdp_mean_lastK = []
    weak_mdp_std_lastK  = []
    weak_cycle_mean_lastK = []
    weak_cycle_std_lastK  = []

    strong_mdp_mean_lastK = []
    strong_mdp_std_lastK  = []
    strong_cycle_mean_lastK = []
    strong_cycle_std_lastK  = []

    # --- stationarity metrics (normalized scale) ---
    delta_weak_mdp = []
    delta_weak_cycle = []
    delta_weak_mdp_se = []
    delta_weak_cycle_se = []

    delta_strong_mdp = []
    delta_strong_cycle = []
    delta_strong_mdp_se = []
    delta_strong_cycle_se = []

    # --- saturation fit params (A,B,C) on normalized mean trajectory ---
    ABC_weak_mdp = []
    ABC_weak_cycle = []
    ABC_strong_mdp = []
    ABC_strong_cycle = []

    ABC_weak_mdp_ok = []
    ABC_weak_cycle_ok = []
    ABC_strong_mdp_ok = []
    ABC_strong_cycle_ok = []

    weak_drug_tables = []
    strong_drug_tables = []

    # all 2-drug pairs (unordered)
    two_drug_pairs = list(itertools.combinations(actions, 2))

    for idx in range(n_sets):
        # ---------- WEAK ----------
        drug_lists = weak_pool[idx]
        weak_drug_tables.append(drug_lists)

        # min/max from ORIGINAL fitness
        gmin, gmax = get_env_minmax(drug_lists)
        den = gmax - gmin

        P, R = build_W_and_R(actions, states, dt, drug_lists)
        vi = mdp.ValueIteration(transitions=P, reward=R, discount=0.99,
                                epsilon=1e-4, max_iter=1000)
        vi.run()

        # --- 2-drug cycling baseline: ABAB... over generations for each pair ---
        pair_windows = []
        pair_full_trajs = []  # store full trajs for metrics
        for (d1, d2) in two_drug_pairs:
            fit_trajs = run_wf(drug_lists, n_reps=n_reps, picker_kind="cycle", picker_data=(d1, d2))
            pair_full_trajs.append(fit_trajs)
            pair_windows.append(fit_trajs[:, -LAST_K:])             # (n_reps, LAST_K)

        pair_means_raw = [w.mean() for w in pair_windows]
        best_pair_idx = int(np.argmin(pair_means_raw))
        best_pair_window = pair_windows[best_pair_idx]              # last K
        best_pair_fits_full = pair_full_trajs[best_pair_idx]        # full trajectory

        # --- MDP runs: last K gens x reps ---
        mdp_fits = run_wf(drug_lists, n_reps=n_reps, picker_kind="mdp", picker_data=vi.policy)
        mdp_window = mdp_fits[:, -LAST_K:]                         # (n_reps, LAST_K)

        # --- normalize per replicate & generation ---
        best_pair_window_n = (best_pair_window - gmin) / den
        mdp_window_n       = (mdp_window - gmin) / den

        # --- per-generation stats across replicates (after scaling) ---
        weak_cycle_mean_lastK.append(best_pair_window_n.mean(axis=0))
        weak_cycle_std_lastK.append(best_pair_window_n.std(axis=0, ddof=0))

        weak_mdp_mean_lastK.append(mdp_window_n.mean(axis=0))
        weak_mdp_std_lastK.append(mdp_window_n.std(axis=0, ddof=0))

        # -------- stationarity metrics (normalized) --------
        mdp_norm = normalize_trajs(mdp_fits, gmin, den)                 # (n_reps, t_step+1)
        cyc_norm = normalize_trajs(best_pair_fits_full, gmin, den)      # (n_reps, t_step+1)

        m_delta, m_se, _ = stationary_delta(mdp_norm, last=50, prev=50)
        c_delta, c_se, _ = stationary_delta(cyc_norm, last=50, prev=50)

        delta_weak_mdp.append(m_delta)
        delta_weak_mdp_se.append(m_se)
        delta_weak_cycle.append(c_delta)
        delta_weak_cycle_se.append(c_se)

        # -------- fit A(1 - B e^{-Ct}) to mean normalized trajectory --------
        (m_ABC, m_ok) = fit_saturation_params(mdp_norm, dt=dt)
        (c_ABC, c_ok) = fit_saturation_params(cyc_norm, dt=dt)

        ABC_weak_mdp.append(m_ABC)
        ABC_weak_cycle.append(c_ABC)
        ABC_weak_mdp_ok.append(m_ok)
        ABC_weak_cycle_ok.append(c_ok)

        # --- ratio of means on the *normalized* scale ---
        if den > 0:
            mdp_rep_mean = mdp_window.mean(axis=1)                 # (n_reps,)
            best_rep_mean = best_pair_window.mean(axis=1)          # (n_reps,)

            # normalize per replicate
            mdp_rep_mean_n = (mdp_rep_mean - gmin) / den
            best_rep_mean_n = (best_rep_mean - gmin) / den

            mean_mdp = mdp_rep_mean_n.mean()
            mean_best = best_rep_mean_n.mean()

            std_mdp = mdp_rep_mean_n.std(ddof=0)
            std_best = best_rep_mean_n.std(ddof=0)

            SE_mdp = std_mdp / np.sqrt(n_reps)
            SE_best = std_best / np.sqrt(n_reps)

            eps = 1e-12
            R_val = mean_mdp / max(mean_best, eps)

            # error propagation using SEs
            rel_m = (SE_mdp / max(mean_mdp, eps))**2
            rel_s = (SE_best / max(mean_best, eps))**2
            R_se = R_val * np.sqrt(rel_m + rel_s)

            ratios_weak.append(R_val)
            ratios_weak_se.append(R_se)
        else:
            ratios_weak.append(0.0)
            ratios_weak_se.append(0.0)

        # ---------- STRONG ----------
        drug_lists = strong_pool[idx]
        strong_drug_tables.append(drug_lists)

        gmin, gmax = get_env_minmax(drug_lists)
        den = gmax - gmin

        P, R = build_W_and_R(actions, states, dt, drug_lists)
        vi = mdp.ValueIteration(transitions=P, reward=R, discount=0.99,
                                epsilon=1e-4, max_iter=1000)
        vi.run()

        # 2-drug cycling baseline in strong regime, same ABAB pattern
        pair_windows = []
        pair_full_trajs = []  # store full trajs for metrics
        for (d1, d2) in two_drug_pairs:
            fit_trajs = run_wf(drug_lists, n_reps=n_reps, picker_kind="cycle", picker_data=(d1, d2))
            pair_full_trajs.append(fit_trajs)
            pair_windows.append(fit_trajs[:, -LAST_K:])             # (n_reps, LAST_K)

        pair_means_raw = [w.mean() for w in pair_windows]
        best_pair_idx = int(np.argmin(pair_means_raw))
        best_pair_window = pair_windows[best_pair_idx]              # last K
        best_pair_fits_full = pair_full_trajs[best_pair_idx]        # full trajectory

        # MDP runs — last K gens
        mdp_fits = run_wf(drug_lists, n_reps=n_reps, picker_kind="mdp", picker_data=vi.policy)
        mdp_window = mdp_fits[:, -LAST_K:]                         # (n_reps, LAST_K)

        # --- normalize per replicate & generation ---
        best_pair_window_n = (best_pair_window - gmin) / den
        mdp_window_n       = (mdp_window - gmin) / den

        # --- per-generation stats across replicates (after scaling) ---
        strong_cycle_mean_lastK.append(best_pair_window_n.mean(axis=0))
        strong_cycle_std_lastK.append(best_pair_window_n.std(axis=0, ddof=0))

        strong_mdp_mean_lastK.append(mdp_window_n.mean(axis=0))
        strong_mdp_std_lastK.append(mdp_window_n.std(axis=0, ddof=0))

        # -------- stationarity metrics (normalized) --------
        mdp_norm = normalize_trajs(mdp_fits, gmin, den)                 # (n_reps, t_step+1)
        cyc_norm = normalize_trajs(best_pair_fits_full, gmin, den)      # (n_reps, t_step+1)

        m_delta, m_se, _ = stationary_delta(mdp_norm, last=50, prev=50)
        c_delta, c_se, _ = stationary_delta(cyc_norm, last=50, prev=50)

        delta_strong_mdp.append(m_delta)
        delta_strong_mdp_se.append(m_se)
        delta_strong_cycle.append(c_delta)
        delta_strong_cycle_se.append(c_se)

        # -------- fit A(1 - B e^{-Ct}) to mean normalized trajectory --------
        (m_ABC, m_ok) = fit_saturation_params(mdp_norm, dt=dt)
        (c_ABC, c_ok) = fit_saturation_params(cyc_norm, dt=dt)

        ABC_strong_mdp.append(m_ABC)
        ABC_strong_cycle.append(c_ABC)
        ABC_strong_mdp_ok.append(m_ok)
        ABC_strong_cycle_ok.append(c_ok)

        # ----- ratio of means on the normalized scale -----
        if den > 0:
            mdp_rep_mean = mdp_window.mean(axis=1)
            best_rep_mean = best_pair_window.mean(axis=1)

            mdp_rep_mean_n = (mdp_rep_mean - gmin) / den
            best_rep_mean_n = (best_rep_mean - gmin) / den

            mean_mdp = mdp_rep_mean_n.mean()
            mean_best = best_rep_mean_n.mean()

            std_mdp = mdp_rep_mean_n.std(ddof=0)
            std_best = best_rep_mean_n.std(ddof=0)

            SE_mdp = std_mdp / np.sqrt(n_reps)
            SE_best = std_best / np.sqrt(n_reps)

            eps = 1e-12
            R_val = mean_mdp / max(mean_best, eps)

            rel_m = (SE_mdp / max(mean_mdp, eps))**2
            rel_s = (SE_best / max(mean_best, eps))**2
            R_se = R_val * np.sqrt(rel_m + rel_s)

            ratios_strong.append(R_val)
            ratios_strong_se.append(R_se)
        else:
            ratios_strong.append(0.0)
            ratios_strong_se.append(0.0)

        # checkpoint
        if (idx + 1) % 50 == 0:
            np.savez(f"mdp_sweep_weak_two_checkpoint_{idx+1}_new.npz",
                    ratios_weak=np.array(ratios_weak),
                    ratios_strong=np.array(ratios_strong),
                    ratios_weak_se=np.array(ratios_weak_se),
                    ratios_strong_se=np.array(ratios_strong_se),
                    weak_drug_tables=np.array(weak_drug_tables, dtype=object),
                    strong_drug_tables=np.array(strong_drug_tables, dtype=object),
                    weak_mdp_mean_lastK=np.array(weak_mdp_mean_lastK),
                    weak_mdp_std_lastK=np.array(weak_mdp_std_lastK),
                    weak_cycle_mean_lastK=np.array(weak_cycle_mean_lastK),
                    weak_cycle_std_lastK=np.array(weak_cycle_std_lastK),
                    strong_mdp_mean_lastK=np.array(strong_mdp_mean_lastK),
                    strong_mdp_std_lastK=np.array(strong_mdp_std_lastK),
                    strong_cycle_mean_lastK=np.array(strong_cycle_mean_lastK),
                    strong_cycle_std_lastK=np.array(strong_cycle_std_lastK),
                    delta_weak_mdp=np.array(delta_weak_mdp),
                    delta_weak_mdp_se=np.array(delta_weak_mdp_se),
                    delta_weak_cycle=np.array(delta_weak_cycle),
                    delta_weak_cycle_se=np.array(delta_weak_cycle_se),
                    delta_strong_mdp=np.array(delta_strong_mdp),
                    delta_strong_mdp_se=np.array(delta_strong_mdp_se),
                    delta_strong_cycle=np.array(delta_strong_cycle),
                    delta_strong_cycle_se=np.array(delta_strong_cycle_se),
                    ABC_weak_mdp=np.array(ABC_weak_mdp),
                    ABC_weak_cycle=np.array(ABC_weak_cycle),
                    ABC_strong_mdp=np.array(ABC_strong_mdp),
                    ABC_strong_cycle=np.array(ABC_strong_cycle),
                    ABC_weak_mdp_ok=np.array(ABC_weak_mdp_ok),
                    ABC_weak_cycle_ok=np.array(ABC_weak_cycle_ok),
                    ABC_strong_mdp_ok=np.array(ABC_strong_mdp_ok),
                    ABC_strong_cycle_ok=np.array(ABC_strong_cycle_ok))
            
    # final save
    np.savez("mdp_sweep_weak_mu_two_new.npz",
            ratios_weak=np.array(ratios_weak),
            ratios_strong=np.array(ratios_strong),
            ratios_weak_se=np.array(ratios_weak_se),
            ratios_strong_se=np.array(ratios_strong_se),
            weak_drug_tables=np.array(weak_drug_tables, dtype=object),
            strong_drug_tables=np.array(strong_drug_tables, dtype=object),
            # last-100-gen mean/std per landscape ---
            weak_mdp_mean_lastK=np.array(weak_mdp_mean_lastK),
            weak_mdp_std_lastK=np.array(weak_mdp_std_lastK),
            weak_cycle_mean_lastK=np.array(weak_cycle_mean_lastK),
            weak_cycle_std_lastK=np.array(weak_cycle_std_lastK),
            strong_mdp_mean_lastK=np.array(strong_mdp_mean_lastK),
            strong_mdp_std_lastK=np.array(strong_mdp_std_lastK),
            strong_cycle_mean_lastK=np.array(strong_cycle_mean_lastK),
            strong_cycle_std_lastK=np.array(strong_cycle_std_lastK),
            delta_weak_mdp=np.array(delta_weak_mdp),
            delta_weak_mdp_se=np.array(delta_weak_mdp_se),
            delta_weak_cycle=np.array(delta_weak_cycle),
            delta_weak_cycle_se=np.array(delta_weak_cycle_se),
            delta_strong_mdp=np.array(delta_strong_mdp),
            delta_strong_mdp_se=np.array(delta_strong_mdp_se),
            delta_strong_cycle=np.array(delta_strong_cycle),
            delta_strong_cycle_se=np.array(delta_strong_cycle_se),
            ABC_weak_mdp=np.array(ABC_weak_mdp),
            ABC_weak_cycle=np.array(ABC_weak_cycle),
            ABC_strong_mdp=np.array(ABC_strong_mdp),
            ABC_strong_cycle=np.array(ABC_strong_cycle),
            ABC_weak_mdp_ok=np.array(ABC_weak_mdp_ok),
            ABC_weak_cycle_ok=np.array(ABC_weak_cycle_ok),
            ABC_strong_mdp_ok=np.array(ABC_strong_mdp_ok),
            ABC_strong_cycle_ok=np.array(ABC_strong_cycle_ok),
            n_sets=n_sets,
            n_reps=n_reps)