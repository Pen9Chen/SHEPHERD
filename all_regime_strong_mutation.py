import numpy as np
from scipy.sparse import dok_matrix
from scipy.linalg import expm
import mdptoolbox.mdp as mdp
import os
import itertools

# ---------------------------
# global-ish constants
# ---------------------------
N_pop = 10000
mutation_rate = 0.001  # strong mutation Nmu=10
tau = 5000
dt = 1
L = 25
a = 1 / L
N_states = L**3

# for reproducibility on HPC
np.random.seed(42)

# make grid once
states = [(a*(0.5+i), a*(0.5+j), a*(0.5+k))
          for k in range(L) for j in range(L) for i in range(L)]
actions = [0, 1, 2, 3]
num_genotypes = 4
genotypes = [format(i, '02b') for i in range(num_genotypes)]  # 4 -> 2 bits
t_step = int(tau / dt)

LAST_K = 100

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

def run_wf(picker, drug_lists, n_reps=1, schedule=None):
    """
    Wright-Fisher simulation.

    picker(freq) is used when schedule is None (e.g. single-drug, MDP).
    If schedule is not None, we use a = schedule(gen, freq) instead,
    which allows time-dependent policies like 2-drug cycling.
    """
    all_fit = np.zeros((n_reps, t_step+1))
    # build mutation matrix once
    Q = np.zeros((num_genotypes, num_genotypes))
    for i in range(num_genotypes):
        for j in range(num_genotypes):
            if i != j:
                hamming_dist = sum(a != b for a, b in zip(genotypes[i], genotypes[j]))
                if hamming_dist == 1:
                    Q[i, j] = mutation_rate
    for i in range(num_genotypes):
        Q[i, i] = 1 - Q[i].sum()

    for r in range(n_reps):
        counts = np.array([N_pop] + [0]*(num_genotypes-1), int)
        freq   = counts / N_pop
        fit_traj = np.zeros(t_step+1)

        # t=0
        a0 = picker(freq) if schedule is None else schedule(0, freq)
        f0 = np.array(get_f(a0, drug_lists))
        fit_traj[0] = freq.dot(f0)

        for gen in range(1, t_step+1):
            if schedule is None:
                a = picker(freq)
            else:
                a = schedule(gen, freq)
            f_vec = np.array(get_f(a, drug_lists), float)
            w_bar = freq.dot(f_vec)
            freq_sel = (freq * f_vec) / w_bar
            freq_mut = Q.T @ freq_sel
            counts = np.random.multinomial(N_pop, freq_mut)
            freq = counts / N_pop
            fit_traj[gen] = freq.dot(f_vec)

        all_fit[r] = fit_traj
    return all_fit

# ---------------------------
# dominance check + samplers
# ---------------------------
def has_dominant_drug(drug_lists):
    """
    True if some drug is the best (i.e. lowest fitness) for ALL 4 genotypes.
    """
    M = np.array([drug_lists[d] for d in range(4)])  # shape (4,4)
    for d in range(4):
        ok_all = True
        for g in range(4):
            if M[d, g] > M[:, g].min() + 1e-9:
                ok_all = False
                break
        if ok_all:
            return True
    return False

def sample_nondominant_set_from_range(s_min, s_max, max_tries=50):
    for _ in range(max_tries):
        drug_lists = {}
        for d in range(4):
            s = np.random.uniform(s_min, s_max, size=4)
            drug_lists[d] = list(1.0 * (1 + s))
        if not has_dominant_drug(drug_lists):
            return drug_lists
    return drug_lists  # last one

# ---- weak: max Ns < 1 (use 0.999/N) ----
def sample_weak_set(base=1.0):
    s_max = 0.999 / N_pop     # N*s_max = 0.999 < 1
    s_min = -s_max
    return sample_nondominant_set_from_range(s_min, s_max)

# ---- strong: min Ns > 1 ----
def sample_strong_set(base=1.0):
    s_min = 1.001 / N_pop
    s_max = 0.05
    for _ in range(50):
        drug_lists = {}
        for d in range(4):
            mags = np.random.uniform(s_min, s_max, size=4)
            signs = np.random.choice([-1, 1], size=4)
            s = signs * mags
            drug_lists[d] = list(base * (1 + s))
        if not has_dominant_drug(drug_lists):
            return drug_lists
    return drug_lists

# ---------------------------
# env helper
# ---------------------------
def get_env_minmax(drug_lists):
    """Return (global_min, global_max) across all drugs/genotypes for this env."""
    all_fits = np.concatenate([np.array(v) for v in drug_lists.values()])
    return all_fits.min(), all_fits.max()

# ---------------------------
# main sweep
# ---------------------------
n_sets = int(os.environ.get("N_SETS", "500"))
n_reps = int(os.environ.get("N_REPS", "1000"))

ratios_weak = []
ratios_strong = []
ratios_weak_sd = []
ratios_strong_sd = []

weak_drug_tables = []
strong_drug_tables = []

# all 2-drug pairs (unordered)
two_drug_pairs = list(itertools.combinations(actions, 2))

for idx in range(n_sets):
    # ---------- WEAK ----------
    drug_lists = sample_weak_set()
    weak_drug_tables.append(drug_lists)
    
    # min/max from ORIGINAL fitness
    gmin, gmax = get_env_minmax(drug_lists)
    den = gmax - gmin
    
    P, R = build_W_and_R(actions, states, dt, drug_lists)
    vi = mdp.ValueIteration(transitions=P, reward=R, discount=0.99,
                            epsilon=1e-4, max_iter=1000)
    vi.run()
    
    def mdp_picker(freq, policy=vi.policy):
        idxs = freq_to_state_idx(freq)
        return policy[idxs]
    
    # --- 2-drug cycling baseline: ABAB... over generations for each pair ---
    pair_windows = []  # list of arrays, each (n_reps, LAST_K)
    for (d1, d2) in two_drug_pairs:
        def cycle_schedule(gen, freq, d1=d1, d2=d2):
            # alternate every generation: d1, d2, d1, d2, ...
            return d1 if (gen % 2 == 1) else d2
        
        # picker is unused when schedule is provided; pass dummy
        fit_trajs = run_wf(lambda freq: d1, drug_lists, n_reps=n_reps,
                           schedule=cycle_schedule)           # (n_reps, t_step+1)
        window = fit_trajs[:, -LAST_K:]                        # (n_reps, LAST_K)
        pair_windows.append(window)
    
    # choose best 2-drug cycling pair by mean fitness over last K gens x reps
    pair_means_raw = [w.mean() for w in pair_windows]          # scalar per pair
    best_pair_idx = int(np.argmin(pair_means_raw))
    best_pair_window = pair_windows[best_pair_idx]             # (n_reps, LAST_K)
    
    # --- MDP runs: last K gens x reps ---
    mdp_fits = run_wf(mdp_picker, drug_lists, n_reps=n_reps)
    mdp_window = mdp_fits[:, -LAST_K:]                         # (n_reps, LAST_K)
    
    # Flatten over both replicates *and* last K gens for mean/SD
    mdp_vals = mdp_window.reshape(-1)                          # (n_reps * LAST_K,)
    base_vals = best_pair_window.reshape(-1)                   # (n_reps * LAST_K,)
    
    # --- ratio of means on the *normalized* scale ---
    if den > 0:
        # means & SDs in original scale
        m_mean = mdp_vals.mean()
        m_sd   = mdp_vals.std(ddof=1)
        s_mean = base_vals.mean()
        s_sd   = base_vals.std(ddof=1)
    
        # convert to normalized scale: y = (x - gmin)/den
        m_mean_n = (m_mean - gmin) / den
        s_mean_n = (s_mean - gmin) / den
        m_sd_n   = m_sd / den
        s_sd_n   = s_sd / den
    
        eps = 1e-12  # guard for zeros
        # ratio of means
        R_val = m_mean_n / max(s_mean_n, eps)
    
        # error propagation for division, using SD from both reps and last K gens
        rel_m = (m_sd_n / max(m_mean_n, eps))**2
        rel_s = (s_sd_n / max(s_mean_n, eps))**2
        R_sd  = R_val * np.sqrt(rel_m + rel_s)
    
        ratios_weak.append(R_val)
        ratios_weak_sd.append(R_sd)
    else:
        # degenerate environment: everything identical -> normalized values are all 0
        # define neutral ratio and zero SD to keep arrays well-defined
        ratios_weak.append(1.0)
        ratios_weak_sd.append(0.0)


    # ---------- STRONG ----------
    drug_lists = sample_strong_set()
    strong_drug_tables.append(drug_lists)
    gmin, gmax = get_env_minmax(drug_lists)
    den = gmax - gmin
    
    P, R = build_W_and_R(actions, states, dt, drug_lists)
    vi = mdp.ValueIteration(transitions=P, reward=R, discount=0.99,
                            epsilon=1e-4, max_iter=1000)
    vi.run()
    
    def mdp_picker(freq, policy=vi.policy):
        idxs = freq_to_state_idx(freq)
        return policy[idxs]
    
    # 2-drug cycling baseline in strong regime, same ABAB pattern
    pair_windows = []
    for (d1, d2) in two_drug_pairs:
        def cycle_schedule(gen, freq, d1=d1, d2=d2):
            return d1 if (gen % 2 == 1) else d2
        
        fit_trajs = run_wf(lambda freq: d1, drug_lists, n_reps=n_reps,
                           schedule=cycle_schedule)           # (n_reps, t_step+1)
        window = fit_trajs[:, -LAST_K:]                        # (n_reps, LAST_K)
        pair_windows.append(window)
    
    pair_means_raw = [w.mean() for w in pair_windows]
    best_pair_idx = int(np.argmin(pair_means_raw))
    best_pair_window = pair_windows[best_pair_idx]             # (n_reps, LAST_K)
    
    # MDP runs — last K gens
    mdp_fits = run_wf(mdp_picker, drug_lists, n_reps=n_reps)
    mdp_window = mdp_fits[:, -LAST_K:]                         # (n_reps, LAST_K)
    
    mdp_vals = mdp_window.reshape(-1)
    base_vals = best_pair_window.reshape(-1)
    
    # ----- ratio of means on the normalized scale -----
    if den > 0:
        # means & SDs in original scale
        m_mean = mdp_vals.mean()
        m_sd   = mdp_vals.std(ddof=1)
        s_mean = base_vals.mean()
        s_sd   = base_vals.std(ddof=1)
    
        # normalize: y = (x - gmin)/den
        m_mean_n = (m_mean - gmin) / den
        s_mean_n = (s_mean - gmin) / den
        m_sd_n   = m_sd / den
        s_sd_n   = s_sd / den
    
        eps = 1e-12
        R_val = m_mean_n / max(s_mean_n, eps)
    
        # error propagation for division with SD from reps + last K gens
        rel_m = (m_sd_n / max(m_mean_n, eps))**2
        rel_s = (s_sd_n / max(s_mean_n, eps))**2
        R_sd  = R_val * np.sqrt(rel_m + rel_s)
    
        ratios_strong.append(R_val)
        ratios_strong_sd.append(R_sd)
    else:
        # degenerate environment: all identical -> normalized values are 0
        ratios_strong.append(1.0)
        ratios_strong_sd.append(0.0)

    # checkpoint
    if (idx + 1) % 50 == 0:
        np.savez(f"mdp_sweep_strong_two_checkpoint_{idx+1}.npz",
                 ratios_weak=np.array(ratios_weak),
                 ratios_strong=np.array(ratios_strong),
                 ratios_weak_sd=np.array(ratios_weak_sd),
                 ratios_strong_sd=np.array(ratios_strong_sd))

# final save
np.savez("mdp_sweep_strong_mu_two.npz",
         ratios_weak=np.array(ratios_weak),
         ratios_strong=np.array(ratios_strong),
         ratios_weak_sd=np.array(ratios_weak_sd),
         ratios_strong_sd=np.array(ratios_strong_sd),
         weak_drug_tables=np.array(weak_drug_tables, dtype=object),
         strong_drug_tables=np.array(strong_drug_tables, dtype=object),
         n_sets=n_sets,
         n_reps=n_reps)