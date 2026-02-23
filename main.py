# app.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy.signal import detrend, butter, filtfilt, find_peaks
from scipy.interpolate import interp1d

# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(page_title="TUG Markov + End (retro limitado) + Eventos G1/G2", layout="wide")
st.title("📱 TUG — Markov + FIM retro (limitado ao intervalo movimento→baseline final) + eventos G1/G2")

st.markdown(
    """
### Pipeline
1) detrend → interpolação p/ **100 Hz** → low-pass (15 Hz) → **norma** do giroscópio  
2) **k-means 1D** na norma (K estados)  
3) Baselines adaptativos (início e fim)  
4) Dois mapas de estados:
   - `states_start`: baseline inicial vira **0**
   - `states_end`: baseline final vira **0**
5) **Start** por Markov+LL (queda persistente após baseline inicial)  
6) **End (recomendado)**:  
   - acha `rest0_start`: começo do repouso final persistente (`states_end==0`)  
   - procura o **último movimento forte** (`states_end>=Δ`) **somente** no intervalo:
     \n\n  **[start+offset, rest0_start)**  
7) G1/G2: picos na norma entre (start+offset) e end; componentes delimitados por retorno ao estado 1.
"""
)

# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.header("⚙️ Parâmetros")

    fs = st.number_input("Frequência alvo (Hz)", min_value=10.0, max_value=500.0, value=100.0, step=10.0)
    lowpass_hz = st.number_input("Low-pass (Hz)", min_value=1.0, max_value=80.0, value=15.0, step=1.0)

    st.divider()
    k_states = st.slider("K (k-means na norma)", min_value=3, max_value=12, value=7, step=1)

    st.divider()
    st.subheader("Baselines adaptativos")
    win_s = st.number_input("Duração janela baseline (s)", min_value=0.5, max_value=6.0, value=2.0, step=0.5)
    search_first_s = st.number_input("Buscar baseline inicial nos primeiros (s)", min_value=2.0, max_value=40.0, value=10.0, step=1.0)
    guard_start_s = st.number_input("Guarda início (s)", min_value=0.0, max_value=5.0, value=0.5, step=0.1)
    search_last_s = st.number_input("Buscar baseline final nos últimos (s)", min_value=2.0, max_value=40.0, value=10.0, step=1.0)
    guard_end_s = st.number_input("Guarda final (s)", min_value=0.0, max_value=5.0, value=0.5, step=0.1)
    step_s = st.number_input("Passo varredura baseline (s)", min_value=0.01, max_value=0.50, value=0.05, step=0.01)

    st.divider()
    st.subheader("Start (Markov + LL)")
    W_s = st.number_input("Janela LL W (s)", min_value=0.05, max_value=1.50, value=0.20, step=0.05)
    k_sigma_start = st.number_input("kσ início (thr = μ − kσ·σ)", min_value=0.5, max_value=10.0, value=3.0, step=0.5)

    st.subheader("R-grid (persistência)")
    r_min_s = st.number_input("R mínimo (s)", min_value=0.01, max_value=1.00, value=0.05, step=0.01)
    r_max_s = st.number_input("R máximo (s)", min_value=0.01, max_value=1.50, value=0.15, step=0.01)
    r_step_s = st.number_input("Passo R (s)", min_value=0.01, max_value=0.50, value=0.01, step=0.01)

    st.divider()
    st.subheader("End (retro limitado ao intervalo movimento→baseline final)")
    delta_states = st.number_input("Δ estados acima do repouso final", min_value=1, max_value=10, value=2, step=1)
    R_rest_s = st.number_input("Persistência repouso final (s)", min_value=0.02, max_value=2.0, value=0.15, step=0.01)
    R_move_s = st.number_input("Persistência movimento forte (s)", min_value=0.02, max_value=2.0, value=0.10, step=0.01)
    lookback_cap_s = st.number_input("Cap máximo de busca antes do repouso final (s)", min_value=1.0, max_value=60.0, value=30.0, step=1.0)

    st.divider()
    st.subheader("Eventos G1/G2")
    peak_search_offset_s = st.number_input("Offset (s): buscar picos em [start+offset, end]", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
    peak_prom = st.number_input("Prominência mínima (norma)", min_value=0.0, max_value=10.0, value=0.20, step=0.05)
    peak_min_dist_s = st.number_input("Distância mínima entre picos (s)", min_value=0.0, max_value=5.0, value=0.50, step=0.10)

    st.subheader("Componente por retorno a um estado")
    return_state = st.number_input("Estado de retorno (padrão=1)", min_value=0, max_value=50, value=1, step=1)
    run_state_s = st.number_input("Run do estado retorno (s)", min_value=0.01, max_value=1.00, value=0.10, step=0.01)

    st.divider()
    show_debug = st.checkbox("Mostrar debug", value=False)

uploads = st.file_uploader(
    "📂 Envie um ou mais arquivos .txt (separados por ';': tempo(ms); gx; gy; gz)",
    type=["txt"],
    accept_multiple_files=True
)

# =========================================================
# Helpers
# =========================================================
RNG = np.random.default_rng(42)

def read_gyro_txt_bytes(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes), sep=";", engine="python")
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    if df.shape[1] < 4:
        raise ValueError("Esperado: tempo(ms) + gx + gy + gz (>=4 colunas).")
    return df

def preprocess_to_norm(df: pd.DataFrame, fs: float, lowpass_hz: float):
    t_ms = df.iloc[:, 0].to_numpy(float)
    x = df.iloc[:, 1].to_numpy(float)
    y = df.iloc[:, 2].to_numpy(float)
    z = df.iloc[:, 3].to_numpy(float)

    m = np.isfinite(t_ms) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    t_ms, x, y, z = t_ms[m], x[m], y[m], z[m]

    t = t_ms / 1000.0
    order = np.argsort(t)
    t, x, y, z = t[order], x[order], y[order], z[order]

    _, idx = np.unique(t, return_index=True)
    t, x, y, z = t[idx], x[idx], y[idx], z[idx]

    t_uniform = np.arange(t[0], t[-1], 1.0 / fs)

    fx = interp1d(t, x, kind="linear", fill_value="extrapolate")
    fy = interp1d(t, y, kind="linear", fill_value="extrapolate")
    fz = interp1d(t, z, kind="linear", fill_value="extrapolate")
    x_i, y_i, z_i = fx(t_uniform), fy(t_uniform), fz(t_uniform)

    x_i = detrend(x_i); y_i = detrend(y_i); z_i = detrend(z_i)

    if lowpass_hz >= (fs / 2.0):
        raise ValueError("Low-pass deve ser menor que fs/2.")
    b, a = butter(4, lowpass_hz / (fs / 2.0), btype="low")
    x_f = filtfilt(b, a, x_i)
    y_f = filtfilt(b, a, y_i)
    z_f = filtfilt(b, a, z_i)

    norm = np.sqrt(x_f*x_f + y_f*y_f + z_f*z_f)
    return t_uniform, norm

def kmeans_1d(x: np.ndarray, k: int, max_iter: int = 40, tol: float = 1e-6):
    qs = np.linspace(0.0, 1.0, k + 2)[1:-1]
    centers = np.quantile(x, qs).astype(float)
    for _ in range(max_iter):
        d = np.abs(x[:, None] - centers[None, :])
        labels = np.argmin(d, axis=1)
        new_centers = centers.copy()
        for j in range(k):
            mask = labels == j
            new_centers[j] = float(np.mean(x[mask])) if np.any(mask) else float(x[RNG.integers(0, len(x))])
        shift = np.max(np.abs(new_centers - centers))
        centers = new_centers
        if shift < tol:
            break
    order = np.argsort(centers)
    centers = centers[order]
    inv = np.empty_like(order)
    inv[order] = np.arange(k)
    labels = inv[labels]
    return labels.astype(int), centers

def pick_quiet_window(norm: np.ndarray, fs: float, win_s: float, start_s: float, end_s: float, step_s: float):
    n = len(norm)
    Wn = int(round(win_s * fs))
    step = max(1, int(round(step_s * fs)))

    i_start = int(round(start_s * fs))
    i_end = int(round(end_s * fs))
    i_start = max(0, min(n - 1, i_start))
    i_end = max(0, min(n, i_end))

    if i_end - i_start < Wn:
        i0 = max(0, min(n - Wn, i_start))
        i1 = min(n, i0 + Wn)
        return i0, i1, float(np.var(norm[i0:i1]))

    best_i0, best_i1, best_score = None, None, np.inf
    for i0 in range(i_start, i_end - Wn + 1, step):
        i1 = i0 + Wn
        x = norm[i0:i1]
        if not np.all(np.isfinite(x)):
            continue
        score = float(np.var(x))
        if score < best_score:
            best_i0, best_i1, best_score = i0, i1, score

    if best_i0 is None:
        i0 = max(0, n - Wn)
        i1 = n
        return i0, i1, float(np.var(norm[i0:i1]))
    return best_i0, best_i1, best_score

def relabel_baseline_as_zero(labels: np.ndarray, i0: int, i1: int):
    base = labels[i0:i1]
    u, c = np.unique(base, return_counts=True)
    base_label = int(u[np.argmax(c)])

    rel = labels.copy()
    rel[labels == base_label] = 0

    new_lab = 1
    for lab in np.unique(labels):
        lab = int(lab)
        if lab != base_label:
            rel[labels == lab] = new_lab
            new_lab += 1
    return rel.astype(int), base_label

def transition_matrix(seq: np.ndarray, n_states: int, eps: float = 1e-10):
    counts = np.zeros((n_states, n_states), float)
    np.add.at(counts, (seq[:-1], seq[1:]), 1.0)
    rs = counts.sum(axis=1, keepdims=True)
    A = np.divide(counts, rs, out=np.zeros_like(counts), where=rs > 0)
    A[A == 0] = eps
    return A

def sliding_ll(seq: np.ndarray, A: np.ndarray, W: int):
    lp = np.log(A[seq[:-1], seq[1:]])
    cs = np.concatenate([[0.0], np.cumsum(lp)])
    ll = np.full(seq.shape[0], np.nan, float)
    idx = np.arange(W, seq.shape[0] - 1)
    ll[idx] = cs[idx] - cs[idx - W]
    return ll

def first_persistent(ll: np.ndarray, start: int, thr: float, R: int):
    for i in range(start, len(ll) - R):
        w = ll[i:i + R]
        if np.all(np.isfinite(w)) and np.all(w < thr):
            return i
    return None

def make_R_list_samples(fs: float, r_min_s: float, r_max_s: float, r_step_s: float):
    r_vals = np.arange(r_min_s, r_max_s + 1e-12, r_step_s)
    R_list = [max(1, int(round(fs * r))) for r in r_vals]
    return sorted(list(set(R_list)))

def detect_start_markov_grid(states: np.ndarray, i0_b: int, i1_b: int, W: int, R_list: list[int], k_sigma: float):
    n_states = int(states.max() + 1)
    A0 = transition_matrix(states[i0_b:i1_b], n_states=n_states)
    ll = sliding_ll(states, A0, W=W)

    ref = ll[i0_b:i1_b]
    ref = ref[np.isfinite(ref)]
    if len(ref) < 5:
        return None, ll, (np.nan, np.nan, np.nan)

    mu, sd = float(np.mean(ref)), float(np.std(ref))
    thr = mu - k_sigma * sd

    candidates = []
    for R in R_list:
        start_search = max(i1_b + W, 0)
        idx = first_persistent(ll, start=start_search, thr=thr, R=R)
        if idx is not None and idx >= i1_b:
            candidates.append(idx)

    if not candidates:
        return None, ll, (mu, sd, thr)

    return int(min(candidates)), ll, (mu, sd, thr)

# ---------- END retro (limitado ao intervalo movimento→baseline final) ----------
def find_first_run_eq_from_end(states: np.ndarray, run_len: int, value: int):
    n = len(states)
    if n < run_len:
        return None
    for i0 in range(n - run_len, -1, -1):
        if np.all(states[i0:i0 + run_len] == value):
            return i0
    return None

def find_last_run_ge_in_window(states: np.ndarray, start_limit: int, end_exclusive: int, run_len: int, thr_state: int):
    """
    Procura (de trás pra frente) o ÚLTIMO run onde states >= thr_state,
    mas SOMENTE no intervalo [start_limit, end_exclusive).
    Retorna o índice FINAL do run.
    """
    start_limit = max(0, int(start_limit))
    end_exclusive = min(len(states), int(end_exclusive))
    if end_exclusive - start_limit < run_len:
        return None

    for i0 in range(end_exclusive - run_len, start_limit - 1, -1):
        if np.all(states[i0:i0 + run_len] >= thr_state):
            return i0 + run_len - 1
    return None

def detect_end_retro_strong_limited(states_end: np.ndarray, fs: float, delta_states: int,
                                    R_rest_s: float, R_move_s: float,
                                    min_search_i: int, lookback_cap_s: float):
    """
    1) acha rest0_start: início do repouso final persistente (run de 0 no final)
    2) define janela de busca do fim: [max(min_search_i, rest0_start-cap), rest0_start)
    3) acha último run de movimento forte (>= delta_states) dentro dessa janela
    4) end = final desse run
    """
    R_rest = max(1, int(round(R_rest_s * fs)))
    R_move = max(1, int(round(R_move_s * fs)))
    thr_state = int(delta_states)  # como repouso final=0 no states_end

    rest0_start = find_first_run_eq_from_end(states_end, run_len=R_rest, value=0)
    if rest0_start is None:
        return None, None, None, {"reason": "No final rest (0-run) found", "R_rest": R_rest}

    cap = max(1, int(round(lookback_cap_s * fs)))
    start_limit = max(int(min_search_i), rest0_start - cap)
    end_exclusive = rest0_start  # não entra na baseline final

    last_strong_end = find_last_run_ge_in_window(
        states_end,
        start_limit=start_limit,
        end_exclusive=end_exclusive,
        run_len=R_move,
        thr_state=thr_state
    )
    if last_strong_end is None:
        return None, rest0_start, None, {
            "reason": "No strong-movement run in [movement, baseline_final)",
            "start_limit": int(start_limit),
            "end_exclusive": int(end_exclusive),
            "R_move": int(R_move),
            "thr_state": int(thr_state),
        }

    return int(last_strong_end), rest0_start, int(last_strong_end), {
        "R_rest": int(R_rest),
        "R_move": int(R_move),
        "thr_state": int(thr_state),
        "start_limit": int(start_limit),
        "end_exclusive": int(end_exclusive),
        "rest0_start": int(rest0_start),
    }

# =========================================================
# G1/G2 + components
# =========================================================
def find_two_peaks(norm: np.ndarray, i_start: int, i_end: int, fs: float, prom: float, min_dist_s: float):
    if i_start is None or i_end is None:
        return None, None
    i_start = int(max(0, i_start))
    i_end = int(min(len(norm) - 1, i_end))
    if i_end - i_start < 5:
        return None, None

    seg = norm[i_start:i_end + 1]
    dist = max(1, int(round(min_dist_s * fs)))
    peaks, _ = find_peaks(seg, prominence=prom, distance=dist)
    if len(peaks) < 2:
        return None, None

    peaks_global = np.sort(peaks + i_start)
    return int(peaks_global[0]), int(peaks_global[-1])

def component_bounds_from_runs(states: np.ndarray, peak_i: int, i_start: int, i_end: int, run_len: int, return_state: int):
    r_end = None
    for j in range(min(peak_i - 1, i_end), i_start + run_len - 2, -1):
        if np.all(states[j - run_len + 1: j + 1] == return_state):
            r_end = j
            break

    r_start = None
    for j in range(max(peak_i + 1, i_start), i_end - run_len + 2):
        if np.all(states[j: j + run_len] == return_state):
            r_start = j
            break

    comp_start = (r_end + 1) if r_end is not None else i_start
    comp_end = (r_start - 1) if r_start is not None else i_end

    comp_start = int(max(i_start, min(peak_i, comp_start)))
    comp_end = int(min(i_end, max(peak_i, comp_end)))
    return comp_start, comp_end

# =========================================================
# Run
# =========================================================
if not uploads:
    st.info("Envie ao menos 1 arquivo .txt para rodar a análise.")
    st.stop()

W = max(1, int(round(W_s * fs)))
R_list = make_R_list_samples(fs, r_min_s, r_max_s, r_step_s)
run_len = max(1, int(round(run_state_s * fs)))
return_state = int(return_state)
peak_offset_samples = int(round(peak_search_offset_s * fs))

results = []
cache = {}

for up in uploads:
    name = up.name
    try:
        df = read_gyro_txt_bytes(up.getvalue())
        t, norm = preprocess_to_norm(df, fs=fs, lowpass_hz=lowpass_hz)

        labels, _ = kmeans_1d(norm, k=k_states)

        # Baseline inicial
        i0_b, i1_b, _ = pick_quiet_window(
            norm, fs=fs, win_s=win_s,
            start_s=guard_start_s, end_s=search_first_s, step_s=step_s
        )

        # Baseline final (últimos search_last_s, exclui guard_end_s)
        total_s = float(t[-1] - t[0])
        end_region_start_s = max(0.0, total_s - search_last_s)
        end_region_end_s = max(0.0, total_s - guard_end_s)
        i0_f, i1_f, _ = pick_quiet_window(
            norm, fs=fs, win_s=win_s,
            start_s=end_region_start_s, end_s=end_region_end_s, step_s=step_s
        )

        # Dois mapas
        states_start, base_label_start = relabel_baseline_as_zero(labels, i0=i0_b, i1=i1_b)
        states_end, base_label_end = relabel_baseline_as_zero(labels, i0=i0_f, i1=i1_f)

        # Start
        start_i, ll_start, (_, _, thr_s) = detect_start_markov_grid(
            states_start, i0_b=i0_b, i1_b=i1_b, W=W, R_list=R_list, k_sigma=k_sigma_start
        )

        # Janela de movimento p/ end e picos: start+offset
        min_search_i = None
        if start_i is not None:
            min_search_i = min(len(norm) - 1, start_i + peak_offset_samples)

        # End retro limitado: procura somente em [min_search_i, rest0_start)
        end_i = None
        rest0_start = None
        end_candidate = None
        debug_end = {}

        if min_search_i is not None:
            end_i, rest0_start, end_candidate, debug_end = detect_end_retro_strong_limited(
                states_end=states_end,
                fs=fs,
                delta_states=int(delta_states),
                R_rest_s=float(R_rest_s),
                R_move_s=float(R_move_s),
                min_search_i=int(min_search_i),
                lookback_cap_s=float(lookback_cap_s),
            )

        # Picos
        g1_i = g2_i = None
        if min_search_i is not None and end_i is not None:
            g1_i, g2_i = find_two_peaks(
                norm=norm, i_start=min_search_i, i_end=end_i, fs=fs,
                prom=peak_prom, min_dist_s=peak_min_dist_s
            )

        # Componentes por retorno ao estado 1 (states_start)
        g1_cs = g1_ce = g2_cs = g2_ce = None
        if g1_i is not None and start_i is not None and end_i is not None:
            g1_cs, g1_ce = component_bounds_from_runs(
                states_start, peak_i=g1_i, i_start=start_i, i_end=end_i,
                run_len=run_len, return_state=return_state
            )
        if g2_i is not None and start_i is not None and end_i is not None:
            g2_cs, g2_ce = component_bounds_from_runs(
                states_start, peak_i=g2_i, i_start=start_i, i_end=end_i,
                run_len=run_len, return_state=return_state
            )

        def safe_time(idx):
            if idx is None or idx < 0 or idx >= len(t):
                return np.nan
            return float(t[idx])

        start_t = safe_time(start_i)
        end_t = safe_time(end_i)
        dur = (end_t - start_t) if np.isfinite(start_t) and np.isfinite(end_t) else np.nan

        results.append({
            "file": name,
            "start_s": start_t,
            "end_s": end_t,
            "duration_s": dur,
            "min_search_s (start+offset)": safe_time(min_search_i),
            "rest0_start_s": safe_time(rest0_start),
            "end_candidate_s": safe_time(end_candidate),
            "base_label_start(kmeans)": int(base_label_start),
            "base_label_end(kmeans)": int(base_label_end),
            "delta_states": int(delta_states),
            "R_rest_s": float(R_rest_s),
            "R_move_s": float(R_move_s),
            "thr_start": float(thr_s) if np.isfinite(thr_s) else np.nan,
            "G1_s": safe_time(g1_i),
            "G2_s": safe_time(g2_i),
            "debug_end": "" if not show_debug else str(debug_end),
        })

        cache[name] = dict(
            t=t, norm=norm,
            i0_b=i0_b, i1_b=i1_b, i0_f=i0_f, i1_f=i1_f,
            start_i=start_i, end_i=end_i,
            rest0_start=rest0_start,
            min_search_i=min_search_i,
            g1_i=g1_i, g2_i=g2_i,
            g1_cs=g1_cs, g1_ce=g1_ce, g2_cs=g2_cs, g2_ce=g2_ce,
            ll_start=ll_start,
        )

    except Exception as e:
        results.append({"file": name, "error": str(e)})

res_df = pd.DataFrame(results)

st.subheader("📊 Resultados")
st.dataframe(res_df, use_container_width=True)

st.download_button(
    "⬇️ Baixar CSV",
    data=res_df.to_csv(index=False).encode("utf-8"),
    file_name="markov_tug_results_end_retro_limited.csv",
    mime="text/csv"
)

ok_files = [r["file"] for r in results if "error" not in r]
if not ok_files:
    st.warning("Nenhum arquivo foi processado com sucesso.")
    st.stop()

st.subheader("📈 Visualização por arquivo")
sel = st.selectbox("Escolha o arquivo", ok_files)

d = cache[sel]
t = d["t"]
norm = d["norm"]

i0_b, i1_b = d["i0_b"], d["i1_b"]
i0_f, i1_f = d["i0_f"], d["i1_f"]
start_i, end_i = d["start_i"], d["end_i"]
rest0_start = d["rest0_start"]
min_search_i = d["min_search_i"]

g1_i, g2_i = d["g1_i"], d["g2_i"]
g1_cs, g1_ce = d["g1_cs"], d["g1_ce"]
g2_cs, g2_ce = d["g2_cs"], d["g2_ce"]

colA, colB = st.columns(2)

with colA:
    fig = plt.figure(figsize=(7, 3))
    plt.plot(t, norm)

    plt.axvspan(t[i0_b], t[i1_b - 1], alpha=0.2, label="baseline inicial")
    plt.axvspan(t[i0_f], t[i1_f - 1], alpha=0.2, label="baseline final")

    if start_i is not None:
        plt.axvline(t[start_i], label="start")
    if min_search_i is not None:
        plt.axvline(t[min_search_i], linestyle=":", label="start+offset (min_search)")
    if rest0_start is not None:
        plt.axvline(t[rest0_start], linestyle="--", label="início repouso final (0-run)")
        # sombrear janela de busca do fim
        if min_search_i is not None and min_search_i < rest0_start:
            plt.axvspan(t[min_search_i], t[rest0_start], alpha=0.08, label="janela busca END")

    if end_i is not None:
        plt.axvline(t[end_i], label="end (último mov. forte)")

    if g1_i is not None:
        plt.axvline(t[g1_i], linestyle="--", label="G1")
        if g1_cs is not None and g1_ce is not None:
            plt.axvspan(t[g1_cs], t[g1_ce], alpha=0.15)

    if g2_i is not None:
        plt.axvline(t[g2_i], linestyle="--", label="G2")
        if g2_cs is not None and g2_ce is not None:
            plt.axvspan(t[g2_cs], t[g2_ce], alpha=0.15)

    plt.xlabel("Tempo (s)")
    plt.ylabel("Norma do giroscópio")
    plt.title("Norma + baselines + start/end + eventos")
    plt.legend(fontsize=7, loc="best")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

with colB:
    fig = plt.figure(figsize=(7, 3))
    plt.plot(t, d["ll_start"], label="LL (repouso inicial)")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Sliding log-likelihood")
    plt.title("LL para detectar START (fim é retro por estados)")
    plt.legend(fontsize=7, loc="best")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)
