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
st.set_page_config(page_title="TUG Markov + End (retro) + Eventos G1/G2", layout="wide")
st.title("📱 TUG — Markov + detecção robusta de FIM (retro) + eventos G1/G2")

st.markdown(
    """
### O que este app faz
1) **Pré-processa**: detrend → interpola p/ **100 Hz** → filtra (low-pass) → **norma** do giroscópio  
2) **K-means 1D** na norma (K estados)  
3) **Baselines adaptativos**: encontra janelas “mais paradas” no início e no fim  
4) Cria **dois mapas de estados**:
   - `states_start`: baseline inicial vira **0** (repouso inicial)
   - `states_end`: baseline final vira **0** (repouso final)
5) **Início (start)** por Markov+LL (queda persistente)  
6) **Fim (end)** (modo recomendado):  
   - acha repouso final persistente (`states_end == 0`)  
   - volta no tempo e marca o **último movimento forte** (`states_end >= 2`) antes do repouso
7) **G1/G2**: 2 picos na norma entre (start + offset) e end; componentes delimitados por retorno ao **estado 1**.
"""
)

# =========================================================
# Sidebar Parameters
# =========================================================
with st.sidebar:
    st.header("⚙️ Parâmetros")

    fs = st.number_input("Frequência alvo (Hz)", min_value=10.0, max_value=500.0, value=100.0, step=10.0)
    lowpass_hz = st.number_input("Low-pass (Hz)", min_value=1.0, max_value=80.0, value=15.0, step=1.0)

    st.divider()
    k_states = st.slider("K (estados do k-means 1D)", min_value=3, max_value=12, value=7, step=1)

    st.divider()
    st.subheader("Baselines adaptativos")
    win_s = st.number_input("Duração janela baseline (s)", min_value=0.5, max_value=6.0, value=2.0, step=0.5)

    search_first_s = st.number_input("Buscar baseline inicial nos primeiros (s)", min_value=2.0, max_value=40.0, value=10.0, step=1.0)
    guard_start_s = st.number_input("Guarda no início (s)", min_value=0.0, max_value=5.0, value=0.5, step=0.1)

    search_last_s = st.number_input("Buscar baseline final nos últimos (s)", min_value=2.0, max_value=40.0, value=10.0, step=1.0)
    guard_end_s = st.number_input("Guarda no final (s)", min_value=0.0, max_value=5.0, value=0.5, step=0.1)

    step_s = st.number_input("Passo varredura (s)", min_value=0.01, max_value=0.50, value=0.05, step=0.01)

    st.divider()
    st.subheader("Início por Markov (LL)")
    W_s = st.number_input("Janela LL W (s)", min_value=0.05, max_value=1.50, value=0.20, step=0.05)
    k_sigma_start = st.number_input("kσ início (thr = μ − kσ·σ)", min_value=0.5, max_value=10.0, value=3.0, step=0.5)

    st.subheader("Persistência R (R-grid)")
    r_min_s = st.number_input("R mínimo (s)", min_value=0.01, max_value=1.00, value=0.05, step=0.01)
    r_max_s = st.number_input("R máximo (s)", min_value=0.01, max_value=1.50, value=0.15, step=0.01)
    r_step_s = st.number_input("Passo R (s)", min_value=0.01, max_value=0.50, value=0.01, step=0.01)

    st.divider()
    st.subheader("Fim (End) — escolha do método")
    end_method = st.radio(
        "Método de fim",
        options=[
            "Retro: último movimento forte antes do repouso final (recomendado)",
            "Markov LL: retorno ao repouso final (mais conservador)"
        ],
        index=0
    )

    st.subheader("Fim Retro (estado ≥ baseline+Δ)")
    delta_states = st.number_input("Δ estados acima do repouso final", min_value=1, max_value=10, value=2, step=1)
    R_rest_s = st.number_input("Persistência repouso final (s)", min_value=0.02, max_value=2.0, value=0.15, step=0.01)
    R_move_s = st.number_input("Persistência movimento forte (s)", min_value=0.02, max_value=2.0, value=0.10, step=0.01)
    lookback_cap_s = st.number_input("Janela máxima para voltar atrás (s)", min_value=1.0, max_value=60.0, value=30.0, step=1.0)

    st.subheader("End LL (se usar)")
    k_sigma_end = st.number_input("kσ fim (thr = μ − kσ·σ)", min_value=0.5, max_value=10.0, value=3.0, step=0.5)
    end_aggregator = st.selectbox(
        "Agregador entre candidatos (fim LL)",
        options=["mais tarde (recomendado)", "mais cedo"],
        index=0
    )
    end_event_as_last_movement = st.checkbox("Se fim LL: usar end_i - 1 (última amostra antes do repouso)", value=True)

    st.divider()
    st.subheader("Eventos G1/G2 (picos)")
    peak_search_offset_s = st.number_input("Offset início p/ buscar picos (s)", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
    peak_prom = st.number_input("Prominência mínima (norma)", min_value=0.0, max_value=10.0, value=0.20, step=0.05)
    peak_min_dist_s = st.number_input("Distância mínima entre picos (s)", min_value=0.0, max_value=5.0, value=0.50, step=0.10)

    st.subheader("Componente via retorno a um estado")
    return_state = st.number_input("Estado de retorno (padrão = 1)", min_value=0, max_value=50, value=1, step=1)
    run_state_s = st.number_input("Sequência curta do estado de retorno (s)", min_value=0.01, max_value=1.00, value=0.10, step=0.01)

    st.divider()
    st.subheader("Opções")
    show_debug = st.checkbox("Mostrar debug", value=False)

st.divider()

uploads = st.file_uploader(
    "📂 Envie um ou mais arquivos .txt (colunas separadas por ';': tempo(ms); gx; gy; gz)",
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
            if np.any(mask):
                new_centers[j] = float(np.mean(x[mask]))
            else:
                new_centers[j] = float(x[RNG.integers(0, len(x))])

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

def first_persistent(ll: np.ndarray, start: int, thr: float, R: int, mode: str):
    for i in range(start, len(ll) - R):
        w = ll[i:i + R]
        if not np.all(np.isfinite(w)):
            continue
        if mode == "lt" and np.all(w < thr):
            return i
        if mode == "ge" and np.all(w >= thr):
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
        return None, ll, (np.nan, np.nan, np.nan), A0

    mu, sd = float(np.mean(ref)), float(np.std(ref))
    thr = mu - k_sigma * sd

    candidates = []
    for R in R_list:
        start_search = max(i1_b + W, 0)
        idx = first_persistent(ll, start=start_search, thr=thr, R=R, mode="lt")
        if idx is not None and idx >= i1_b:
            candidates.append(idx)

    if not candidates:
        return None, ll, (mu, sd, thr), A0

    start_i = int(min(candidates))
    return start_i, ll, (mu, sd, thr), A0

def detect_end_ll_retro(states_end: np.ndarray, i0_f: int, i1_f: int, W: int, R_list: list[int], k_sigma: float,
                        start_i: int | None, end_agg: str):
    n_states = int(states_end.max() + 1)
    Af = transition_matrix(states_end[i0_f:i1_f], n_states=n_states)
    ll = sliding_ll(states_end, Af, W=W)

    ref = ll[i0_f:i1_f]
    ref = ref[np.isfinite(ref)]
    if len(ref) < 5:
        return None, ll, (np.nan, np.nan, np.nan), Af

    mu, sd = float(np.mean(ref)), float(np.std(ref))
    thr = mu - k_sigma * sd

    min_i = 0 if start_i is None else max(0, start_i + 1)

    candidates = []
    for R in R_list:
        max_i = max(min_i, i0_f - R - 1)
        idx = None
        for i in range(max_i, min_i, -1):
            w = ll[i:i + R]
            if np.all(np.isfinite(w)) and np.all(w >= thr):
                idx = i
                break
        if idx is not None and idx <= i0_f:
            candidates.append(idx)

    if not candidates:
        return None, ll, (mu, sd, thr), Af

    end_i = int(max(candidates) if end_agg == "latest" else min(candidates))
    return end_i, ll, (mu, sd, thr), Af

# ---------- END retro: last strong movement before final rest ----------
def find_last_run_ge(states: np.ndarray, end_exclusive: int, run_len: int, thr_state: int, start_limit: int = 0):
    """
    Procura (de trás pra frente) o ÚLTIMO run de comprimento run_len onde states >= thr_state.
    Retorna o ÍNDICE FINAL do run (última amostra do run). Se não achar, retorna None.
    """
    end_exclusive = min(end_exclusive, len(states))
    start_limit = max(0, start_limit)
    if end_exclusive - start_limit < run_len:
        return None

    for i0 in range(end_exclusive - run_len, start_limit - 1, -1):
        if np.all(states[i0:i0 + run_len] >= thr_state):
            return i0 + run_len - 1
    return None

def find_first_run_eq_from_end(states: np.ndarray, run_len: int, value: int, end_exclusive: int | None = None):
    """
    Procura (de trás pra frente) um run de states == value e retorna o ÍNDICE INICIAL do run.
    (Ou seja, onde começa o repouso final persistente)
    """
    if end_exclusive is None:
        end_exclusive = len(states)
    end_exclusive = min(end_exclusive, len(states))
    if end_exclusive < run_len:
        return None

    for i0 in range(end_exclusive - run_len, -1, -1):
        if np.all(states[i0:i0 + run_len] == value):
            return i0
    return None

def detect_end_retro_strong(states_end: np.ndarray, start_i: int | None, fs: float,
                            delta_states: int, R_rest_s: float, R_move_s: float, lookback_cap_s: float):
    """
    1) acha começo do repouso final persistente (run de 0 no final)
    2) volta até achar o último run persistente de movimento forte (>= delta_states)
    3) end = final desse run
    """
    R_rest = max(1, int(round(R_rest_s * fs)))
    R_move = max(1, int(round(R_move_s * fs)))
    thr_state = int(delta_states)  # como repouso final = 0

    # limitar o quanto voltamos atrás (evita capturar movimento muito antigo)
    cap = max(1, int(round(lookback_cap_s * fs)))
    end_excl = len(states_end)

    rest0_start = find_first_run_eq_from_end(states_end, run_len=R_rest, value=0, end_exclusive=end_excl)
    if rest0_start is None:
        return None, None, None, {"reason": "No final rest run found", "R_rest": R_rest}

    # não volte antes de start_i (se existir)
    # e também não volte mais que 'cap' segundos antes do rest0_start
    start_limit = 0
    if start_i is not None:
        start_limit = max(start_limit, int(start_i))
    start_limit = max(start_limit, rest0_start - cap)

    last_strong_end = find_last_run_ge(
        states_end,
        end_exclusive=rest0_start,  # procura antes do repouso final
        run_len=R_move,
        thr_state=thr_state,
        start_limit=start_limit
    )

    if last_strong_end is None:
        return None, rest0_start, None, {"reason": "No strong movement run found", "R_move": R_move, "thr_state": thr_state}

    end_i = int(last_strong_end)
    return end_i, rest0_start, last_strong_end, {
        "R_rest": R_rest, "R_move": R_move, "thr_state": thr_state, "rest0_start": int(rest0_start), "start_limit": int(start_limit)
    }

# =========================================================
# G1/G2 + components via RETURN-STATE runs
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
    # volta (antes do pico) procurando um run do return_state
    r_end = None
    for j in range(min(peak_i - 1, i_end), i_start + run_len - 2, -1):
        if np.all(states[j - run_len + 1: j + 1] == return_state):
            r_end = j
            break

    # vai (depois do pico) procurando um run do return_state
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
end_agg = "latest" if end_aggregator.startswith("mais tarde") else "earliest"

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

        # baselines adaptativas
        i0_b, i1_b, _ = pick_quiet_window(
            norm, fs=fs, win_s=win_s,
            start_s=guard_start_s,
            end_s=search_first_s,
            step_s=step_s
        )

        total_s = float(t[-1] - t[0])
        end_region_start_s = max(0.0, total_s - search_last_s)
        end_region_end_s = max(0.0, total_s - guard_end_s)
        i0_f, i1_f, _ = pick_quiet_window(
            norm, fs=fs, win_s=win_s,
            start_s=end_region_start_s,
            end_s=end_region_end_s,
            step_s=step_s
        )

        # dois mapas de estados
        states_start, base_label_start = relabel_baseline_as_zero(labels, i0=i0_b, i1=i1_b)
        states_end, base_label_end = relabel_baseline_as_zero(labels, i0=i0_f, i1=i1_f)

        # START por Markov
        start_i, ll_start, (mu_s, sd_s, thr_s), _ = detect_start_markov_grid(
            states_start, i0_b=i0_b, i1_b=i1_b, W=W, R_list=R_list, k_sigma=k_sigma_start
        )

        # END
        ll_end = np.full_like(ll_start, np.nan, dtype=float)
        mu_e = sd_e = thr_e = np.nan
        end_i_raw = None
        rest0_start = None
        debug_end = {}

        if end_method.startswith("Retro"):
            end_i, rest0_start, end_i_raw, debug_end = detect_end_retro_strong(
                states_end=states_end,
                start_i=start_i,
                fs=fs,
                delta_states=int(delta_states),
                R_rest_s=float(R_rest_s),
                R_move_s=float(R_move_s),
                lookback_cap_s=float(lookback_cap_s)
            )
        else:
            end_i_raw, ll_end, (mu_e, sd_e, thr_e), _ = detect_end_ll_retro(
                states_end, i0_f=i0_f, i1_f=i1_f, W=W, R_list=R_list, k_sigma=k_sigma_end,
                start_i=start_i, end_agg=end_agg
            )
            end_i = end_i_raw
            if end_event_as_last_movement and end_i is not None:
                end_i = max(0, end_i - 1)

        # picos entre (start + offset) e end
        peak_start_i = None
        if start_i is not None:
            peak_start_i = min(len(norm) - 1, start_i + peak_offset_samples)

        g1_i, g2_i = find_two_peaks(
            norm=norm,
            i_start=peak_start_i,
            i_end=end_i,
            fs=fs,
            prom=peak_prom,
            min_dist_s=peak_min_dist_s,
        )

        g1_cs = g1_ce = g2_cs = g2_ce = None
        if g1_i is not None and start_i is not None and end_i is not None:
            g1_cs, g1_ce = component_bounds_from_runs(
                states_start, peak_i=g1_i, i_start=start_i, i_end=end_i, run_len=run_len, return_state=return_state
            )
        if g2_i is not None and start_i is not None and end_i is not None:
            g2_cs, g2_ce = component_bounds_from_runs(
                states_start, peak_i=g2_i, i_start=start_i, i_end=end_i, run_len=run_len, return_state=return_state
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
            "base_label_start(kmeans)": int(base_label_start),
            "base_label_end(kmeans)": int(base_label_end),
            "end_method": end_method,
            "rest0_start_s": safe_time(rest0_start),
            "end_candidate_s": safe_time(end_i_raw),
            "delta_states": int(delta_states),
            "R_rest_s": float(R_rest_s),
            "R_move_s": float(R_move_s),
            "thr_start": float(thr_s) if np.isfinite(thr_s) else np.nan,
            "thr_end": float(thr_e) if np.isfinite(thr_e) else np.nan,
            "G1_s": safe_time(g1_i),
            "G2_s": safe_time(g2_i),
            "G1_comp_start_s": safe_time(g1_cs),
            "G1_comp_end_s": safe_time(g1_ce),
            "G2_comp_start_s": safe_time(g2_cs),
            "G2_comp_end_s": safe_time(g2_ce),
            "debug_end": "" if not show_debug else str(debug_end),
        })

        cache[name] = dict(
            t=t, norm=norm,
            labels=labels,
            states_start=states_start, states_end=states_end,
            ll_start=ll_start, ll_end=ll_end,
            i0_b=i0_b, i1_b=i1_b, i0_f=i0_f, i1_f=i1_f,
            start_i=start_i, end_i=end_i, end_i_raw=end_i_raw,
            rest0_start=rest0_start,
            peak_start_i=peak_start_i,
            thr_s=thr_s, thr_e=thr_e,
            g1_i=g1_i, g2_i=g2_i,
            g1_cs=g1_cs, g1_ce=g1_ce, g2_cs=g2_cs, g2_ce=g2_ce,
        )

    except Exception as e:
        results.append({"file": name, "error": str(e)})

res_df = pd.DataFrame(results)

st.subheader("📊 Resultados")
st.dataframe(res_df, use_container_width=True)

st.download_button(
    "⬇️ Baixar CSV",
    data=res_df.to_csv(index=False).encode("utf-8"),
    file_name="markov_tug_results_end_retro.csv",
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
end_i_raw = d["end_i_raw"]
rest0_start = d["rest0_start"]
peak_start_i = d["peak_start_i"]

g1_i, g2_i = d["g1_i"], d["g2_i"]
g1_cs, g1_ce = d["g1_cs"], d["g1_ce"]
g2_cs, g2_ce = d["g2_cs"], d["g2_ce"]

ll_start = d["ll_start"]
ll_end = d["ll_end"]
thr_s, thr_e = d["thr_s"], d["thr_e"]

colA, colB = st.columns(2)

with colA:
    fig = plt.figure(figsize=(7, 3))
    plt.plot(t, norm)

    plt.axvspan(t[i0_b], t[i1_b - 1], alpha=0.2, label="baseline inicial")
    plt.axvspan(t[i0_f], t[i1_f - 1], alpha=0.2, label="baseline final")

    if start_i is not None:
        plt.axvline(t[start_i], label="start")
    if peak_start_i is not None:
        plt.axvline(t[peak_start_i], linestyle=":", label="start+offset")
    if rest0_start is not None:
        plt.axvline(t[rest0_start], linestyle="--", label="início repouso final (0-run)")
    if end_i_raw is not None:
        plt.axvline(t[end_i_raw], linestyle="--", label="end candidato")
    if end_i is not None:
        plt.axvline(t[end_i], label="end (aceito)")

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
    plt.plot(t, ll_start, label="LL (repouso inicial)")
    if np.isfinite(thr_s):
        plt.axhline(thr_s, linestyle="--", label="thr início")

    plt.plot(t, ll_end, label="LL (repouso final) [só se método LL]")
    if np.isfinite(thr_e):
        plt.axhline(thr_e, linestyle="--", label="thr fim")

    if start_i is not None:
        plt.axvline(t[start_i], label="start")
    if rest0_start is not None:
        plt.axvline(t[rest0_start], linestyle="--", label="início repouso final")
    if end_i is not None:
        plt.axvline(t[end_i], label="end")

    plt.xlabel("Tempo (s)")
    plt.ylabel("Sliding log-likelihood")
    plt.title("LL (para start; e para end se escolhido)")
    plt.legend(fontsize=7, loc="best")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)
