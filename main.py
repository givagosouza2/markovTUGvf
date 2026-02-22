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
st.set_page_config(page_title="TUG Markov + Eventos (G1/G2)", layout="wide")
st.title("📱 TUG — Markov (baselines adaptativos + LL + R-grid) + Eventos G1/G2")

st.markdown(
    """
**Etapas:**
- Preprocessamento: detrend → interpolação p/ 100 Hz → low-pass 15 Hz → norma
- Discretização: k-means 1D (K estados) e relabel do estado 0 (repouso inicial)
- Start/End: Markov + log-verossimilhança (LL) com loop em R (persistência)
- Eventos: busca de **dois picos** entre start e end: **G1 (primeiro)** e **G2 (último)**  
  e delimitação do componente de cada pico via **sequência curta de retorno ao estado escolhido** (padrão = estado 1)
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
    st.subheader("Log-verossimilhança (LL)")
    W_s = st.number_input("Janela LL W (s)", min_value=0.05, max_value=1.50, value=0.20, step=0.05)
    k_sigma_start = st.number_input("kσ início (thr = μ − kσ·σ)", min_value=0.5, max_value=10.0, value=3.0, step=0.5)
    k_sigma_end = st.number_input("kσ fim (thr = μ − kσ·σ)", min_value=0.5, max_value=10.0, value=3.0, step=0.5)

    st.divider()
    st.subheader("Loop em persistência R (R-grid)")
    r_min_s = st.number_input("R mínimo (s)", min_value=0.01, max_value=1.00, value=0.05, step=0.01)
    r_max_s = st.number_input("R máximo (s)", min_value=0.01, max_value=1.50, value=0.15, step=0.01)
    r_step_s = st.number_input("Passo R (s)", min_value=0.01, max_value=0.50, value=0.01, step=0.01)

    end_aggregator = st.selectbox(
        "Como escolher o fim entre candidatos?",
        options=["mais tarde (recomendado)", "mais cedo"],
        index=0
    )

    st.divider()
    st.subheader("Eventos G1/G2 (picos)")
    peak_prom = st.number_input("Prominência mínima (norma)", min_value=0.0, max_value=10.0, value=0.20, step=0.05)
    peak_min_dist_s = st.number_input("Distância mínima entre picos (s)", min_value=0.0, max_value=5.0, value=0.50, step=0.10)

    st.subheader("Componente via retorno a um estado")
    return_state = st.number_input("Estado de retorno (padrão = 1)", min_value=0, max_value=50, value=1, step=1)
    run_state_s = st.number_input("Sequência curta do estado de retorno (s)", min_value=0.01, max_value=1.00, value=0.10, step=0.01)

    st.divider()
    st.subheader("Opções")
    use_mean_penalty = st.checkbox("Penalizar janelas com média alta (var + λ·mean²)", value=False)
    lam = st.number_input("λ (se penalização ativa)", min_value=0.0, max_value=10.0, value=0.25, step=0.05)

    end_event_as_last_movement = st.checkbox("Fim = última amostra antes do repouso (end_i - 1)", value=True)
    show_debug = st.checkbox("Mostrar debug", value=False)

st.divider()

uploads = st.file_uploader(
    "📂 Envie um ou mais arquivos .txt (separados por ';' com colunas: tempo(ms); gx; gy; gz)",
    type=["txt"],
    accept_multiple_files=True
)

# =========================================================
# Core helpers
# =========================================================
RNG = np.random.default_rng(42)

def read_gyro_txt_bytes(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes), sep=";", engine="python")
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[df.columns[0]])
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

def pick_quiet_window(norm: np.ndarray, fs: float, win_s: float, start_s: float, end_s: float,
                      step_s: float, use_mean_penalty: bool, lam: float):
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
        v = float(np.var(x))
        if use_mean_penalty:
            m = float(np.mean(x))
            score = v + lam*(m*m)
        else:
            score = v
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

# =========================================================
# R-grid detectors
# =========================================================
def make_R_list_samples(fs: float, r_min_s: float, r_max_s: float, r_step_s: float):
    if r_step_s <= 0:
        return [max(1, int(round(r_min_s * fs)))]
    if r_max_s < r_min_s:
        r_max_s = r_min_s
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

def detect_end_markov_retro_grid(states: np.ndarray, i0_f: int, i1_f: int, W: int, R_list: list[int], k_sigma: float,
                                 start_i: int | None, end_agg: str):
    n_states = int(states.max() + 1)
    Af = transition_matrix(states[i0_f:i1_f], n_states=n_states)
    ll = sliding_ll(states, Af, W=W)

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

    if end_agg == "earliest":
        end_i = int(min(candidates))
    else:
        end_i = int(max(candidates))
    return end_i, ll, (mu, sd, thr), Af

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
    g1 = int(peaks_global[0])
    g2 = int(peaks_global[-1])
    if g1 == g2:
        return None, None
    return g1, g2

def last_run_end_before(states: np.ndarray, peak_i: int, i_min: int, run_len: int, state_val: int):
    start = max(i_min + run_len - 1, 0)
    end = min(peak_i - 1, len(states) - 1)
    for j in range(end, start - 1, -1):
        if np.all(states[j - run_len + 1: j + 1] == state_val):
            return j
    return None

def first_run_start_after(states: np.ndarray, peak_i: int, i_max: int, run_len: int, state_val: int):
    j_start = max(peak_i + 1, 0)
    j_end = min(i_max - run_len + 1, len(states) - run_len)
    for j in range(j_start, j_end + 1):
        if np.all(states[j: j + run_len] == state_val):
            return j
    return None

def component_bounds_from_runs(states: np.ndarray, peak_i: int, i_start: int, i_end: int, run_len: int, return_state: int):
    """
    Delimita componente usando runs do return_state (padrão=1):
      comp_start = last_run_end_before + 1
      comp_end   = first_run_start_after - 1
    Se não encontrar run, usa fallback i_start/i_end.
    """
    r_end = last_run_end_before(states, peak_i, i_min=i_start, run_len=run_len, state_val=return_state)
    r_start = first_run_start_after(states, peak_i, i_max=i_end, run_len=run_len, state_val=return_state)

    comp_start = (r_end + 1) if r_end is not None else i_start
    comp_end = (r_start - 1) if r_start is not None else i_end

    comp_start = int(max(i_start, min(peak_i, comp_start)))
    comp_end = int(min(i_end, max(peak_i, comp_end)))
    return comp_start, comp_end, r_end, r_start

# =========================================================
# Run
# =========================================================
if not uploads:
    st.info("Envie ao menos 1 arquivo .txt para rodar a análise.")
    st.stop()

W = int(round(W_s * fs))
if W < 1:
    st.error("W precisa ser >= 1 amostra. Aumente W_s.")
    st.stop()

R_list = make_R_list_samples(fs, r_min_s, r_max_s, r_step_s)
if len(R_list) == 0:
    st.error("R_list vazio. Verifique r_min_s / r_max_s / r_step_s.")
    st.stop()

end_agg = "latest" if end_aggregator.startswith("mais tarde") else "earliest"
peak_min_dist = peak_min_dist_s
run_len = max(1, int(round(run_state_s * fs)))
return_state = int(return_state)

results = []
cache = {}

for up in uploads:
    name = up.name
    try:
        df = read_gyro_txt_bytes(up.getvalue())
        t, norm = preprocess_to_norm(df, fs=fs, lowpass_hz=lowpass_hz)

        labels, _ = kmeans_1d(norm, k=k_states)

        # Baseline inicial adaptativo
        i0_b, i1_b, score_b = pick_quiet_window(
            norm, fs=fs, win_s=win_s,
            start_s=guard_start_s,
            end_s=search_first_s,
            step_s=step_s,
            use_mean_penalty=use_mean_penalty, lam=lam
        )

        # Estado 0 (repouso inicial)
        states, base_label = relabel_baseline_as_zero(labels, i0=i0_b, i1=i1_b)

        # START
        start_i, ll_start, (_, _, thr_s), _ = detect_start_markov_grid(
            states, i0_b=i0_b, i1_b=i1_b, W=W, R_list=R_list, k_sigma=k_sigma_start
        )
        start_t = float(t[start_i]) if start_i is not None else np.nan

        # Baseline final adaptativo
        total_s = float(t[-1] - t[0])
        end_region_start_s = max(0.0, total_s - search_last_s)
        end_region_end_s = max(0.0, total_s - guard_end_s)

        i0_f, i1_f, score_f = pick_quiet_window(
            norm, fs=fs, win_s=win_s,
            start_s=end_region_start_s,
            end_s=end_region_end_s,
            step_s=step_s,
            use_mean_penalty=use_mean_penalty, lam=lam
        )

        # END
        end_i, ll_end, (_, _, thr_e), _ = detect_end_markov_retro_grid(
            states, i0_f=i0_f, i1_f=i1_f, W=W, R_list=R_list, k_sigma=k_sigma_end,
            start_i=start_i, end_agg=end_agg
        )

        if end_event_as_last_movement and end_i is not None:
            end_i = max(0, end_i - 1)

        end_t = float(t[end_i]) if end_i is not None else np.nan
        dur = float(end_t - start_t) if np.isfinite(start_t) and np.isfinite(end_t) else np.nan

        # Peaks G1/G2
        g1_i, g2_i = find_two_peaks(
            norm=norm,
            i_start=start_i if start_i is not None else None,
            i_end=end_i if end_i is not None else None,
            fs=fs,
            prom=peak_prom,
            min_dist_s=peak_min_dist,
        )

        # Components around peaks using return_state runs
        g1_cs = g1_ce = g2_cs = g2_ce = None
        if g1_i is not None and start_i is not None and end_i is not None:
            g1_cs, g1_ce, _, _ = component_bounds_from_runs(
                states, peak_i=g1_i, i_start=start_i, i_end=end_i, run_len=run_len, return_state=return_state
            )
        if g2_i is not None and start_i is not None and end_i is not None:
            g2_cs, g2_ce, _, _ = component_bounds_from_runs(
                states, peak_i=g2_i, i_start=start_i, i_end=end_i, run_len=run_len, return_state=return_state
            )

        def safe_time(idx):
            if idx is None or idx < 0 or idx >= len(t):
                return np.nan
            return float(t[idx])

        def safe_val(idx):
            if idx is None or idx < 0 or idx >= len(norm):
                return np.nan
            return float(norm[idx])

        results.append({
            "file": name,
            "start_s": start_t,
            "end_s": end_t,
            "duration_s": dur,

            "G1_time_s": safe_time(g1_i),
            "G1_value": safe_val(g1_i),
            "G1_comp_start_s": safe_time(g1_cs),
            "G1_comp_end_s": safe_time(g1_ce),
            "G1_comp_dur_s": safe_time(g1_ce) - safe_time(g1_cs) if (g1_cs is not None and g1_ce is not None) else np.nan,

            "G2_time_s": safe_time(g2_i),
            "G2_value": safe_val(g2_i),
            "G2_comp_start_s": safe_time(g2_cs),
            "G2_comp_end_s": safe_time(g2_ce),
            "G2_comp_dur_s": safe_time(g2_ce) - safe_time(g2_cs) if (g2_cs is not None and g2_ce is not None) else np.nan,

            "return_state": return_state,
            "run_len_samples": int(run_len),

            "initBL_t0_s": float(t[i0_b]),
            "initBL_t1_s": float(t[i1_b - 1]),
            "finalBL_t0_s": float(t[i0_f]),
            "finalBL_t1_s": float(t[i1_f - 1]),

            "thr_start": float(thr_s) if np.isfinite(thr_s) else np.nan,
            "thr_end": float(thr_e) if np.isfinite(thr_e) else np.nan,

            "W_samples": int(W),
            "R_list_samples": ",".join(map(str, R_list)),
            "n_samples_100Hz": int(len(t)),
        })

        cache[name] = dict(
            t=t, norm=norm, states=states,
            ll_start=ll_start, ll_end=ll_end,
            i0_b=i0_b, i1_b=i1_b,
            i0_f=i0_f, i1_f=i1_f,
            start_i=start_i, end_i=end_i,
            thr_s=thr_s, thr_e=thr_e,
            g1_i=g1_i, g2_i=g2_i,
            g1_cs=g1_cs, g1_ce=g1_ce,
            g2_cs=g2_cs, g2_ce=g2_ce,
            return_state=return_state,
            run_len=run_len,
        )

    except Exception as e:
        results.append({"file": name, "error": str(e)})

res_df = pd.DataFrame(results)

st.subheader("📊 Resultados")
st.dataframe(res_df, use_container_width=True)

csv_bytes = res_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Baixar CSV (resultados)",
    data=csv_bytes,
    file_name="markov_tug_results_streamlit_with_G1G2_returnState.csv",
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
ll_start = d["ll_start"]
ll_end = d["ll_end"]

i0_b, i1_b = d["i0_b"], d["i1_b"]
i0_f, i1_f = d["i0_f"], d["i1_f"]
start_i, end_i = d["start_i"], d["end_i"]
thr_s, thr_e = d["thr_s"], d["thr_e"]

g1_i, g2_i = d["g1_i"], d["g2_i"]
g1_cs, g1_ce = d["g1_cs"], d["g1_ce"]
g2_cs, g2_ce = d["g2_cs"], d["g2_ce"]

colA, colB = st.columns(2)

with colA:
    fig = plt.figure(figsize=(7, 3))
    plt.plot(t, norm)

    plt.axvspan(t[i0_b], t[i1_b - 1], alpha=0.2)
    plt.axvspan(t[i0_f], t[i1_f - 1], alpha=0.2)

    if start_i is not None:
        plt.axvline(t[start_i])
    if end_i is not None:
        plt.axvline(t[end_i])

    if g1_i is not None:
        plt.axvline(t[g1_i], linestyle="--")
        if g1_cs is not None and g1_ce is not None:
            plt.axvspan(t[g1_cs], t[g1_ce], alpha=0.15)

    if g2_i is not None:
        plt.axvline(t[g2_i], linestyle="--")
        if g2_cs is not None and g2_ce is not None:
            plt.axvspan(t[g2_cs], t[g2_ce], alpha=0.15)

    plt.xlabel("Tempo (s)")
    plt.ylabel("Norma do giroscópio")
    plt.title(f"Norma + baselines + start/end + G1/G2 (retorno ao estado {d['return_state']})")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

with colB:
    fig = plt.figure(figsize=(7, 3))
    plt.plot(t, ll_start, label="LL (baseline inicial)")
    if np.isfinite(thr_s):
        plt.axhline(thr_s, linestyle="--", label="thr início")

    plt.plot(t, ll_end, label="LL (baseline final)")
    if np.isfinite(thr_e):
        plt.axhline(thr_e, linestyle="--", label="thr fim")

    if start_i is not None:
        plt.axvline(t[start_i])
    if end_i is not None:
        plt.axvline(t[end_i])

    if g1_i is not None:
        plt.axvline(t[g1_i], linestyle="--")
    if g2_i is not None:
        plt.axvline(t[g2_i], linestyle="--")

    plt.xlabel("Tempo (s)")
    plt.ylabel("Sliding log-likelihood")
    plt.title("LL inicial vs LL final + eventos")
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

if show_debug:
    st.markdown("### 🧪 Debug")
    st.write({
        "return_state": d["return_state"],
        "run_len_samples": int(d["run_len"]),
        "start_idx": None if start_i is None else int(start_i),
        "end_idx": None if end_i is None else int(end_i),
        "g1_idx": None if g1_i is None else int(g1_i),
        "g2_idx": None if g2_i is None else int(g2_i),
    })
