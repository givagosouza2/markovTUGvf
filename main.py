# app.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy.signal import detrend, butter, filtfilt
from scipy.interpolate import interp1d

# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(page_title="TUG Markov (Adaptive Baselines + R-grid)", layout="wide")
st.title("📱 TUG — Segmentação por Cadeia de Markov (Baselines adaptativos + LL + loop em R)")

st.markdown(
    """
**O que este app faz (robusto):**
- Baseline **inicial** e **final** são escolhidos automaticamente como as janelas de **menor variância** da norma.
- O início (start) é detectado por **queda persistente** da LL sob o Markov do baseline inicial.
- O fim (end) é detectado por **compatibilidade persistente** da LL sob o Markov do baseline final, com **busca retrógrada**.
- Para aumentar robustez, o app faz um **loop for em várias persistências R** (R-grid) e:
  - **Start:** escolhe o **mais cedo** entre os candidatos (menor índice).
  - **End:** por padrão escolhe o **mais tarde** entre os candidatos (maior índice), pois é o “último retorno ao repouso”.
    (você pode trocar para “mais cedo” se quiser)
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
        "Como escolher o fim entre os candidatos?",
        options=["mais tarde (recomendado)", "mais cedo"],
        index=0
    )

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

    # remove duplicates in time
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

    # reorder by center magnitude
    order = np.argsort(centers)
    centers = centers[order]
    inv = np.empty_like(order)
    inv[order] = np.arange(k)
    labels = inv[labels]
    return labels.astype(int), centers

def pick_quiet_window(norm: np.ndarray, fs: float, win_s: float, start_s: float, end_s: float,
                      step_s: float, use_mean_penalty: bool, lam: float):
    """
    Escolhe janela de menor variância (ou var + lam*mean^2) entre [start_s, end_s].
    Retorna (i0, i1, score).
    """
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
    # LL over transitions in a sliding window of length W transitions
    lp = np.log(A[seq[:-1], seq[1:]])
    cs = np.concatenate([[0.0], np.cumsum(lp)])
    ll = np.full(seq.shape[0], np.nan, float)
    idx = np.arange(W, seq.shape[0] - 1)
    ll[idx] = cs[idx] - cs[idx - W]
    return ll

def first_persistent(ll: np.ndarray, start: int, thr: float, R: int, mode: str):
    """
    Returns first index i >= start such that ll[i:i+R] persistently:
      mode="lt": < thr
      mode="ge": >= thr
    """
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
# R-grid detectors (your idea)
# =========================================================
def make_R_list_samples(fs: float, r_min_s: float, r_max_s: float, r_step_s: float):
    if r_step_s <= 0:
        return [max(1, int(round(r_min_s * fs)))]
    if r_max_s < r_min_s:
        r_max_s = r_min_s
    r_vals = np.arange(r_min_s, r_max_s + 1e-12, r_step_s)
    R_list = [max(1, int(round(fs * r))) for r in r_vals]
    # unique, sorted
    R_list = sorted(list(set(R_list)))
    return R_list

def detect_start_markov_grid(states: np.ndarray, i0_b: int, i1_b: int, W: int, R_list: list[int], k_sigma: float):
    """
    Fit Markov on initial baseline.
    Find start candidates for each R in R_list, then choose the EARLIEST (min index).
    Search ALWAYS starts after baseline end (+W) to avoid early detection.
    """
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

    start_i = int(min(candidates))  # your "momento inferior"
    return start_i, ll, (mu, sd, thr), A0

def detect_end_markov_retro_grid(states: np.ndarray, i0_f: int, i1_f: int, W: int, R_list: list[int], k_sigma: float,
                                 start_i: int | None, end_agg: str):
    """
    Fit Markov on final baseline.
    Retro-scan for each R; candidates are indices where ll[i:i+R] >= thr.
    IMPORTANT: restrict search to BEFORE the start of final baseline (<= i0_f).
    Aggregation:
      - 'latest' (recommended): choose max(candidates) (closest to final rest)
      - 'earliest': choose min(candidates)
    """
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
        max_i = max(min_i, i0_f - R - 1)  # do not search after baseline start
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
        end_i = int(max(candidates))  # recommended
    return end_i, ll, (mu, sd, thr), Af

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
    st.error("R_list ficou vazio. Verifique r_min_s / r_max_s / r_step_s.")
    st.stop()

end_agg = "latest" if end_aggregator.startswith("mais tarde") else "earliest"

results = []
cache = {}

for up in uploads:
    name = up.name
    try:
        df = read_gyro_txt_bytes(up.getvalue())
        t, norm = preprocess_to_norm(df, fs=fs, lowpass_hz=lowpass_hz)

        labels, _ = kmeans_1d(norm, k=k_states)

        # Adaptive initial baseline: within first [guard_start, search_first]
        i0_b, i1_b, score_b = pick_quiet_window(
            norm, fs=fs, win_s=win_s,
            start_s=guard_start_s,
            end_s=search_first_s,
            step_s=step_s,
            use_mean_penalty=use_mean_penalty, lam=lam
        )

        # relabel state 0 using initial baseline window
        states, base_label = relabel_baseline_as_zero(labels, i0=i0_b, i1=i1_b)

        # START (R-grid, choose earliest)
        start_i, ll_start, (mu_s, sd_s, thr_s), _ = detect_start_markov_grid(
            states, i0_b=i0_b, i1_b=i1_b, W=W, R_list=R_list, k_sigma=k_sigma_start
        )
        start_t = float(t[start_i]) if start_i is not None else np.nan

        # Adaptive final baseline region in seconds
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

        # END (R-grid, retro, aggregated)
        end_i, ll_end, (mu_e, sd_e, thr_e), _ = detect_end_markov_retro_grid(
            states, i0_f=i0_f, i1_f=i1_f, W=W, R_list=R_list, k_sigma=k_sigma_end,
            start_i=start_i, end_agg=end_agg
        )

        # Optional: define end as last movement sample before entry into rest
        if end_event_as_last_movement and end_i is not None:
            end_i = max(0, end_i - 1)

        end_t = float(t[end_i]) if end_i is not None else np.nan
        dur = float(end_t - start_t) if np.isfinite(start_t) and np.isfinite(end_t) else np.nan

        results.append({
            "file": name,
            "start_s": start_t,
            "end_s": end_t,
            "duration_s": dur,
            "initBL_t0_s": float(t[i0_b]),
            "initBL_t1_s": float(t[i1_b - 1]),
            "initBL_score": float(score_b),
            "finalBL_t0_s": float(t[i0_f]),
            "finalBL_t1_s": float(t[i1_f - 1]),
            "finalBL_score": float(score_f),
            "thr_start": float(thr_s) if np.isfinite(thr_s) else np.nan,
            "thr_end": float(thr_e) if np.isfinite(thr_e) else np.nan,
            "R_list_samples": ",".join(map(str, R_list)),
            "W_samples": int(W),
            "n_samples_100Hz": int(len(t)),
        })

        cache[name] = dict(
            t=t, norm=norm,
            ll_start=ll_start, ll_end=ll_end,
            i0_b=i0_b, i1_b=i1_b,
            i0_f=i0_f, i1_f=i1_f,
            start_i=start_i, end_i=end_i,
            thr_s=thr_s, thr_e=thr_e
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
    file_name="markov_tug_results_streamlit_Rgrid.csv",
    mime="text/csv"
)

ok_files = [r["file"] for r in results if "error" not in r]
if not ok_files:
    st.warning("Nenhum arquivo foi processado com sucesso.")
    st.stop()

st.subheader("📈 Visualização por arquivo")
sel = st.selectbox("Escolha o arquivo", ok_files)

data = cache[sel]
t = data["t"]
norm = data["norm"]
ll_start = data["ll_start"]
ll_end = data["ll_end"]
i0_b, i1_b = data["i0_b"], data["i1_b"]
i0_f, i1_f = data["i0_f"], data["i1_f"]
start_i, end_i = data["start_i"], data["end_i"]
thr_s, thr_e = data["thr_s"], data["thr_e"]

colA, colB = st.columns(2)

with colA:
    fig = plt.figure(figsize=(7, 3))
    plt.plot(t, norm)
    # baselines
    plt.axvspan(t[i0_b], t[i1_b - 1], alpha=0.2)
    plt.axvspan(t[i0_f], t[i1_f - 1], alpha=0.2)
    # events
    if start_i is not None:
        plt.axvline(t[start_i])
    if end_i is not None:
        plt.axvline(t[end_i])
    plt.xlabel("Tempo (s)")
    plt.ylabel("Norma do giroscópio")
    plt.title("Norma + baselines adaptativos + start/end")
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
    plt.xlabel("Tempo (s)")
    plt.ylabel("Sliding log-likelihood")
    plt.title("LL inicial vs LL final (Markov)")
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

if show_debug:
    st.markdown("### 🧪 Debug")
    st.write({
        "W_samples": int(W),
        "R_list_samples": R_list,
        "init_baseline_idx": (int(i0_b), int(i1_b)),
        "final_baseline_idx": (int(i0_f), int(i1_f)),
        "start_idx": None if start_i is None else int(start_i),
        "end_idx": None if end_i is None else int(end_i),
        "start_search_from_idx": int(i1_b + W),
        "end_search_max_idx": int(i0_f - 1),
        "thr_start": float(thr_s) if np.isfinite(thr_s) else None,
        "thr_end": float(thr_e) if np.isfinite(thr_e) else None,
        "end_aggregator": end_agg,
    })
