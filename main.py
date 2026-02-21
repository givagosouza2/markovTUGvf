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
st.set_page_config(page_title="TUG Markov (Adaptive Baselines)", layout="wide")
st.title("📱 TUG — Segmentação por Cadeia de Markov (Baselines adaptativos + LL)")

st.markdown(
    """
**Pipeline:**
1) detrend (x,y,z) → 2) interpolação para 100 Hz → 3) low-pass 15 Hz → 4) norma → 5) k-means 1D (K estados)  
6) **Baseline inicial adaptativo**: janela com menor variância nos primeiros X s (com guarda)  
7) **Início**: queda persistente de log-verossimilhança (LL) sob Markov do baseline inicial (**busca começa no fim do baseline**)  
8) **Baseline final adaptativo**: janela com menor variância nos últimos X s (com guarda)  
9) **Fim**: compatibilidade persistente (LL alta) sob Markov do baseline final (**busca retrógrada limitada até i0_f**)  
"""
)

# =========================================================
# Parameters
# =========================================================
with st.sidebar:
    st.header("⚙️ Parâmetros")

    fs = st.number_input("Frequência alvo (Hz)", min_value=10.0, max_value=500.0, value=100.0, step=10.0)
    lowpass_hz = st.number_input("Low-pass (Hz)", min_value=1.0, max_value=80.0, value=15.0, step=1.0)

    st.divider()
    k_states = st.slider("K (número de estados do k-means)", min_value=3, max_value=12, value=7, step=1)

    st.divider()
    st.subheader("Baselines adaptativos")
    win_s = st.number_input("Duração da janela baseline (s)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)

    search_first_s = st.number_input("Buscar baseline inicial nos primeiros (s)", min_value=2.0, max_value=30.0, value=10.0, step=1.0)
    guard_start_s = st.number_input("Guardar no início (s)", min_value=0.0, max_value=3.0, value=0.5, step=0.1)

    search_last_s = st.number_input("Buscar baseline final nos últimos (s)", min_value=2.0, max_value=30.0, value=10.0, step=1.0)
    guard_end_s = st.number_input("Guardar no final (s)", min_value=0.0, max_value=3.0, value=0.5, step=0.1)

    step_s = st.number_input("Passo da varredura (s)", min_value=0.01, max_value=0.50, value=0.05, step=0.01)

    st.divider()
    st.subheader("Detecção por LL (Markov)")
    W_s = st.number_input("Janela LL W (s)", min_value=0.05, max_value=1.00, value=0.20, step=0.05)
    R_s = st.number_input("Persistência R (s)", min_value=0.05, max_value=1.00, value=0.10, step=0.05)

    k_sigma_start = st.number_input("kσ início (thr = μ − kσ·σ)", min_value=1.0, max_value=8.0, value=3.0, step=0.5)
    k_sigma_end = st.number_input("kσ fim (thr = μ − kσ·σ)", min_value=1.0, max_value=8.0, value=3.0, step=0.5)

    st.divider()
    st.subheader("Opções")
    use_mean_penalty = st.checkbox("Penalizar janelas com média alta (var + λ·mean²)", value=False)
    lam = st.number_input("λ (se penalização ativa)", min_value=0.0, max_value=5.0, value=0.25, step=0.05)

    end_event_as_last_movement = st.checkbox("Definir fim como última amostra antes de entrar no repouso", value=True)

    show_debug = st.checkbox("Mostrar detalhes (debug)", value=False)

st.divider()

uploads = st.file_uploader(
    "📂 Envie um ou mais arquivos .txt (separados por ';' com colunas: tempo(ms); gx; gy; gz)",
    type=["txt"],
    accept_multiple_files=True
)

# =========================================================
# Core
# =========================================================
RNG = np.random.default_rng(42)


def read_gyro_txt_bytes(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes), sep=";", engine="python")
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[df.columns[0]])
    if df.shape[1] < 4:
        raise ValueError("Esperado: tempo(ms) + gx + gy + gz (4 colunas ou mais).")
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

    x_i = detrend(x_i)
    y_i = detrend(y_i)
    z_i = detrend(z_i)

    if lowpass_hz >= (fs / 2.0):
        raise ValueError("Low-pass deve ser menor que fs/2.")
    b, a = butter(4, lowpass_hz / (fs / 2.0), btype="low")
    x_f = filtfilt(b, a, x_i)
    y_f = filtfilt(b, a, y_i)
    z_f = filtfilt(b, a, z_i)

    norm = np.sqrt(x_f * x_f + y_f * y_f + z_f * z_f)
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
    """
    Escolhe a janela (win_s) com menor variância (ou var + lam*mean^2) entre [start_s, end_s].
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
            score = v + lam * (m * m)
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


# ------------------------------
# FIXED: search starts AFTER baseline end (+W)
# ------------------------------
def detect_start_markov(states: np.ndarray, i0_b: int, i1_b: int, W: int, R: int, k_sigma: float):
    n_states = int(states.max() + 1)
    A0 = transition_matrix(states[i0_b:i1_b], n_states=n_states)
    ll = sliding_ll(states, A0, W=W)

    ref = ll[i0_b:i1_b]
    ref = ref[np.isfinite(ref)]
    mu, sd = float(np.mean(ref)), float(np.std(ref))
    thr = mu - k_sigma * sd

    # ✅ start search after baseline + W (avoids NaN region & prevents early detections)
    start_search = max(i1_b + W, 0)
    start_i = first_persistent(ll, start=start_search, thr=thr, R=R, mode="lt")

    # hard constraint
    if start_i is not None and start_i < i1_b:
        start_i = None

    return start_i, ll, (mu, sd, thr), A0


# ------------------------------
# FIXED: end search is RETRO but limited to i0_f (baseline start)
# ------------------------------
def detect_end_markov_retro(states: np.ndarray, i0_f: int, i1_f: int, W: int, R: int, k_sigma: float, start_i: int | None):
    n_states = int(states.max() + 1)
    Af = transition_matrix(states[i0_f:i1_f], n_states=n_states)
    ll = sliding_ll(states, Af, W=W)

    ref = ll[i0_f:i1_f]
    ref = ref[np.isfinite(ref)]
    mu, sd = float(np.mean(ref)), float(np.std(ref))
    thr = mu - k_sigma * sd

    # min boundary (do not cross start)
    min_i = 0 if start_i is None else max(0, start_i + 1)

    # ✅ max boundary: do not search after baseline-final start (i0_f)
    # we want "entry into rest" BEFORE final baseline.
    max_i = max(min_i, i0_f - R - 1)

    end_i = None
    for i in range(max_i, min_i, -1):
        w = ll[i:i + R]
        if np.all(np.isfinite(w)) and np.all(w >= thr):
            end_i = i
            break

    # hard constraint
    if end_i is not None and end_i > i0_f:
        end_i = None

    return end_i, ll, (mu, sd, thr), Af


# =========================================================
# Run
# =========================================================
if not uploads:
    st.info("Envie ao menos 1 arquivo .txt para rodar a análise.")
    st.stop()

W = int(round(W_s * fs))
R = int(round(R_s * fs))
if W < 1 or R < 1:
    st.error("W e R precisam ser >= 1 amostra. Aumente W_s / R_s.")
    st.stop()

results = []
cache = {}

for up in uploads:
    name = up.name
    try:
        df = read_gyro_txt_bytes(up.getvalue())
        t, norm = preprocess_to_norm(df, fs=fs, lowpass_hz=lowpass_hz)

        labels, centers = kmeans_1d(norm, k=k_states)

        # Baseline inicial adaptativo (nos primeiros X s, depois do guard)
        i0_b, i1_b, score_b = pick_quiet_window(
            norm, fs=fs, win_s=win_s,
            start_s=guard_start_s,
            end_s=search_first_s,
            step_s=step_s, use_mean_penalty=use_mean_penalty, lam=lam
        )

        # Relabel state 0 using the chosen initial baseline window
        states, base_label = relabel_baseline_as_zero(labels, i0=i0_b, i1=i1_b)

        # Início (start) - busca começa no fim do baseline
        start_i, ll_start, (mu_s, sd_s, thr_s), _ = detect_start_markov(
            states, i0_b=i0_b, i1_b=i1_b, W=W, R=R, k_sigma=k_sigma_start
        )
        start_t = float(t[start_i]) if start_i is not None else np.nan

        # Baseline final adaptativo: dentro dos últimos X s (antes do guard final)
        total_s = float(t[-1] - t[0])
        end_region_start_s = max(0.0, total_s - search_last_s)
        end_region_end_s = max(0.0, total_s - guard_end_s)

        i0_f, i1_f, score_f = pick_quiet_window(
            norm, fs=fs, win_s=win_s,
            start_s=end_region_start_s,
            end_s=end_region_end_s,
            step_s=step_s, use_mean_penalty=use_mean_penalty, lam=lam
        )

        # Fim (end) - retro scan limitado até i0_f
        end_i, ll_end, (mu_e, sd_e, thr_e), _ = detect_end_markov_retro(
            states, i0_f=i0_f, i1_f=i1_f, W=W, R=R, k_sigma=k_sigma_end, start_i=start_i
        )

        # opcional: marcar fim como última amostra ANTES da entrada no repouso
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
            "thr_start": float(thr_s),
            "thr_end": float(thr_e),
            "n_samples_100Hz": int(len(t)),
        })

        cache[name] = dict(
            t=t, norm=norm, states=states,
            ll_start=ll_start, ll_end=ll_end,
            i0_b=i0_b, i1_b=i1_b, i0_f=i0_f, i1_f=i1_f,
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
    file_name="markov_tug_results_streamlit.csv",
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
    # baseline windows
    plt.axvspan(t[i0_b], t[i1_b - 1], alpha=0.2)
    plt.axvspan(t[i0_f], t[i1_f - 1], alpha=0.2)
    # start/end
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
    plt.axhline(thr_s, linestyle="--", label="thr início")
    plt.plot(t, ll_end, label="LL (baseline final)")
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
        "init_baseline_idx": (int(i0_b), int(i1_b)),
        "final_baseline_idx": (int(i0_f), int(i1_f)),
        "start_idx": None if start_i is None else int(start_i),
        "end_idx": None if end_i is None else int(end_i),
        "thr_start": float(thr_s),
        "thr_end": float(thr_e),
        "W_samples": int(W),
        "R_samples": int(R),
        "constraints": {
            "start_search_from": int(i1_b + W),
            "end_search_to": int(i0_f - 1),
        }
    })
