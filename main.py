import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import soundfile as sf
import io
import time

# setup
st.set_page_config(page_title="what the fletch", layout="wide")

# fun styling stuff 
# (mostly just centering things)
st.html(
    """
    <style>
    .stMainBlockContainer, 
    div[data-testid="stMainBlockContainer"] {
        max-width: 950px !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        margin: 0 auto !important;
    }
    /* 1. Center all headers and subheaders */
    h1, h2, h3, h4, h5, h6 {
        text-align: center !important;
    }
    
    .stWidget label {
        display: flex !important;
        justify-content: center !important;
        text-align: center !important;
        width: 100% !important;
    }
    
    [role="radiogroup"] {
        justify-content: center !important;
    }
    
    [role="radiogroup"] label {
        text-align: center !important;
    }
    
    .stTextInput input, .stNumberInput input {
        text-align: center !important;
    }

    div[data-testid="stRadio"] label[data-testid="stWidgetLabel"] {
        display: flex !important;
        justify-content: center !important;
        width: 100% !important;
    }
    
    </style>
    """
)


# setting u p constants
TEST_FREQS = np.round(np.logspace(np.log10(50), np.log10(12500), 6)).astype(int).tolist()
LEVELS = {"Low": 0.03, "Comfortable": 0.12, "Loud": 0.35}
DEMO_FILE = "airplane mode.mp3"

# hello TAs! for streamlit apps to work correctly and avoid weird refreshing issues, 
# I have to use session_state to store the threshold and levels data.
# This is because streamlit reruns the script on every interaction,
# so without session_state, the data would be lost. just a heads up in case you're
# wondering what this is!! :)

if "threshold" not in st.session_state:
    st.session_state.threshold = {}
if "levels" not in st.session_state:
    st.session_state.levels = {}

# helper func's

def read_audio(file_bytes):
    data, fs = sf.read(io.BytesIO(file_bytes), dtype="float32", always_2d=False)
    return data, fs

def tone_gen(freq, duration=1.5, amp=0.2, fs=44100, fade=0.05):
    n = int(fs*duration)
    t = np.arange(n)/fs
    tone = amp * np.sin(2*np.pi * freq * t)
    nf = max(1, int(fs*fade))
    ramp = 0.5*(1 - np.cos(np.linspace(0, np.pi, nf)))
    tone[:nf] *= ramp
    tone[-nf:] *= ramp[::-1]
    return tone.astype(np.float32), fs

def to_wav_bytes(x, fs, normalize=False):
    if normalize:
        peak = np.max(np.abs(x))
        if peak > 1e-9:
            x = x*(0.98 / peak)
    buf = io.BytesIO()
    sf.write(buf, np.clip(x, -1, 1), fs, subtype="PCM_16", format="WAV") # pcm_16 sf.available_subtypes()
    return buf.getvalue()

def process(x, fs, freqs_hz, gains_db, block=4096, hop=1024):
    window = np.hanning(block).astype(np.float32)
    bin_freqs = np.fft.rfftfreq(block, d=1.0 / fs)
    gain_db = np.interp(bin_freqs, freqs_hz, gains_db,
                         left=gains_db[0], right=gains_db[-1])
    gain = (10** (gain_db / 20.0)).astype(np.float32)

    x = np.asarray(x, dtype=np.float32)
    xp = np.concatenate([np.zeros(block, dtype=np.float32), x, np.zeros(block, dtype=np.float32)])
    
    out = np.zeros_like(xp)
    norm = np.zeros_like(xp)

    n_frames = 1 + (len(xp) - block) // hop
    for i in range(n_frames):
        s = i * hop
        frame = xp[s:s + block] *window
        spec = np.fft.rfft(frame) *gain
        y = np.fft.irfft(spec, n=block).astype(np.float32)
        out[s:s + block] += y*window
        norm[s:s + block] += window** 2

    norm[norm < 1e-8] = 1e-8
    out = out / norm
    return out[block:block + len(x)]


def apply_elc_curve(samples, fs, freqs_hz, gains_db, block=4096, hop=1024):
    t0 = time.perf_counter()
    if samples.ndim == 1:
        out = process(samples, fs, freqs_hz, gains_db, block, hop)
    else:
        out = np.stack([process(samples[:, c], fs, freqs_hz, gains_db, block, hop) for c in range(samples.shape[1])], axis=1)
    elapsed = time.perf_counter() - t0
    dur = samples.shape[0] / fs
    return out, elapsed, dur


# streamlit app here

left_pad, content_col, right_pad = st.columns([1, 3, 1])
with content_col:
    title_col, logo_col = st.columns([1, 3], vertical_alignment="center")
    with title_col:
        st.image("LOGO.png", width=300)
    with logo_col:
        st.title("What the Fletch?!")

# elc part 1
st.header("1. Measure hearing contours")

st.subheader("Threshold of hearing")
cols = st.columns(2)
for i, f in enumerate(TEST_FREQS):
    with cols[i % 2]:
        amp = st.number_input(f"{f} Hz amplitude", min_value=0.000001, max_value=0.3,
                               value=0.01, step=0.0001, format="%.4f", key=f"thr_{f}")
        tone, fs = tone_gen(f, amp=amp)
        st.audio(to_wav_bytes(tone, fs), format="audio/wav")
        if st.button(f"Record threshold {f} Hz", key=f"thrbtn_{f}"):
            st.session_state.threshold[f] = amp
        if f in st.session_state.threshold:
            st.caption(f"recorded: {st.session_state.threshold[f]:.4f}")

st.subheader("Equal-loudness contours")
level_choice = st.selectbox("Level to calibrate", list(LEVELS))
ref_amp = st.number_input("1 kHz reference amplitude", min_value=0.000001, max_value=5.0,
                           value=LEVELS[level_choice], step=0.000001, format="%.3f",
                           key=f"ref_{level_choice}")
ref_tone, fs = tone_gen(1000, amp=ref_amp)
st.audio(to_wav_bytes(ref_tone, fs), format="audio/wav")

st.session_state.levels.setdefault(level_choice, {})[1000] = ref_amp

mcols = st.columns(2)
for i, f in enumerate([f for f in TEST_FREQS if f != 1000]):
    with mcols[i % 2]:
        amp = st.number_input(f"Match {f} Hz to reference", min_value=0.000001, max_value=5.0,
                               value=ref_amp, step=0.000001, format="%.4f",
                               key=f"match_{level_choice}_{f}")
        tone, fs = tone_gen(f, amp=amp)
        st.audio(to_wav_bytes(tone, fs), format="audio/wav")
        if st.button(f"Record match {f} Hz", key=f"matchbtn_{level_choice}_{f}"):
            st.session_state.levels[level_choice][f] = amp
        if f in st.session_state.levels[level_choice]:
            rel = 20 * np.log10(st.session_state.levels[level_choice][f] / ref_amp)
            st.caption(f"recorded: {rel:+.1f} dB vs 1 kHz")

if st.session_state.threshold or st.session_state.levels:
    fig, ax = plt.subplots(figsize=(8, 4))
    if st.session_state.threshold:
        fs_ = sorted(st.session_state.threshold)
        ax.plot(fs_, [20 * np.log10(st.session_state.threshold[f]) for f in fs_],
                "o-", label="Threshold")
    for lvl, data in st.session_state.levels.items():
        if len(data) < 2:
            continue
        r = data[1000]
        fs_ = sorted(data)
        ax.plot(fs_, [20 * np.log10(data[f]) for f in fs_], "o-", label=lvl)
    ax.set_xscale("log")
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    st.pyplot(fig)

if st.button("Reset measurements"):
    st.session_state.threshold = {}
    st.session_state.levels = {}
    st.rerun()

st.divider()

# audio equalizer
st.header("2. ELC Audio Equalizer")

src = st.radio("Input", ["Demo file", "Upload file"], horizontal=True)
audio_bytes = None
if src == "Upload file":
    up = st.file_uploader("WAV / FLAC", type=["wav", "flac", "ogg"])
    if up:
        audio_bytes = up.read()
else:
        st.write("'Airplane Mode' is an original rap composition I made before learning about audio equalization!")
        with open(DEMO_FILE, "rb") as f:
            audio_bytes = f.read()
        st.caption(f"Using demo file: {DEMO_FILE} (currently unequalized)")

if audio_bytes:
    samples, fs = read_audio(audio_bytes)
    st.audio(audio_bytes)
    st.caption(f"{samples.shape[0]/fs:.2f}s, {fs} Hz, "
               f"{1 if samples.ndim==1 else samples.shape[1]} ch")

    ready_levels = [l for l, d in st.session_state.levels.items() if len(d) >= 2]
    if not ready_levels:
        st.warning("Record at least one equal-loudness level to build a curve.")
    else:
        curve_level = st.selectbox("ELC curve to apply", ready_levels)
        strength = st.number_input("Correction strength (%)", min_value=0, max_value=150,
                                    value=100, step=5)
        max_gain = 24
        block = 4096
        hop = block // 4

        data = st.session_state.levels[curve_level]
        ref = data[1000]
        f_meas = np.array(sorted(data), dtype=float)
        gains = np.array([20 * np.log10(data[f]) for f in sorted(data)])
        gains = np.clip(gains * (strength / 100.0), -max_gain, max_gain)

        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.plot(f_meas, gains, "o-")
        ax2.axhline(0, color="gray", linewidth=0.6)
        ax2.set_xscale("log")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Gain (dB)")
        ax2.grid(True, which="both", alpha=0.3)
        st.pyplot(fig2)

        if st.button("Process audio", type="primary"):
            out, elapsed, dur = apply_elc_curve(samples, fs, f_meas, gains, block, hop)
            st.success(f"Processed {dur:.2f}s in {elapsed*1000:.0f}ms")

            out_bytes = to_wav_bytes(out, fs, normalize=True)
            st.audio(out_bytes, format="audio/wav")
            st.download_button("Download WAV", out_bytes, "elc_equalized.wav", "audio/wav")