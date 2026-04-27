import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import io

# ---------------------------------------------------------
# 1. Page Configuration and Title
# ---------------------------------------------------------
st.set_page_config(page_title="Fourier Audio Studio", layout="wide")
st.title("Interactive Fourier Transform Studio (With Equalizer)")
st.write("Upload an audio track, analyze its frequency spectrum, and selectively boost or suppress specific frequency bands.")

# ---------------------------------------------------------
# 2. File Uploading Setup
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Upload Audio File (*.WAV, *.MP3, *.AAC)", type=['wav', 'mp3', 'aac'])

if uploaded_file is not None:
    st.markdown("---")
    st.subheader("1. Original Audio Analysis")
    
    with st.spinner("Loading audio file..."):
        y, sr = librosa.load(uploaded_file, sr=None)
    
    st.audio(uploaded_file, format='audio/wav')
    
    duration = len(y) / sr
    time_axis = np.linspace(0, duration, len(y))
    
    fig_orig, ax_orig = plt.subplots(figsize=(12, 3))
    ax_orig.plot(time_axis, y, color='blue', linewidth=0.5)
    ax_orig.set_title("Time-Domain: Original Waveform")
    ax_orig.set_xlabel("Time (seconds)")
    ax_orig.set_ylabel("Amplitude")
    ax_orig.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig_orig)
    plt.close(fig_orig)

    # ---------------------------------------------------------
    # 3. Fourier Decomposition
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("2. Frequency Decomposition Spectrum")
    
    N = len(y)
    
    with st.spinner("Computing Fast Fourier Transform..."):
        Y_freq = np.fft.fft(y)
        frequencies = np.fft.fftfreq(N, 1/sr)
        
    half_N = N // 2
    pos_freqs = frequencies[:half_N]
    pos_mags = np.abs(Y_freq)[:half_N]
    
    fig_spec, ax_spec = plt.subplots(figsize=(12, 3))
    ax_spec.plot(pos_freqs, pos_mags, color='purple', linewidth=0.5)
    ax_spec.set_title("Frequency-Domain: Magnitude Spectrum")
    ax_spec.set_xlabel("Frequency (Hz)")
    ax_spec.set_ylabel("Magnitude")
    ax_spec.set_xlim([0, sr/2])
    ax_spec.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig_spec)
    plt.close(fig_spec)

    # ---------------------------------------------------------
    # 4. Digital Equalizer: Modify Frequencies
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("3. Digital Equalizer: Modify Frequency Amplitude")
    st.write("Select a target frequency range and use the multiplier to either boost (increase) or suppress (decrease) its volume.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        min_freq = st.slider("Target Min Frequency (Hz)", min_value=0.0, max_value=float(sr/2), value=0.0, step=10.0)
    with col2:
        max_freq = st.slider("Target Max Frequency (Hz)", min_value=0.0, max_value=float(sr/2), value=float(sr/2), step=10.0)
    with col3:
        # New Gain Slider: 1.0 is no change. 0.0 is mute. 5.0 is massive volume boost.
        gain = st.slider("Amplitude Multiplier (Gain)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
    
    # Create a copy of the original frequency array so we do not overwrite the raw data
    Y_modified = Y_freq.copy()
    
    # Mask: Find frequencies that are INSIDE the targeted range
    freq_mask = (np.abs(frequencies) >= min_freq) & (np.abs(frequencies) <= max_freq)
    
    # Mathematical Application: Multiply the targeted frequencies by the gain slider value
    Y_modified[freq_mask] = Y_modified[freq_mask] * gain

    # ---------------------------------------------------------
    # 5. Inverse Transform and Error Calculation
    # ---------------------------------------------------------
    if st.button("Apply Equalizer & Reconstruct Audio", type="primary"):
        with st.spinner("Reconstructing sound wave..."):
            
            # Compute the Inverse Fast Fourier Transform (IFFT)
            y_recon = np.fft.ifft(Y_modified).real
            
            fig_recon, ax_recon = plt.subplots(figsize=(12, 3))
            ax_recon.plot(time_axis, y_recon, color='green', linewidth=0.5)
            ax_recon.set_title(f"Reconstructed Waveform (Multiplier: {gain}x applied to {min_freq}Hz - {max_freq}Hz)")
            ax_recon.set_xlabel("Time (seconds)")
            ax_recon.set_ylabel("Amplitude")
            ax_recon.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig_recon)
            plt.close(fig_recon)
            
            buffer = io.BytesIO()
            sf.write(buffer, y_recon, sr, format='WAV')
            st.audio(buffer, format='audio/wav')
            
            mse_error = np.mean((y - y_recon)**2)
            
            st.markdown("### Error Analysis")
            st.write(f"**Mean Squared Error (MSE):** `{mse_error:.10f}`")
            
            if gain == 1.0:
                 st.success("The error is near zero because a multiplier of 1.0 leaves the original signal perfectly unchanged!")
            elif gain > 1.0:
                 st.info("The error is greater than zero because you ADDED energy to the original sound wave (amplification).")
            elif gain < 1.0:
                 st.warning("The error is greater than zero because you REMOVED energy from the original sound wave (suppression).")