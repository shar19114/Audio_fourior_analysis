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
st.title("Interactive Fourier Transform Studio")
st.write("Upload an audio track to decompose it into its frequency components, filter it, and reconstruct the sound.")

# ---------------------------------------------------------
# 2. File Uploading Setup
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Upload Audio File (*.WAV, *.MP3, *.AAC)", type=['wav', 'mp3', 'aac'])

if uploaded_file is not None:
    st.markdown("---")
    st.subheader("1. Original Audio Analysis")
    
    # Load the audio using Librosa. 
    # sr=None tells Librosa to keep the file's exact original sample rate.
    with st.spinner("Loading audio file..."):
        y, sr = librosa.load(uploaded_file, sr=None)
    
    st.audio(uploaded_file, format='audio/wav')
    
    # Create the time axis for the original wave
    duration = len(y) / sr
    time_axis = np.linspace(0, duration, len(y))
    
    # Plot Original Waveform
    fig_orig, ax_orig = plt.subplots(figsize=(12, 3))
    ax_orig.plot(time_axis, y, color='blue', linewidth=0.5)
    ax_orig.set_title("Time-Domain: Original Waveform")
    ax_orig.set_xlabel("Time (seconds)")
    ax_orig.set_ylabel("Amplitude")
    ax_orig.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig_orig)
    plt.close(fig_orig) # Close the plot to free up memory

    # ---------------------------------------------------------
    # 3. Fourier Decomposition (Math in Action)
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("2. Frequency Decomposition Spectrum")
    
    N = len(y)
    
    with st.spinner("Computing Fast Fourier Transform..."):
        # Calculate the 1D discrete Fourier Transform
        Y_freq = np.fft.fft(y)
        # Calculate the corresponding frequencies for the x-axis
        frequencies = np.fft.fftfreq(N, 1/sr)
        
    # We only need the positive half of the frequencies to visualize the spectrum
    half_N = N // 2
    pos_freqs = frequencies[:half_N]
    pos_mags = np.abs(Y_freq)[:half_N]
    
    # Plot the Frequency Spectrum
    fig_spec, ax_spec = plt.subplots(figsize=(12, 3))
    ax_spec.plot(pos_freqs, pos_mags, color='purple', linewidth=0.5)
    ax_spec.set_title("Frequency-Domain: Magnitude Spectrum")
    ax_spec.set_xlabel("Frequency (Hz)")
    ax_spec.set_ylabel("Magnitude")
    ax_spec.set_xlim([0, sr/2]) # Limit x-axis to the Nyquist frequency
    ax_spec.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig_spec)
    plt.close(fig_spec)

    # ---------------------------------------------------------
    # 4. User Interface for Filtering Frequencies
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("3. Filter and Reconstruct")
    st.write("Use the sliders to define which frequencies to keep. Everything outside this range will be muted (set to zero).")
    
    # Create a layout with two columns for the sliders
    col1, col2 = st.columns(2)
    with col1:
        min_freq = st.slider("Minimum Frequency (Hz)", min_value=0.0, max_value=float(sr/2), value=0.0, step=10.0)
    with col2:
        max_freq = st.slider("Maximum Frequency (Hz)", min_value=0.0, max_value=float(sr/2), value=float(sr/2), step=10.0)
    
    # Apply the mathematical filter mask
    Y_filtered = Y_freq.copy()
    # Mask: Find frequencies that are strictly LESS than min_freq OR GREATER than max_freq
    freq_mask = (np.abs(frequencies) < min_freq) | (np.abs(frequencies) > max_freq)
    # Mute them by turning their amplitude to 0
    Y_filtered[freq_mask] = 0

    # ---------------------------------------------------------
    # 5. Inverse Transform and Error Calculation
    # ---------------------------------------------------------
    if st.button("Apply Filter & Reconstruct Audio", type="primary"):
        with st.spinner("Reconstructing sound wave..."):
            # Compute the Inverse Fast Fourier Transform (IFFT)
            # .real is used because tiny imaginary rounding errors can occur
            y_recon = np.fft.ifft(Y_filtered).real
            
            # Plot the Reconstructed Waveform
            fig_recon, ax_recon = plt.subplots(figsize=(12, 3))
            ax_recon.plot(time_axis, y_recon, color='green', linewidth=0.5)
            ax_recon.set_title(f"Reconstructed Waveform ({min_freq} Hz to {max_freq} Hz)")
            ax_recon.set_xlabel("Time (seconds)")
            ax_recon.set_ylabel("Amplitude")
            ax_recon.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig_recon)
            plt.close(fig_recon)
            
            # Save the new audio to a buffer and display the player
            buffer = io.BytesIO()
            sf.write(buffer, y_recon, sr, format='WAV')
            st.audio(buffer, format='audio/wav')
            
            # Calculate the Mean Squared Error (MSE)
            mse_error = np.mean((y - y_recon)**2)
            
            st.markdown("### Error Analysis")
            st.write(f"**Mean Squared Error (MSE):** `{mse_error:.10f}`")
            
            if mse_error < 1e-10:
                 st.success("The error is near zero! Because you kept all frequencies, the mathematics perfectly restored your original sound file.")
            else:
                 st.warning("The error is greater than zero. Because you removed certain frequency components, the reconstructed wave no longer perfectly matches the original wave.")