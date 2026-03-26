import numpy as np
import librosa
from numpy import ndarray
import random
import scipy
import pysptk
from scipy.fft import ifft
import torch
import torchaudio
import math

################################################################################################################
#------------------------------------------------- EXTRACTION -------------------------------------------------#
################################################################################################################


#____________________________FO Trajectory______________________#

def extract_f0(waveform,
               fs, win_size, hop_size,
               fmin: int = 70, fmax: int = 400) -> ndarray:
    """
    input
    -----
      waveform: np.array, (c, n) or (n), where c is the #. channel
      fs: int, sampling rate
      win_size: int, window size
      hop_size: int, hop size
      fmin: int, minimum of F0
      fmax: int, maximum of F0

    output
    ------
      f0_freqs: f0 frequency (c, n_frames), or (n_frames)
    """
    x = np.copy(waveform)
    x = librosa.effects.preemphasis(x)
    autocorr_win_len = win_size // 2

    f0_freqs, _, _ = librosa.pyin(
        x, sr=fs,
        frame_length=win_size, hop_length=hop_size, center=True,
        win_length=autocorr_win_len, fmin=fmin, fmax=fmax, n_thresholds=7
    )

    # pyworld_analysis
    # x = np.array(x, dtype=np.float64)
    # _f0, t = pw.dio(x, sr)    # raw pitch extractor
    # f0_freqs = pw.stonemask(x, _f0, t, sr)  # pitch refinement

    return f0_freqs


#______________________Line Spectral Frequencies____________________#

def get_lsf(waveform, order=10, frame_length=400, hop_length=160):
    """
    Computes Line Spectral Frequencies (LSF) from a waveform.

    Args:
        waveform (np.ndarray): Input signal of shape (L,).
        order (int): The order of the LPC analysis.
        frame_length (int): Length of each windowed frame.
        hop_length (int): Number of samples between successive frames.

    Returns:
        np.ndarray: LSF coefficients of shape (NumFrames, order).
    """
    # Ensure waveform is float64 (standard for SPTK C-wrappers)
    waveform = waveform.astype(np.float64)

    # 1. Framing
    # Instead of util.frame, we'll use a list comprehension for reliability
    frames = [waveform[i:i + frame_length] for i in range(0, len(waveform) - frame_length, hop_length)]
    frames = np.stack(frames)

    # 2. Scipy Windowing
    # 'blackman' is standard, but 'hamming' or 'hann' are also common
    window = scipy.signal.get_window('blackman', frame_length, fftbins=False)
    windowed_frames = frames * window

    # 3. LPC Analysis
    # Note: lpc returns (order + 1) coefficients
    lpc_coeffs = pysptk.lpc(windowed_frames, order)

    # 4. Convert LPC to LSF
    try:
        # 2. Attempt conversion
        lsf = pysptk.lpc2lsp(lpc_coeffs)
        return lsf
    except RuntimeError:
        # 3. Fallback: If it fails, the filter was likely unstable.
        # We can 'stabilize' it by slightly moving roots inside the circle
        # (Often called bandwidth expansion)
        gamma = 0.9
        lpc_stable = lpc_coeffs * (gamma ** np.arange(lpc_coeffs.shape[1]))
        return pysptk.lpc2lsp(lpc_stable)

    return lsf

def lsf_extraction(wav, stft_params, lsf_order=20):
    """ just wrapper over extrac_f0
    """
    lsf_traj = get_lsf(
        wav + 1e-6 * np.random.randn(len(wav)),
        order=lsf_order,
        frame_length=stft_params.win_size,
        hop_length=stft_params.hop_size
    )

    return lsf_traj


#___________________________Phase Features_________________________#

def extract_phase_features(y, sr, win_length=512, hop_length=256, n_fft=None):
    """
    Extracts a comprehensive suite of phase-domain representations from a speech signal.

    This function computes the instantaneous phase, unwrapped phase, Group Delay (GD).

    Args:
        y (np.ndarray):
            The input audio time-series (1D array).
        sr (int):
            The sampling rate of the input signal.
        win_length (int, optional):
            The number of samples in an analysis window. Defaults to 512
        hop_length (int, optional):
            The number of samples between successive frames. Defaults to 256.
        n_fft (int, optional):
            The number of FFT bins. Larger values provide higher frequency
            resolution. Defaults to win_length

    Returns:
        dict: A dictionary containing:
            - 'phase': Raw wrapped phase in range [-pi, pi], (T, n_fft/2+1)
            - 'unwrapped': Phase with 2pi discontinuities removed, (T, n_fft/2+1)
            - 'group_delay': tandard group delay (time-weighted method), (T, n_fft/2+1)
    """
    # 1. Standard STFT and Phase
    if n_fft is None:
        n_fft = win_length
    stft = librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    # 2. Unwrapped Phase
    # Unwrapping across the frequency axis (axis=0)
    unwrapped_phase = np.unwrap(phase, axis=0)

    # Compute the derivative along the frequency axis
    # np.diff calculates phase[f+1] - phase[f]
    d_phi = np.diff(unwrapped_phase, axis=0)

    # 3. Define the frequency step (delta_omega)
    # delta_omega = 2 * pi * (df / sr) where df is sr/n_fft
    # This simplifies to 2 * pi / n_fft
    d_omega = 2 * np.pi / n_fft

    # 4. Calculate Group Delay: GD = -d_phi / d_omega
    # Result is in samples. Divide by 'sr' to get seconds.
    group_delay_samples = -(d_phi / d_omega)

    # Since np.diff reduces the dimension by 1, we pad it back
    # to match the original STFT shape
    group_delay = np.pad(group_delay_samples, ((0, 1), (0, 0)), mode='edge')

    return {
        "phase": phase.T,
        "unwrapped": unwrapped_phase.T,
        "group_delay": group_delay.T,
    }

def phase_extraction(wav, key, stft_params):
    # the extraction should be done after manipulation of the minimum phase signal.
    # extract phase representation
    x_phase = extract_phase_features(wav, stft_params.sample_rate, stft_params.win_size, stft_params.hop_size)
    return x_phase[key]


#___________________________Energy Inconsistency_________________________#

def apply_db(spec):
    spec_db = torchaudio.functional.amplitude_to_DB(
        spec,
        multiplier=10.0,
        amin=1e-10,
        db_multiplier=0
    )
    return spec_db

def spec_extraction(y, stft_params):
    spec_transform = torchaudio.transforms.Spectrogram(
        n_fft=stft_params.fft_size,
        hop_length=stft_params.hop_size,
        power=2
    )
    spec = spec_transform(torch.Tensor(y).unsqueeze(0))
    spec = spec.squeeze(0)
    feat = apply_db(spec)

    return feat

def energy_extraction(y, stft_params):
    frames = y.unfold(0, stft_params.fft_size, stft_params.hop_size)
    energy = (frames ** 2).mean(dim=1)
    # delta = torch.diff(energy, prepend=energy[:1])
    # feat = torch.stack([energy, delta], dim=0)

    return energy



################################################################################################################
#------------------------------------------------ MANIPULATIONS ------------------------------------------------#
################################################################################################################


#____________________________FO Inconsistency______________________#

def f0_segment_shift(f0_traj, seg_len_s, hop_s, p_seg):
    f0_mod = f0_traj.copy()
    voiced = f0_mod > 0

    seg_len_frames = int(seg_len_s / hop_s)
    i = 0

    while i < len(f0_mod):
        if not voiced[i]:
            i += 1
            continue

        # find voiced segment f0_mod[seg_start : seg_end]
        seg_start = i
        while i < len(f0_mod) and voiced[i]:
            i += 1
        seg_end = i

        # chunk the voiced segment in pseudo-phonemes, shift some
        j = seg_start
        while j < seg_end:
            seg_j_end = min(j + seg_len_frames, seg_end)

            if np.random.rand() < p_seg:
                shift = random.choice([
                    random.uniform(0.87, 0.92),
                    random.uniform(1.08, 1.3)
                ])
                f0_mod[j:seg_j_end] *= shift

            j = seg_j_end

    return f0_mod

def f0_segment_shuffle(f0_traj, seg_len_s, hop_s, p_shuffle):
    f0_mod = f0_traj.copy()
    voiced = f0_mod > 0

    seg_len_frames = int(seg_len_s / hop_s)

    segments = []
    indices = []

    i = 0
    while i < len(f0_mod):
        if not voiced[i]:
            i += 1
            continue

        # find voiced segment f0_mod[seg_start : seg_end]
        seg_start = i
        while i < len(f0_mod) and voiced[i]:
            i += 1
        seg_end = i

        # chunk the voiced segment in pseudo-phonemes
        j = seg_start
        while j < seg_end:
            seg_j_end = min(j + seg_len_frames, seg_end)

            segments.append(f0_mod[j:seg_j_end].copy())
            indices.append((j, seg_j_end))
            j = seg_j_end

    if len(segments) < 2:
        return f0_mod

    # partial shuffle
    perm = list(range(len(segments)))
    for i in range(len(perm)):
        if random.random() < p_shuffle:
            j = random.randrange(len(perm))
            perm[i], perm[j] = perm[j], perm[i]

    # re-insert
    for out_i, src_i in enumerate(perm):
        start, end = indices[out_i]
        seg = segments[src_i]
        L = min(end - start, len(seg))
        f0_mod[start:start + L] = seg[:L]

    return f0_mod

def flatten_f0(f0_traj, strength=0.5, dyn_strength=0.0):
    f0_mod = f0_traj.copy()
    voiced = f0_mod > 0

    # global range compression
    f0_ref = np.median(f0_mod)

    f0_mod[voiced] = (
        (1.0 - strength) * f0_mod[voiced]
        + strength * f0_ref
    )

    # damp slopes too
    if dyn_strength > 0:
        f0_smooth = np.copy(f0_mod)

        # simple 3-frame local average
        f0_smooth[1:-1] = (
            f0_mod[:-2] + f0_mod[1:-1] + f0_mod[2:]
        ) / 3.0

        # blend original with smoothed
        f0_mod[voiced] = (
            (1.0 - dyn_strength) * f0_mod[voiced]
            + dyn_strength * f0_smooth[voiced]
        )

    return f0_mod

def apply_f0_mod(f0_traj, mod='f0_incons', hop_s=0.016):
    if mod == 'f0_incons':
        seg_len = random.uniform(0.08, 0.16)

        # Random params for shift
        p_seg = random.uniform(0.4, 0.7)
        f0_traj_mod = f0_segment_shift(
            f0_traj,
            seg_len_s=seg_len,
            hop_s=hop_s,
            p_seg=p_seg
        )
        # Random params for shuffle
        p_shuffle = random.uniform(0.2, 0.5)
        f0_traj_mod = f0_segment_shuffle(
            f0_traj_mod,  # apply on shifted result
            seg_len_s=seg_len,
            hop_s=hop_s,
            p_shuffle=p_shuffle
        )

    elif mod == 'flat_pitch':
        f0_traj_mod = flatten_f0(
            f0_traj,
            strength=0.4,
            dyn_strength=0.4
        )

    # elif mod == 'f0_shift':
    #     seg_len = random.uniform(0.08, 0.16)  # 80–160 ms
    #     p_seg = random.uniform(0.4, 0.7)
    #
    #     f0_traj_mod = f0_segment_shift(
    #         f0_traj,
    #         seg_len_s=seg_len,
    #         hop_s=hop_s,
    #         p_seg=p_seg
    #     )
    #
    # elif mod == 'f0_shuffle':
    #     seg_len = random.uniform(0.08, 0.16)
    #     p_shuffle = random.uniform(0.1, 0.3)
    #
    #     f0_traj_mod = f0_segment_shuffle(
    #         f0_traj,
    #         seg_len_s=seg_len,
    #         hop_s=hop_s,
    #         p_shuffle=p_shuffle
    #     )

    else:
        f0_traj_mod = None
        print("Does not recognize mod: ", mod)

    return f0_traj_mod


#____________________________Temporal Oversmoothing______________________#

def apply_lsf_mod(lsf, sigma=2.0):
    """
    Simulates oversmoothing on LSFs along the temporal axis.

    Args:
        lsf (np.ndarray): LSF array of shape (NumFrames, Order).
        sigma (float): Smoothing strength. Higher values = more oversmoothing.

    Returns:
        np.ndarray: The smoothed LSF coefficients.
    """
    # We apply the filter along axis 0 (the temporal/frame axis)
    # This blurs the transition of each individual LSF over time
    smoothed_lsf = scipy.ndimage.gaussian_filter1d(lsf, sigma=sigma, axis=0)

    # Ensure the LSF ordering property is maintained (stability)
    # LSFs must be strictly increasing: 0 < w1 < w2 ... < pi
    smoothed_lsf = np.sort(smoothed_lsf, axis=1)

    return smoothed_lsf

def lsf_manipulation_smoothing(lsf_traj):
    return apply_lsf_mod(lsf_traj)


#_______________________________Minimum Phase____________________________#

def phase_manipulation(wav, key, stft_params):
    ## The order here is different,
    ## the signal is manipulated, and the phase is extracted
    ## Hence, we have to feed wav as input

    # manipulation to minimum phase signal
    x_mod = stft_to_min_phase(wav, stft_params.sample_rate, stft_params.win_size, stft_params.hop_size)

    # extract phase representation
    x_mod_phase = extract_phase_features(x_mod, stft_params.sample_rate, stft_params.win_size, stft_params.hop_size)
    return x_mod_phase[key]

def stft_to_min_phase(y, sr, win_length=512, hop_length=256, n_fft=None):
    """ y_min_phase = stft_to_min_phase(y, sr, win_length=512, hop_length=256, n_fft=None)

    input
    -----
      y: signal, np.array
      sr: int, sampling rate (Hz)
      win_length: int, window length, (512)
      hop_length: int, hope length, (256)
      n_fft: int, number of fft points

    output
    ------
      y_min_phase: signal, minimum phased, np.array, (T, n_fft//2+1)
    """
    if n_fft is None:
        n_fft = win_length
        # 1. Get the Magnitude Spectrum (discarding original phase)
    stft = librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    mag = np.abs(stft)

    # Initialize a buffer for the new minimum phase STFT
    min_phase_stft = np.zeros_like(stft, dtype=complex)

    # 2. Create the full symmetric spectrum (Mirroring)
    # Concatenate the magnitude bins to represent a full N-point FFT
    full_mag = np.vstack([mag, mag[-2:0:-1, :]])  # Shape: (N, T)

    # 3. Batch Real Cepstrum
    # axis=0 applies the transform across the frequency bins for every frame
    log_mag = np.log(full_mag + 1e-12)
    real_cepstrum = ifft(log_mag, axis=0).real

    # 4. Create a Vectorized Lifter (Window)
    n = full_mag.shape[0]
    window = np.zeros((n, 1))  # Column vector for broadcasting
    window[0] = 1
    window[n // 2] = 1
    window[1:n // 2] = 2

    # 5. Apply Window and Transform back (all frames at once)
    # Broadcasting across time axis
    min_phase_cepstrum = real_cepstrum * window
    complex_spec = np.exp(scipy.fft.fft(min_phase_cepstrum, axis=0))

    # 6. Extract positive frequencies and reconstruct
    min_phase_stft = complex_spec[:n_fft // 2 + 1, :]

    # Reconstruct the time-domain signal
    y_min_phase = librosa.istft(min_phase_stft, hop_length=hop_length)

    return y_min_phase.T


#___________________________Energy Inconsistency_________________________#

def spec_manipulation(y, stft_params):
    y = torch.tensor(y).float()
    n_samples = y.shape[-1] if y.dim() > 1 else y.shape[0]

    # Select time frames
    spec_dummy = torchaudio.transforms.Spectrogram(
        n_fft=stft_params.fft_size,
        hop_length=stft_params.hop_size,
        power=2
    )(y.unsqueeze(0))  # just to get n_time
    n_time = spec_dummy.shape[-1]
    # width = random.randint(10, 50)
    width = random.randint(6, 30)

    start_idx = random.randint(0, max(n_time - width, 0))
    end_idx = start_idx + width
    end_idx = min(end_idx, n_time)

    hop_length = stft_params.hop_size
    fft_size = stft_params.fft_size

    start_sample = start_idx * hop_length
    end_sample = min(n_samples, (end_idx - 1) * hop_length + fft_size)

    y_zeroed = y.clone()

    attenuation = random.uniform(0.0, math.sqrt(0.5))
    y_zeroed[start_sample:end_sample] *= attenuation

    spec_transform = torchaudio.transforms.Spectrogram(
        n_fft=fft_size,
        hop_length=hop_length,
        power=2
    )
    spec = spec_transform(y_zeroed.unsqueeze(0)).squeeze(0)
    feat = apply_db(spec)

    return feat

def energy_manipulation(y, stft_params):
    y = torch.tensor(y).float()

    frames = y.unfold(0, stft_params.fft_size, stft_params.hop_size)
    energy = (frames ** 2).mean(dim=1)

    n_time = energy.shape[0]
    width = random.randint(6, 30)

    start_idx = random.randint(0, max(n_time - width, 0))
    end_idx = min(start_idx + width, n_time)

    energy_mod = energy.clone()

    attenuation = random.uniform(0.0, 0.7)
    energy_mod[start_idx:end_idx] *= attenuation

    # delta = torch.diff(energy_mod, prepend=energy_mod[:1])
    # feat = torch.stack([energy_mod, delta], dim=0)

    return energy_mod



# import matplotlib.pyplot as plt
#
# plt.imshow(mask_old.squeeze(), origin='lower', aspect='auto')
# plt.show()
# plt.imshow(mask_new.squeeze(), origin='lower', aspect='auto')
# plt.show()
#
# plt.imshow(mag, origin='lower', aspect='auto')
# plt.show()
# plt.imshow(mag_new.squeeze(), origin='lower', aspect='auto')
# plt.show()
#
# plt.imshow((mag-mag_new.squeeze()), origin='lower', aspect='auto')
# plt.show()
#
# plt.imshow(torchaudio.functional.amplitude_to_DB(mag, multiplier=20.0, amin=1e-10, db_multiplier=1.0), origin='lower', aspect='auto')
# plt.show()
# plt.imshow(torchaudio.functional.amplitude_to_DB(mag_new.squeeze(), multiplier=20.0, amin=1e-10, db_multiplier=1.0), origin='lower', aspect='auto')
# plt.show()
#
# ciao = torchaudio.functional.amplitude_to_DB(mag, multiplier=20.0, amin=1e-10, db_multiplier=1.0) - torchaudio.functional.amplitude_to_DB(mag_new.squeeze(), multiplier=20.0, amin=1e-10, db_multiplier=1.0)
# plt.imshow(ciao, origin='lower', aspect='auto')
# plt.show()