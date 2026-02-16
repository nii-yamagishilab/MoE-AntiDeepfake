import numpy as np
import librosa
from numpy import ndarray
import random
import scipy
import pysptk


#--------------------------- EXTRACTION ------------------------#

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


#--------------------------- MANIPULATIONS ------------------------#

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

def flatten_f0(f0_traj, strength=0.9, p=0.5):

    f0_mod = f0_traj.copy()
    voiced = ~np.isnan(f0_mod)

    f0_ref = np.nanmedian(f0_mod)
    mask = (np.random.rand(len(f0_mod)) < p) & voiced

    f0_mod[mask] = (
            (1.0 - strength) * f0_mod[mask]
            + strength * f0_ref
    )

    return f0_mod

def apply_f0_mod(f0_traj, mod='f0_shift', hop_s=0.016):
    if mod == 'f0_shift':
        seg_len = random.uniform(0.08, 0.16)  # 80–160 ms
        p_seg = random.uniform(0.4, 0.7)

        f0_traj_mod = f0_segment_shift(
            f0_traj,
            seg_len_s=seg_len,
            hop_s=hop_s,
            p_seg=p_seg
        )

    elif mod == 'f0_shuffle':
        seg_len = random.uniform(0.08, 0.16)
        p_shuffle = random.uniform(0.1, 0.3)

        f0_traj_mod = f0_segment_shuffle(
            f0_traj,
            seg_len_s=seg_len,
            hop_s=hop_s,
            p_shuffle=p_shuffle
        )

    elif mod == 'flat_pitch':
        f0_traj_mod = flatten_f0(f0_traj,
                                 strength=0.5,
                                 p=1)

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


