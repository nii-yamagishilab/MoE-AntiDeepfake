import numpy as np
import librosa
from numpy import ndarray
import random


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
                    random.uniform(0.895, 0.945),
                    random.uniform(1.065, 1.15)
                    # random.uniform(0.10, 0.20),
                    # random.uniform(1.80, 1.90)
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

    else:
        f0_traj_mod = None
        print("Does not recognize mod: ", mod)

    return f0_traj_mod


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


