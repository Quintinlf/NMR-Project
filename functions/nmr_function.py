import numpy as np
import pandas as pd
import os
from urllib.parse import urlparse
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fftshift, fft, fftfreq

def load_fid_and_preview(source, delimiter='\t', skip_header=1,
                         columns=('X', 'Real', 'Imaginary'),
                         name=None, preview_rows=5):
    """
    Load JEOL ASCII FID (3 columns) from URL or local path, print head, and return (df, name).
    """
    try:
        data = np.genfromtxt(source, delimiter=delimiter, skip_header=skip_header)
    except Exception as e:
        raise RuntimeError(f"Failed to read data from {source}: {e}")

    if data is None or data.size == 0:
        raise ValueError("Loaded data is empty.")

    # Ensure 2D shape and exactly 3 columns expected by this notebook
    if data.ndim == 1:
        raise ValueError("File appears to have 1 column; expected 3 columns (X, Real, Imaginary).")
    if data.shape[1] < 3:
        raise ValueError(f"Found {data.shape[1]} columns; expected 3.")
    if data.shape[1] > 3:
        data = data[:, :3]  # take first three columns

    df = pd.DataFrame(data, columns=list(columns))

    if name is None:
        # Infer a friendly name from the source path/URL
        path = str(source)
        base = os.path.basename(urlparse(path).path) if (path.startswith("http://") or path.startswith("https://")) else os.path.basename(path)
        name = os.path.splitext(base)[0].replace('%20', ' ')

    print(df.head(preview_rows))
    return df, name

def plot_fid(data, title=None, xcol=0, ycol=1, xlabel="Seconds(s)", ylabel="Abundance", invert_x=False, ax=None, show=True):
    """
    Plot time-domain FID from a DataFrame or NumPy array.
    Returns (fig, ax). If show=False, the figure is not shown.
    """
    # Normalize input to NumPy
    arr = data.to_numpy() if hasattr(data, "to_numpy") else np.asarray(data)

    # Validate
    if arr is None or arr.size == 0 or arr.ndim < 2 or max(xcol, ycol) >= arr.shape[1]:
        print("No valid data found in the file.")
        return None, None

    # Create axes if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    # Plot
    ax.plot(arr[:, xcol], arr[:, ycol])
    ax.set_title(title or "")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if invert_x:
        ax.invert_xaxis()

    if show:
        plt.show()

    return fig, ax

# ...existing code...
def plot_full_and_zoom_with_peaks(
    frequencies,
    magnitudes,
    title,
    spectrometer_freq=399.78219838,
    ppm_range=None,
    identify_functional_groups=None,
    ppm_shifts=None,
    identified_groups=None,
    min_distance_hz=7.0,
    height_frac=0.10,
    prominence_frac=0.05,
    ax_full=None,
    ax_zoom=None,
    show=True
):
    """
    Flexible spectrum plotter:
      - Accepts full (negative..positive) or already-positive frequency axis (Hz)
      - Detects peaks
      - Optional functional group tagging
      - Returns dict of results
    """
    freqs = np.asarray(frequencies)
    mags = np.asarray(magnitudes)

    if freqs.size == 0 or mags.size == 0:
        print("Empty spectrum.")
        return {}

    # Determine if full spectrum (has negatives) or only positive
    is_full = (freqs.min() < 0) and (freqs.max() > 0)

    if is_full:
        # Center zero with fftshift
        freqs_shifted = fftshift(freqs)
        mags_shifted = fftshift(mags)
    else:
        # Already positive-only
        freqs_shifted = freqs
        mags_shifted = mags

    # Compute ppm axis
    ppm_axis = freqs_shifted / spectrometer_freq

    # Infer ppm_range if not supplied
    if ppm_range is None:
        ppm_range = float(ppm_axis.max() - ppm_axis.min())
        # For positive-only spectra typical 1H window ~12 ppm; clamp if too large
        if ppm_range <= 0 or ppm_range > 20:
            ppm_range = 12.0

    # Peak detection thresholds
    max_int = float(mags_shifted.max())
    if max_int == 0:
        print("All-zero intensity.")
        return {}

    height = height_frac * max_int
    prominence = prominence_frac * max_int

    n_points = len(mags_shifted)
    # Approximate Hz per point from ppm window
    hz_per_point = (ppm_range * spectrometer_freq) / n_points
    min_distance_pts = max(1, int(min_distance_hz / hz_per_point))

    from scipy.signal import find_peaks
    peaks, properties = find_peaks(
        mags_shifted,
        height=height,
        distance=min_distance_pts,
        prominence=prominence
    )

    # Functional group identification (expects ppm_axis)
    if identified_groups is None and callable(identify_functional_groups) and ppm_shifts is not None:
        identified_groups = identify_functional_groups(ppm_axis, mags_shifted, ppm_shifts)
    identified_groups = identified_groups or []

    # Zoom window
    if identified_groups:
        peak_ppms = [p for p, _ in identified_groups]
        x_min = min(peak_ppms) - 0.1
        x_max = max(peak_ppms) + 0.1
        print("Identified functional groups:")
        for p, g in identified_groups:
            print(f"The Peak at {p:.2f} ppm corresponds to a {g}")
        print(f"Graph automatically zoomed to range: {x_min:.2f}–{x_max:.2f} ppm")
    else:
        # Default 0–ppm_range (assuming downfield to upfield)
        x_min, x_max = 0.0, float(ppm_range)
        # Normalize ordering (largest ppm on left)
        if x_max < x_min:
            x_min, x_max = x_max, x_min
        print("No functional groups identified.")

    created_fig = False
    if ax_full is None or ax_zoom is None:
        fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        created_fig = True
    else:
        fig = ax_full.figure

    # Full plot
    ax_full.plot(ppm_axis, mags_shifted, label="Spectrum (FFT)")
    if peaks.size:
        ax_full.plot(ppm_axis[peaks], mags_shifted[peaks], "x", label="Detected Peaks")
    ax_full.invert_xaxis()
    # Show typical 1H window if makes sense
    left_lim = max(ppm_axis.max(), x_max) if is_full else (ppm_axis.max())
    right_lim = min(ppm_axis.min(), x_min) if is_full else (ppm_axis.min())
    # Fallback to (ppm_range, 0) if positive-only
    if not is_full:
        ax_full.set_xlim(min(x_max, ppm_range), 0)
    ax_full.set_xlabel("Chemical Shift (ppm)")
    ax_full.set_ylabel("Intensity (A.U.)")
    ax_full.set_title(title)
    ax_full.legend()

    # Zoom plot
    ax_zoom.plot(ppm_axis, mags_shifted, label="Spectrum (FFT)")
    if peaks.size:
        ax_zoom.plot(ppm_axis[peaks], mags_shifted[peaks], "x", label="Detected Peaks")
    ax_zoom.invert_xaxis()
    ax_zoom.set_xlim(x_max, x_min)
    ax_zoom.set_xlabel("Chemical Shift (ppm)")
    ax_zoom.set_title(f"{title} (zoom)")
    ax_zoom.legend()

    if show and created_fig:
        plt.tight_layout()
        plt.show()

    return {
        "fig": fig,
        "axes": (ax_full, ax_zoom),
        "ppm_axis": ppm_axis,
        "intensity": mags_shifted,
        "peaks": peaks,
        "properties": properties,
        "identified_groups": identified_groups,
    }

def compute_fft_spectrum(fid_array, time_col=0, real_col=1, window=None, zero_fill=None):
    """
    Compute FFT spectrum from FID array.

    Args:
        fid_array (ndarray): FID data with time in time_col and real signal in real_col.
        time_col (int): Column index for time (seconds).
        real_col (int): Column index for real FID values.
        window (str|None): Optional apodization: 'exp', 'hamming', 'hann', etc.
        zero_fill (int|None): If set and > current length, zero fill to this length (next power of 2 recommended).

    Returns:
        dict with keys:
            real_part
            dt
            n
            frequencies (Hz, signed)
            magnitude (abs FFT, not scaled)
            positive_frequencies (Hz >=0)
            positive_magnitude
    """
    real_part = fid_array[:, real_col].astype(float)
    time_axis = fid_array[:, time_col].astype(float)
    dt = time_axis[1] - time_axis[0]
    n = len(real_part)

    # Optional apodization
    if window:
        w = None
        if window == "exp":
            # simple exponential decay
            t = np.arange(n) * dt
            lw = 1.0  # line‑broadening factor (Hz) tweak if desired
            w = np.exp(-lw * t * np.pi)
        elif window == "hamming":
            w = np.hamming(n)
        elif window == "hann":
            w = np.hanning(n)
        if w is not None:
            real_part = real_part * w

    # Optional zero fill
    if zero_fill and zero_fill > n:
        zf_n = int(zero_fill)
        pad = zf_n - n
        real_part = np.pad(real_part, (0, pad), mode='constant')
        n = len(real_part)

    fft_res = fft(real_part)
    freqs = fftfreq(n, d=dt)
    mag = np.abs(fft_res)

    mask = freqs >= 0
    return {
        "real_part": real_part,
        "dt": dt,
        "n": n,
        "frequencies": freqs,
        "magnitude": mag,
        "positive_frequencies": freqs[mask],
        "positive_magnitude": mag[mask],
    }
# ...existing code...


def identify_functional_groups(positive_frequencies, positive_magnitudes, ppm_shifts):
    """
    Identify functional groups based on peak positions in the spectrum.

    args: 
        positive_frequencies (np.ndarray): Frequencies in ppm.
        positive_magnitudes (np.ndarray): Magnitudes of the FFT result.
        ppm_shifts (dict): Dictionary mapping functional groups to their ppm ranges.
    """
    # Find peaks in the spectrum
    peaks, _ = find_peaks(positive_magnitudes, height=0.1 * max(positive_magnitudes))  # Adjust height threshold as needed
    peak_positions = positive_frequencies[peaks]

    # Map peaks to functional groups
    #place holder for the identified groups
    identified_groups = []
    for peak in peak_positions:
        for group, ppm_range in ppm_shifts.items():
            # Parse the ppm range
            # this takes in the ppm ranges as a float and then makes a list of them (map)
            ppm_min, ppm_max = map(float, ppm_range.replace("ppm", "").split("-"))
            if ppm_min <= peak <= ppm_max:
                identified_groups.append((peak, group))
                break

    return identified_groups


def auto_zoom_functional_groups(positive_frequencies, positive_magnitude, identified_groups, ppm_shifts, buffer=0.01):
    """
    For each functional group with detected peaks, plot a zoomed-in graph
    covering the range from the lowest to highest detected peak in that group,
    with a buffer added to both sides.
    Args:
        positive_frequencies (np.ndarray): Frequencies in ppm.
        positive_magnitude (np.ndarray): Magnitudes of the FFT result.
        identified_groups (list): List of tuples with peak positions and their corresponding functional groups.
        ppm_shifts (dict): Dictionary mapping functional groups to their ppm ranges.
        buffer (float): Buffer to add around the zoomed region in ppm.
    """
    # Group peaks by functional group
    group_peaks = {}
    for peak_ppm, group in identified_groups:
        group_peaks.setdefault(group, []).append(peak_ppm)

    for group, peaks in group_peaks.items():
        ppm_range_str = ppm_shifts.get(group)
        if not ppm_range_str:
            continue
        ppm_min, ppm_max = map(float, ppm_range_str.replace("ppm", "").split("-"))
        # Only consider peaks within the defined ppm range for the group
        peaks_in_range = [ppm for ppm in peaks if ppm_min <= ppm <= ppm_max]
        if not peaks_in_range:
            continue
        # Define zoom window: min and max peak in group ± buffer
        zoom_min = min(peaks_in_range) - buffer
        zoom_max = max(peaks_in_range) + buffer
        # Mask for the zoomed region
        ppm_axis = positive_frequencies / 399.78219838
        mask = (ppm_axis >= zoom_min) & (ppm_axis <= zoom_max)
        # Plot
        plt.figure(figsize=(8, 4))
        plt.plot(ppm_axis[mask], positive_magnitude[mask], label=f"{group} ({zoom_min:.2f}-{zoom_max:.2f} ppm)")
        plt.gca().invert_xaxis()
        plt.title(f"Zoomed Spectrum: {group}")
        plt.xlabel("Chemical Shift (ppm)")
        plt.ylabel("Intensity (A.U.)")
        plt.legend()
        plt.show()
def auto_zoom_functional_groups_with_integration(positive_frequencies, positive_magnitude, identified_groups, ppm_shifts, peak_data, buffer=0.01):
    """
    For each functional group with detected peaks, plot a zoomed-in graph
    including integrated peak annotations.
    Args:
        positive_frequencies (np.ndarray): Frequencies in ppm.
        positive_magnitude (np.ndarray): Magnitudes of the FFT result.
        identified_groups (list): List of tuples with peak positions and their corresponding functional groups.
        ppm_shifts (dict): Dictionary mapping functional groups to their ppm ranges.
        peak_data (list): List of tuples with peak ppm and its integrated area.
        buffer (float): Buffer to add around the zoomed region in ppm.
    """
    # Group peaks by functional group
    group_peaks = {}
    for peak_ppm, group in identified_groups:
        group_peaks.setdefault(group, []).append(peak_ppm)

    for group, peaks in group_peaks.items():
        ppm_range_str = ppm_shifts.get(group)
        if not ppm_range_str:
            continue
        ppm_min, ppm_max = map(float, ppm_range_str.replace("ppm", "").split("-"))
        # Only consider peaks within the defined ppm range for the group
        peaks_in_range = [ppm for ppm in peaks if ppm_min <= ppm <= ppm_max]
        if not peaks_in_range:
            continue

        # Define zoom window
        zoom_min = min(peaks_in_range) - buffer
        zoom_max = max(peaks_in_range) + buffer

        # Create ppm axis
        ppm_axis = positive_frequencies / 399.78219838
        mask = (ppm_axis >= zoom_min) & (ppm_axis <= zoom_max)

        # Plot spectrum
        plt.figure(figsize=(8, 4))
        plt.plot(ppm_axis[mask], positive_magnitude[mask], label=f"{group} ({zoom_min:.2f}-{zoom_max:.2f} ppm)")

        # Annotate integrated peaks within this region
        for ppm, area in peak_data:
            if zoom_min <= ppm <= zoom_max:
                plt.axvline(ppm, color='black', linestyle='--', linewidth=1)
                # Offset the text slightly to the right of the line
                plt.text(
                    ppm + 0.01,  # shift right by 0.01 ppm
                    max(positive_magnitude[mask]) * 0.9,
                    f"{area:.2f}",
                    rotation=0,  # horizontal text
                    va='center',
                    ha='left',
                    fontsize=9,
                    color='black',
                    fontweight='bold'
                )

        plt.gca().invert_xaxis()
        plt.title(f"Zoomed Spectrum: {group}")
        plt.xlabel("Chemical Shift (ppm)")
        plt.ylabel("Intensity (A.U.)")
        plt.legend()
        plt.tight_layout()
        plt.show()
def detect_and_plot_multiplet(
    ppm_axis, intensity, center_ppm, spectrometer_freq, window=0.02,
    integration_lookup=None, group_name=None
):


    # Mask for the region of interest
    mask = (ppm_axis > center_ppm - window) & (ppm_axis < center_ppm + window)
    region_ppm = ppm_axis[mask]
    region_intensity = intensity[mask]

    # Smooth the intensity to reduce noise
    region_intensity = gaussian_filter1d(region_intensity, sigma=2)

    # Find sub-peaks in the region with stricter thresholds
    sub_peaks, _ = find_peaks(
        region_intensity,
        height=0.2 * max(region_intensity),
        prominence=0.1 * max(region_intensity)
    )
    sub_ppms = region_ppm[sub_peaks]

    # Sort sub-peaks and estimate J-couplings
    sub_ppms = np.sort(sub_ppms)
    j_couplings = []
    for i in range(1, len(sub_ppms)):
        ppm_diff = abs(sub_ppms[i] - sub_ppms[i-1])
        if ppm_diff < 0.02:  # adjust threshold as needed
            j_hz = ppm_diff * spectrometer_freq
            j_couplings.append(j_hz)

    # Plot the multiplet region
    plt.figure(figsize=(8, 4))
    label = f"{group_name} ({min(region_ppm):.2f}-{max(region_ppm):.2f} ppm)" if group_name else None
    plt.plot(region_ppm, region_intensity, label=label)

    # Annotate sub-peaks with integration values
    for i, ppm in enumerate(sub_ppms):
        plt.axvline(ppm, color='black', linestyle='--', linewidth=1)
        integration_val = None
        if integration_lookup is not None:
            closest = min(integration_lookup, key=lambda x: abs(x[0] - ppm))
            if abs(closest[0] - ppm) < 0.02:
                integration_val = closest[1]
        y_val = region_intensity[sub_peaks[i]] * 0.95
        if integration_val is not None:
            plt.text(
                ppm + 0.01,
                y_val,
                f"{integration_val:.2f}",
                rotation=0,
                va='center',
                ha='left',
                fontsize=8,
                color='black',
                fontweight='bold'
            )

    # Cluster sub-peaks into multiplets
    multiplets = []
    current = [sub_ppms[0]] if len(sub_ppms) > 0 else []
    for i in range(1, len(sub_ppms)):
        if abs(sub_ppms[i] - sub_ppms[i-1]) < 0.05:
            current.append(sub_ppms[i])
        else:
            multiplets.append(current)
            current = [sub_ppms[i]]
    if current:
        multiplets.append(current)

    # Annotate J values within each multiplet
    for group in multiplets:
        for i in range(1, len(group)):
            ppm1, ppm2 = group[i-1], group[i]
            ppm_diff = abs(ppm2 - ppm1)
            mid_ppm = (ppm1 + ppm2) / 2
            y_mid = max(region_intensity) * 0.8
            j_hz = ppm_diff * spectrometer_freq
            plt.text(
                mid_ppm, y_mid,
                f"J = {j_hz:.3f} Hz",
                ha='center', va='bottom', fontsize=7.5, color='red', fontweight='bold'
            )
            plt.plot([ppm1, ppm2], [y_mid*0.98, y_mid*0.98], color='red', linewidth=1)
    # Annotate J-couplings between sub-peaks
    for i in range(1, len(sub_ppms)):
        ppm_diff = abs(sub_ppms[i] - sub_ppms[i-1])
        if ppm_diff < 0.05:  # Only annotate if peaks are close enough (same multiplet)
            mid_ppm = (sub_ppms[i] + sub_ppms[i-1]) / 2
            y_mid = max(region_intensity) * 0.8
            j_hz = ppm_diff * spectrometer_freq
            plt.text(
                mid_ppm, y_mid,
                f"J = {j_hz:.3f} Hz",
                ha='center', va='bottom', fontsize=7.5, color='red', fontweight='bold'
            )
            plt.plot([sub_ppms[i-1], sub_ppms[i]], [y_mid*0.98, y_mid*0.98], color='red', linewidth=1)

    plt.gca().invert_xaxis()
    plt.title(f"Zoomed Spectrum: {group_name or f'Multiplet near {center_ppm:.2f} ppm'}")
    plt.xlabel("Chemical Shift (ppm)")
    plt.ylabel("Intensity (A.U.)")
    if label:
        plt.legend()
    plt.tight_layout()
    plt.show()
    return j_couplings