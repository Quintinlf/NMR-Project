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