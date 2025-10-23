"""
NMR Peak Assignment Module

This module implements comprehensive 1H NMR peak assignment based on:
1. Chemical equivalent and non-equivalent protons
2. Chemical shift analysis
3. Integration patterns
4. Signal splitting (multiplet analysis)

Based on NMR theory from Chapter 6.6 - Understanding 1H NMR Spectra
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.integrate import simpson
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Tuple, Optional
import warnings


class ChemicalShiftDatabase:
    """
    Database of chemical shift ranges for common functional groups
    Based on Table 6.2 and Figure 6.6c
    """
    
    # Chemical shift ranges (ppm) for common functional groups
    SHIFT_RANGES = {
        # Alkyl protons
        "Primary alkyl (R-CH3)": (0.8, 1.2),
        "Secondary alkyl (R2-CH2)": (1.2, 1.7),
        "Tertiary alkyl (R3-CH)": (1.4, 1.7),
        
        # Protons alpha to functional groups
        "Alpha to C=C (C-CH2-C=C)": (2.0, 2.5),
        "Alpha to C=O (C-CH2-C=O)": (2.0, 2.7),
        "Alpha to aromatic (C-CH2-Ar)": (2.2, 2.8),
        
        # Protons on carbons with electronegative atoms
        "C-CH2-O (ether/alcohol)": (3.3, 3.8),
        "C-CH2-N (amine)": (2.2, 2.9),
        "C-CH2-X (halogen)": (3.6, 4.3),
        
        # Vinylic protons (C=C-H)
        "Alkene protons (C=C-H)": (4.5, 6.5),
        
        # Aromatic protons
        "Aromatic protons (Ar-H)": (6.5, 8.5),
        "Benzene protons": (7.0, 7.5),
        
        # Special functional groups
        "Aldehyde protons (R-CHO)": (9.5, 10.5),
        "Carboxylic acid protons (R-COOH)": (10.5, 12.0),
        
        # Variable chemical shift groups
        "Alcohol protons (R-OH)": (1.0, 5.0),  # Highly variable
        "Amine protons (R-NH2)": (1.0, 5.0),  # Highly variable
        
        # Common compounds
        "TMS reference": (0.0, 0.0),
        "Chloroform-d": (7.26, 7.26),
        "DMSO-d6": (2.50, 2.50),
    }
    
    @classmethod
    def get_shift_range(cls, functional_group: str) -> Tuple[float, float]:
        """Get chemical shift range for a functional group"""
        return cls.SHIFT_RANGES.get(functional_group, (0.0, 12.0))
    
    @classmethod
    def identify_functional_group(cls, chemical_shift: float) -> List[str]:
        """
        Identify possible functional groups for a given chemical shift
        Returns list of possible assignments in order of likelihood
        """
        matches = []
        for group, (min_shift, max_shift) in cls.SHIFT_RANGES.items():
            if min_shift <= chemical_shift <= max_shift:
                # Calculate how central the shift is within the range
                center = (min_shift + max_shift) / 2
                distance = abs(chemical_shift - center)
                matches.append((group, distance))
        
        # Sort by distance from center (most likely first)
        matches.sort(key=lambda x: x[1])
        return [group for group, _ in matches]
    
    @classmethod
    def get_typical_chemical_shifts(cls) -> Dict[str, float]:
        """Get typical (center) chemical shifts for each functional group"""
        return {group: (min_shift + max_shift) / 2 
                for group, (min_shift, max_shift) in cls.SHIFT_RANGES.items()}


class PeakAssignmentAnalyzer:
    """
    Main class for NMR peak assignment analysis
    """
    
    def __init__(self, spectrometer_freq: float = 400.0):
        """
        Initialize the peak assignment analyzer
        
        Args:
            spectrometer_freq: Spectrometer frequency in MHz
        """
        self.spectrometer_freq = spectrometer_freq
        self.shift_db = ChemicalShiftDatabase()
        self.peaks_data = None
        self.assignments = []
        
    def analyze_spectrum(self, 
                        ppm_axis: np.ndarray, 
                        intensity: np.ndarray,
                        height_threshold: float = 0.1,
                        min_distance_hz: float = 10.0,
                        prominence_threshold: float = 0.05) -> Dict:
        """
        Complete analysis of NMR spectrum including peak detection and assignment
        
        Args:
            ppm_axis: Chemical shift axis in ppm
            intensity: Spectrum intensity values
            height_threshold: Minimum peak height (fraction of max intensity)
            min_distance_hz: Minimum distance between peaks in Hz
            prominence_threshold: Minimum peak prominence (fraction of max intensity)
            
        Returns:
            Dictionary containing analysis results
        """
        
        # 1. Peak Detection
        peaks_info = self._detect_peaks(ppm_axis, intensity, 
                                      height_threshold, min_distance_hz, prominence_threshold)
        
        # 2. Peak Integration
        integration_info = self._integrate_peaks(ppm_axis, intensity, peaks_info)
        
        # 3. Chemical Shift Assignment
        assignments = self._assign_chemical_shifts(peaks_info)
        
        # 4. Multiplicity Analysis
        multiplicity_info = self._analyze_multiplicity(ppm_axis, intensity, peaks_info)
        
        # Store results
        self.peaks_data = peaks_info
        self.assignments = assignments
        
        return {
            'peaks': peaks_info,
            'integration': integration_info,
            'assignments': assignments,
            'multiplicity': multiplicity_info,
            'summary': self._generate_summary()
        }
    
    def _detect_peaks(self, ppm_axis: np.ndarray, intensity: np.ndarray,
                     height_threshold: float, min_distance_hz: float, 
                     prominence_threshold: float) -> Dict:
        """Detect peaks in the spectrum"""
        
        max_intensity = np.max(intensity)
        height = height_threshold * max_intensity
        prominence = prominence_threshold * max_intensity
        
        # Convert Hz distance to points
        hz_per_point = (ppm_axis.max() - ppm_axis.min()) * self.spectrometer_freq / len(ppm_axis)
        min_distance_points = max(1, int(min_distance_hz / hz_per_point))
        
        # Find peaks
        peaks_idx, properties = find_peaks(
            intensity,
            height=height,
            distance=min_distance_points,
            prominence=prominence,
            width=1
        )
        
        # Extract peak information
        peak_ppms = ppm_axis[peaks_idx]
        peak_intensities = intensity[peaks_idx]
        peak_heights = properties['peak_heights']
        peak_widths = properties.get('widths', np.ones(len(peaks_idx)))
        
        return {
            'indices': peaks_idx,
            'chemical_shifts': peak_ppms,
            'intensities': peak_intensities,
            'heights': peak_heights,
            'widths': peak_widths,
            'properties': properties
        }
    
    def _integrate_peaks(self, ppm_axis: np.ndarray, intensity: np.ndarray, 
                        peaks_info: Dict) -> Dict:
        """Integrate peak areas"""
        
        peaks_idx = peaks_info['indices']
        properties = peaks_info['properties']
        
        # Use peak bases for integration limits
        left_bases = properties.get('left_bases', peaks_idx - 10)
        right_bases = properties.get('right_bases', peaks_idx + 10)
        
        integrated_areas = []
        for i, peak_idx in enumerate(peaks_idx):
            left = max(0, int(left_bases[i]))
            right = min(len(intensity) - 1, int(right_bases[i]))
            
            # Integrate using Simpson's rule
            ppm_region = ppm_axis[left:right+1]
            intensity_region = intensity[left:right+1]
            
            if len(ppm_region) > 1:
                area = simpson(intensity_region, ppm_region)
            else:
                area = intensity_region[0] if len(intensity_region) > 0 else 0
            
            integrated_areas.append(abs(area))  # Take absolute value
        
        # Normalize to smallest peak = 1.0
        if integrated_areas:
            min_area = min(integrated_areas)
            if min_area > 0:
                normalized_areas = [area / min_area for area in integrated_areas]
            else:
                normalized_areas = integrated_areas
        else:
            normalized_areas = []
        
        return {
            'raw_areas': integrated_areas,
            'normalized_areas': normalized_areas,
            'relative_integrals': self._round_to_nearest_integer(normalized_areas)
        }
    
    def _assign_chemical_shifts(self, peaks_info: Dict) -> List[Dict]:
        """Assign functional groups based on chemical shifts"""
        
        assignments = []
        chemical_shifts = peaks_info['chemical_shifts']
        
        for i, shift in enumerate(chemical_shifts):
            possible_groups = self.shift_db.identify_functional_group(shift)
            
            assignment = {
                'peak_index': i,
                'chemical_shift': shift,
                'possible_assignments': possible_groups[:3],  # Top 3 matches
                'most_likely': possible_groups[0] if possible_groups else "Unknown",
                'confidence': self._calculate_confidence(shift, possible_groups)
            }
            assignments.append(assignment)
        
        return assignments
    
    def _analyze_multiplicity(self, ppm_axis: np.ndarray, intensity: np.ndarray, 
                            peaks_info: Dict) -> List[Dict]:
        """Analyze peak multiplicity and J-coupling constants"""
        
        multiplicity_info = []
        chemical_shifts = peaks_info['chemical_shifts']
        
        for i, center_ppm in enumerate(chemical_shifts):
            # Analyze region around each peak
            window = 0.05  # ± 0.05 ppm window
            mask = (ppm_axis >= center_ppm - window) & (ppm_axis <= center_ppm + window)
            
            if np.sum(mask) < 5:  # Need minimum points for analysis
                multiplicity_info.append({
                    'peak_index': i,
                    'multiplicity': 'singlet',
                    'j_couplings': [],
                    'pattern': 'Simple singlet'
                })
                continue
            
            region_ppm = ppm_axis[mask]
            region_intensity = intensity[mask]
            
            # Smooth to reduce noise
            smoothed_intensity = gaussian_filter1d(region_intensity, sigma=1)
            
            # Find sub-peaks
            sub_peaks_idx, _ = find_peaks(
                smoothed_intensity,
                height=0.3 * np.max(smoothed_intensity),
                distance=3
            )
            
            if len(sub_peaks_idx) <= 1:
                multiplicity = 'singlet'
                j_couplings = []
                pattern = 'Simple singlet'
            else:
                sub_peak_ppms = region_ppm[sub_peaks_idx]
                j_couplings = self._calculate_j_couplings(sub_peak_ppms)
                multiplicity = self._determine_multiplicity(len(sub_peaks_idx), j_couplings)
                pattern = self._describe_splitting_pattern(multiplicity, j_couplings)
            
            multiplicity_info.append({
                'peak_index': i,
                'multiplicity': multiplicity,
                'j_couplings': j_couplings,
                'pattern': pattern,
                'sub_peaks': len(sub_peaks_idx)
            })
        
        return multiplicity_info
    
    def _calculate_j_couplings(self, sub_peak_ppms: np.ndarray) -> List[float]:
        """Calculate J-coupling constants from sub-peak positions"""
        
        if len(sub_peak_ppms) < 2:
            return []
        
        # Sort peaks
        sorted_ppms = np.sort(sub_peak_ppms)
        
        # Calculate differences and convert to Hz
        j_couplings = []
        for i in range(len(sorted_ppms) - 1):
            ppm_diff = sorted_ppms[i+1] - sorted_ppms[i]
            j_hz = ppm_diff * self.spectrometer_freq
            j_couplings.append(j_hz)
        
        return j_couplings
    
    def _determine_multiplicity(self, num_peaks: int, j_couplings: List[float]) -> str:
        """Determine multiplicity name based on number of peaks"""
        
        multiplicity_names = {
            1: 'singlet',
            2: 'doublet', 
            3: 'triplet',
            4: 'quartet',
            5: 'quintet',
            6: 'sextet',
            7: 'septet'
        }
        
        if num_peaks in multiplicity_names:
            return multiplicity_names[num_peaks]
        else:
            return f'multiplet ({num_peaks} peaks)'
    
    def _describe_splitting_pattern(self, multiplicity: str, j_couplings: List[float]) -> str:
        """Generate description of splitting pattern"""
        
        if not j_couplings:
            return f"Simple {multiplicity}"
        
        avg_j = np.mean(j_couplings)
        j_std = np.std(j_couplings) if len(j_couplings) > 1 else 0
        
        if j_std < 2.0:  # Similar J values
            return f"{multiplicity.capitalize()}, J ≈ {avg_j:.1f} Hz"
        else:
            j_values = [f"{j:.1f}" for j in j_couplings]
            return f"{multiplicity.capitalize()}, J = {', '.join(j_values)} Hz"
    
    def _calculate_confidence(self, chemical_shift: float, possible_groups: List[str]) -> float:
        """Calculate confidence score for assignment (0-1)"""
        
        if not possible_groups:
            return 0.0
        
        # Get the range for the most likely assignment
        best_group = possible_groups[0]
        min_shift, max_shift = self.shift_db.get_shift_range(best_group)
        
        # Calculate how central the shift is within the range
        range_width = max_shift - min_shift
        if range_width == 0:
            return 1.0  # Exact match
        
        center = (min_shift + max_shift) / 2
        distance_from_center = abs(chemical_shift - center)
        normalized_distance = distance_from_center / (range_width / 2)
        
        # Convert to confidence (1.0 = center, 0.0 = edge)
        confidence = max(0.0, 1.0 - normalized_distance)
        return confidence
    
    def _round_to_nearest_integer(self, values: List[float], tolerance: float = 0.3) -> List[int]:
        """Round integration values to nearest integers (proton counts)"""
        
        if not values:
            return []
        
        rounded = []
        for val in values:
            nearest_int = round(val)
            if abs(val - nearest_int) <= tolerance:
                rounded.append(nearest_int)
            else:
                rounded.append(int(val))  # Truncate if not close to integer
        
        return rounded
    
    def _generate_summary(self) -> str:
        """Generate a summary of the peak assignment analysis"""
        
        if not self.assignments:
            return "No peaks detected or analyzed."
        
        summary_lines = [
            f"NMR Peak Assignment Summary ({self.spectrometer_freq} MHz)",
            "=" * 50,
            f"Number of signals detected: {len(self.assignments)}",
            ""
        ]
        
        for i, assignment in enumerate(self.assignments, 1):
            shift = assignment['chemical_shift']
            group = assignment['most_likely']
            confidence = assignment['confidence']
            
            summary_lines.append(
                f"Peak {i}: δ {shift:.2f} ppm → {group} "
                f"(confidence: {confidence:.2f})"
            )
        
        return "\n".join(summary_lines)
    
    def plot_assigned_spectrum(self, ppm_axis: np.ndarray, intensity: np.ndarray, 
                              integration_data: Dict = None, 
                              show_assignments: bool = True,
                              show_integrals: bool = True,
                              figsize: Tuple[int, int] = (12, 6)) -> None:
        """Plot spectrum with peak assignments and integrations"""
        
        if self.peaks_data is None:
            raise ValueError("Run analyze_spectrum() first")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot spectrum
        ax.plot(ppm_axis, intensity, 'b-', linewidth=1, label='1H NMR Spectrum')
        
        # Mark detected peaks
        peak_ppms = self.peaks_data['chemical_shifts']
        peak_intensities = self.peaks_data['intensities']
        
        ax.plot(peak_ppms, peak_intensities, 'ro', markersize=6, label='Detected Peaks')
        
        # Add assignments
        if show_assignments:
            for i, assignment in enumerate(self.assignments):
                shift = assignment['chemical_shift']
                group = assignment['most_likely']
                intensity_val = peak_intensities[i]
                
                # Annotate with functional group
                ax.annotate(
                    f"{group}\nδ {shift:.2f}",
                    xy=(shift, intensity_val),
                    xytext=(shift, intensity_val + 0.1 * np.max(intensity)),
                    ha='center', va='bottom',
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                )
        
        # Add integration values
        if show_integrals and integration_data:
            normalized_areas = integration_data.get('normalized_areas', [])
            for i, (shift, area) in enumerate(zip(peak_ppms, normalized_areas)):
                ax.text(
                    shift, -0.05 * np.max(intensity),
                    f"{area:.1f}H",
                    ha='center', va='top',
                    fontsize=9, fontweight='bold',
                    color='red'
                )
        
        # Formatting
        ax.invert_xaxis()
        ax.set_xlabel('Chemical Shift (δ, ppm)', fontsize=12)
        ax.set_ylabel('Intensity (A.U.)', fontsize=12)
        ax.set_title('1H NMR Spectrum with Peak Assignments', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def export_assignments_table(self, integration_data: Dict = None) -> pd.DataFrame:
        """Export peak assignments as a formatted table"""
        
        if not self.assignments:
            return pd.DataFrame()
        
        data = []
        for i, assignment in enumerate(self.assignments):
            row = {
                'Peak': i + 1,
                'Chemical Shift (ppm)': f"{assignment['chemical_shift']:.2f}",
                'Assignment': assignment['most_likely'],
                'Confidence': f"{assignment['confidence']:.2f}",
                'Alternative Assignments': ', '.join(assignment['possible_assignments'][1:3])
            }
            
            # Add integration data if available
            if integration_data:
                normalized_areas = integration_data.get('normalized_areas', [])
                relative_integrals = integration_data.get('relative_integrals', [])
                
                if i < len(normalized_areas):
                    row['Integration (relative)'] = f"{normalized_areas[i]:.1f}"
                if i < len(relative_integrals):
                    row['Proton Count'] = f"{relative_integrals[i]}H"
            
            data.append(row)
        
        return pd.DataFrame(data)


def predict_nmr_signals(molecular_structure: str) -> Dict:
    """
    Predict the number of 1H NMR signals for common molecular structures
    
    This is a simplified implementation for educational purposes.
    Real structure prediction would require more sophisticated molecular analysis.
    
    Args:
        molecular_structure: String description of the molecule
        
    Returns:
        Dictionary with predicted signal information
    """
    
    # Simple pattern matching for common molecules (educational examples)
    predictions = {
        'benzene': {
            'num_signals': 1,
            'explanation': 'All 6 aromatic protons are equivalent due to symmetry',
            'expected_shifts': [7.3],
            'multiplicities': ['singlet'],
            'integrations': [6]
        },
        'acetone': {
            'num_signals': 1,
            'explanation': 'Both methyl groups are equivalent, all 6 protons show one signal',
            'expected_shifts': [2.1],
            'multiplicities': ['singlet'],
            'integrations': [6]
        },
        'acetaldehyde': {
            'num_signals': 2,
            'explanation': 'CH3 group (3H) and CHO proton (1H) are in different environments',
            'expected_shifts': [2.2, 9.8],
            'multiplicities': ['doublet', 'quartet'],
            'integrations': [3, 1]
        },
        'methyl_acetate': {
            'num_signals': 2,
            'explanation': 'OCH3 group and COCH3 group are in different chemical environments',
            'expected_shifts': [3.7, 2.1],
            'multiplicities': ['singlet', 'singlet'],
            'integrations': [3, 3]
        }
    }
    
    structure_key = molecular_structure.lower().replace(' ', '_').replace('-', '_')
    
    if structure_key in predictions:
        return predictions[structure_key]
    else:
        return {
            'num_signals': 'Unknown',
            'explanation': 'Structure not in database. Manual analysis required.',
            'expected_shifts': [],
            'multiplicities': [],
            'integrations': []
        }


# Example usage and testing functions
def example_peak_assignment_analysis():
    """Example of how to use the PeakAssignmentAnalyzer"""
    
    # Create synthetic NMR spectrum for demonstration
    ppm_axis = np.linspace(12, 0, 2000)
    
    # Simulate peaks for methyl acetate
    # Peak 1: OCH3 at 3.7 ppm (3H, singlet)
    # Peak 2: COCH3 at 2.1 ppm (3H, singlet)
    
    intensity = np.zeros_like(ppm_axis)
    
    # Add peaks (Gaussian shapes)
    def add_peak(ppm_center, height, width=0.02):
        peak = height * np.exp(-((ppm_axis - ppm_center) / width) ** 2)
        return peak
    
    intensity += add_peak(3.7, 1.0, 0.02)  # OCH3
    intensity += add_peak(2.1, 1.0, 0.02)  # COCH3
    intensity += np.random.normal(0, 0.01, len(ppm_axis))  # Add noise
    
    # Analyze spectrum
    analyzer = PeakAssignmentAnalyzer(spectrometer_freq=400.0)
    results = analyzer.analyze_spectrum(ppm_axis, intensity)
    
    # Print results
    print("Peak Assignment Analysis Results")
    print("=" * 40)
    print(results['summary'])
    print()
    
    # Show assignments table
    integration_data = results['integration']
    assignments_table = analyzer.export_assignments_table(integration_data)
    print("Assignments Table:")
    print(assignments_table.to_string(index=False))
    
    # Plot results
    analyzer.plot_assigned_spectrum(ppm_axis, intensity, integration_data)
    
    return results


if __name__ == "__main__":
    # Run example
    example_results = example_peak_assignment_analysis()
    
    # Test molecular structure prediction
    print("\n" + "=" * 50)
    print("Molecular Structure Prediction Examples")
    print("=" * 50)
    
    test_molecules = ['benzene', 'acetone', 'acetaldehyde', 'methyl_acetate']
    
    for molecule in test_molecules:
        prediction = predict_nmr_signals(molecule)
        print(f"\n{molecule.replace('_', ' ').title()}:")
        print(f"  Expected signals: {prediction['num_signals']}")
        print(f"  Explanation: {prediction['explanation']}")
        if prediction['expected_shifts']:
            print(f"  Chemical shifts: {prediction['expected_shifts']} ppm")
            print(f"  Integrations: {prediction['integrations']}H")