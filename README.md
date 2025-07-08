# 🧲 NMR-Project

This project provides a comprehensive pipeline for parsing, analyzing, and visualizing Nuclear Magnetic Resonance (NMR) spectroscopy data, with a focus on 1H-NMR.  
It includes data import, Fourier transformation, peak detection, integration, functional group identification, spin-spin coupling analysis, and quantum mechanical simulation of spin systems.

---

## ✨ Features

- 📥 **Data Import & Visualization**  
  Reads JEOL ASCII FID files and visualizes the raw time-domain signal. **(ASCII reccommended)**

- 🔄 **Fourier Transform**  
  Converts the time-domain FID to the frequency domain using FFT, producing the NMR spectrum.

- 📈➕ **Peak Detection & Integration**  
  Detects significant peaks using customizable thresholds and integrates peak areas to estimate relative proton counts.

- 🧬 **Functional Group Identification**  
  Maps detected peaks to chemical functional groups based on their chemical shift (δ, ppm) ranges.

- 🔗 **Spin-Spin Coupling (J-Coupling) Analysis**  
  Detects multiplets, estimates J-coupling constants ($J$), and visualizes multiplet structures with annotated J values.

- ⚛️ **Quantum Mechanical Simulation**  
  Simulates the NMR Hamiltonian for coupled spin systems, computes eigenstates, and animates wavefunction evolution in a potential.

---

## 📁 File Structure
NMR-Project/
│
├── data/
│   └── 13_03_11_indst_1H fid.asc        # Raw JEOL FID ASCII data files
│
├── notebooks/
│   ├── project_1_spring_2025.ipynb      # Main analysis notebook
│   ├── parsing_nmr_data.ipynb           # Data import & FFT walkthrough
│   ├── testing_functions.ipynb          # Notebook for testing/refactoring functions
│   ├── function_ideas.ipynb             # Supplementary ideas and code snippets
│   └── Big_ideas.ipynb                  # Project goals and conceptual notes
│
├── nmr/
│   ├── __init__.py                      # Makes this a Python package
│   ├── nmr_functions.py                 # All reusable NMR analysis functions/classes
│   └── quantum_sim.py                   # Quantum simulation utilities (optional)
│
├── outputs/
│   └── pen.gif                          # Example animation output
│
├── README.md                            # Project overview and usage
├── requirements.txt                     # List of dependencies (for pip install -r)
└── .gitignore                           # Ignore data, outputs, etc. as needed

---

## 🚀 Usage

1. Open `parsing_nmr_data.ipynb` in Jupyter, VS Code, or Colab (**Colab recommended for beginners**).
2. Edit the file path in the data import cell to point to your JEOL FID ASCII file.
3. Run all cells **sequentially** to:
   - 📥 Import and plot raw data
   - 🔄 Perform Fourier transform and plot the spectrum
   - 📈➕ Detect and integrate peaks
   - 🧬 Identify functional groups
   - 🔗 Analyze spin-spin coupling and visualize multiplets
   - ⚛️ Simulate and animate quantum wavefunction evolution

---

## 🛠️ Requirements

- Python 3.7+
- numpy
- pandas
- matplotlib
- seaborn
- scipy
- pillow (for GIF animation)

**Install dependencies with:**  
```sh
pip install numpy pandas matplotlib seaborn scipy pillow
```

---

## 🧩 Key Functions & Notebooks

- **Data Import & FFT:**  
  See the initial code cells in `parsing_nmr_data.ipynb`.

- **Peak Detection & Integration:**  
  Uses `scipy.signal.find_peaks` and `scipy.integrate.simpson`.

- **Functional Group Mapping:**  
  See `identify_functional_groups` and `auto_zoom_functional_groups_with_integration`.

- **Spin-Spin Coupling:**  
  See `detect_and_plot_multiplet` for multiplet and J-coupling analysis.

- **Quantum Simulation:**  
  See the final section for Hamiltonian construction, eigenstate computation, and animation.

---

## 🖼️ Example Output

- **NMR Spectrum:**  
  Plots of the frequency-domain spectrum with detected peaks and functional group annotations.

- **Integration Table:**  
  Printed output of relative proton counts for each peak.

- **Multiplet Visualization:**  
  Zoomed-in plots of multiplets with J-coupling constants annotated.

- **Quantum Animation:**  
  Animated GIF (`pen.gif`) showing the time evolution of a quantum wavefunction in a potential.

---

## 📝 Notes

- The code is modular and can be adapted for other NMR datasets or extended for more advanced analyses.
- For best results, use high-quality FID data and adjust thresholds as needed for your instrument and sample.

---

## 📜 License

This project is for educational and research purposes.

---

## 👤 Author

Created by Quintinlf
For questions, open an issue or contact via GitHub.

---

**See `parsing_nmr_data.ipynb` for full code and documentation.**
