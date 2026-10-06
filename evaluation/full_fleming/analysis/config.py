"""Shared configuration: what the arms are, what the endpoints are, how it is drawn.

Kept in one place so the figures, the tables and anyone reading the repo see the same
definitions. Nothing here computes; it only declares.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "molecules_350.csv")
FIGURE_DIR = os.path.join(HERE, "figures")
RESULT_DIR = os.path.join(HERE, "results")

N_BOOT = 20000
SEED = 0          # fixed, so every interval reproduces exactly rather than approximately

# (display label with mathtext, column in the data file, plain-text label, colour)
#
# The colours are validated for colour-vision deficiency IN THIS ORDER: worst adjacent
# CVD dE 23.1, worst normal-vision dE 24.0 (see validate_palette.py). Reordering changes
# those separations - orange adjacent to violet or green fails the normal-vision floor.
# Green sits just below the 3:1 contrast ratio against white, which is why figures.py
# puts a direct value label on every bar.
ARMS = [
    ("Fleming Plausibility Score",
     "fleming_plausibility", "Fleming plausibility score", "#eb6834"),
    ("LLM (GPT-5.6 Sol + Web Search)\nPlausibility Score",
     "llm_plausibility", "LLM (GPT-5.6 Sol + web search) plausibility", "#2a78d6"),
    (r"D-MPNN  $p_{\mathrm{TB}}$",
     "dmpnn_pTB", "D-MPNN p_TB", "#1baf7a"),
    (r"D-MPNN  $p_{\mathrm{TB}}\,(1-p_{\mathrm{HepG2}})$",
     "dmpnn_pTB_x_1minus_pHepG2", "D-MPNN p_TB x (1 - p_HepG2)", "#4a3aa7"),
]
REFERENCE_ARM = "fleming_plausibility"     # differences are measured against this

# (column, figure label, axis label). Everything signed so higher = better molecule.
ENDPOINTS = [
    ("activity_window", "Selectivity\n(% inhibition − % HepG2 kill)",
     "% inhibition @ 25 µM  −  % HepG2 kill"),
    ("QED", "Drug-likeness\n(QED)", "QED"),
    ("neg_n_alerts", "Freedom from alerts\n(−PAINS/BRENK count)",
     "−PAINS/BRENK alert count"),
]

# Computed and reported, but not shown in Fig. 1. This is the one axis where the ranking
# inverts - the inhibition model is ahead of the agent - so it belongs in the record.
EXTENDED_ENDPOINTS = [
    ("neg_SAscore", "Synthetic accessibility (-SAscore)", "−SAscore"),
]

INK, GRID, AXIS = "black", "#e6e6e6", "#9a9a9a"
SANS = ["Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"]
