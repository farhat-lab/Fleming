"""Plot molecules from an Excel file of SMILES.

Expected input: an .xlsx file where the first column contains SMILES strings.
Outputs: PNG files saved in the plots folder.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw


def _sanitize_filename(text: str, max_len: int = 80) -> str:
	safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
	return safe[:max_len] if safe else "molecule"


def _load_smiles(excel_path: Path) -> list[str]:
	df = pd.read_excel(excel_path)
	if df.empty:
		return []

	first_col = df.columns[0]
	smiles_series = df[first_col].dropna().astype(str)
	return smiles_series.tolist()


def plot_molecules(excel_path: Path, output_dir: Path) -> int:
	output_dir.mkdir(parents=True, exist_ok=True)
	smiles_list = _load_smiles(excel_path)

	if not smiles_list:
		print("No SMILES found in the Excel file.")
		return 0

	plotted = 0
	for index, input_smiles in enumerate(smiles_list, start=1):
		mol = Chem.MolFromSmiles(input_smiles)
		if mol is None:
			print(f"Skipping invalid SMILES at row {index}: {input_smiles}")
			continue

		clean_smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
		safe_name = _sanitize_filename(clean_smiles)
		file_path = output_dir / f"rank{index}_{safe_name}.png"

		img = Draw.MolToImage(mol, kekulize=True, fitImage=True, fitWidth=True)
		img.save(file_path)
		plotted += 1

	print(f"Saved {plotted} molecule plots to {output_dir}")
	return plotted


def main() -> None:
	script_dir = Path(__file__).resolve().parent
	parser = argparse.ArgumentParser(description="Plot molecules from an Excel file.")
	parser.add_argument(
		"--excel",
		default=str(script_dir / "final_gen_molecule_selection.xlsx"),
		help="Path to the Excel file containing SMILES",
	)
	parser.add_argument(
		"--output",
		default=str(script_dir / "plots"),
		help="Output folder for PNG plots",
	)

	args = parser.parse_args()
	plot_molecules(Path(args.excel), Path(args.output))


if __name__ == "__main__":
	main()
