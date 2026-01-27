import pandas as pd
import os
import sys
import argparse
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(base_path)

from ChemMCP.src.chemmcp.tools import BbbpPredictor

# creating this file because chemmcp tools need a separate environment. 
tool = BbbpPredictor()

def run_bbbp(smiles: str):
    output = tool.run_code(smiles=smiles)
    return output

def run_bbbp_batch(smiles_list):
    """Process multiple SMILES strings"""
    results = []
    for i, smiles in enumerate(smiles_list):
        try:
            output = run_bbbp(smiles.strip())
            results.append(f"Molecule {i+1} ({smiles.strip()}): {output}")
        except Exception as e:
            results.append(f"Molecule {i+1} ({smiles.strip()}): Error - {str(e)}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run BBBP prediction on SMILES string(s)')
    parser.add_argument('--smiles', type=str, required=True, 
                       help='SMILES string(s) to predict BBBP for. Can be a single SMILES or comma-separated list of SMILES')
    args = parser.parse_args()
    
    # Check if input contains comma (multiple SMILES)
    if ',' in args.smiles:
        smiles_list = [s.strip() for s in args.smiles.split(',') if s.strip()]
        print("===RESULT===")
        results = run_bbbp_batch(smiles_list)
        for result in results:
            print(result)
    else:
        # Single SMILES
        smiles = args.smiles.strip()
        print("===RESULT===")
        output = run_bbbp(smiles)
        print(f"Molecule ({smiles}): {output}")

    
