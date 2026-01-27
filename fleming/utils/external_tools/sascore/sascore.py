# calculation of synthetic accessibility score as described in:
#
# Estimation of Synthetic Accessibility Score of Drug-like Molecules based on Molecular Complexity and Fragment Contributions
# Peter Ertl and Ansgar Schuffenhauer
# Journal of Cheminformatics 1:8 (2009)
# http://www.jcheminf.com/content/1/1/8
#
# several small modifications to the original paper are included
# particularly slightly different formula for marocyclic penalty
# and taking into account also molecule symmetry (fingerprint density)
#
# for a set of 10k diverse molecules the agreement between the original method
# as implemented in PipelinePilot and this implementation is r2 = 0.97
#
# peter ertl & greg landrum, september 2013
#

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
import math
import pickle
import os.path as op
import gzip

_fscores = None
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2)


def readFragmentScores(name="fpscores.pkl.gz"):
  global _fscores
  # generate the full path filename:
  if name == "fpscores.pkl.gz":
    name = op.join(op.dirname(__file__), name)
  
  if not op.exists(name):
    raise FileNotFoundError(f"Fragment scores file '{name}' not found. Please ensure it is in the same directory as sascore.py.")

  data = pickle.load(gzip.open(name))
  outDict = {}
  for i in data:
    for j in range(1, len(i)):
      outDict[i[j]] = float(i[0])
  _fscores = outDict


def numBridgeheadsAndSpiro(mol, ri=None):
  nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
  nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
  return nBridgehead, nSpiro


def calculateScore(m):
  if not isinstance(m, Chem.Mol):
    raise ValueError("Input must be an RDKit Mol object.")
  if not m.GetNumAtoms():
    return None # Or handle as an error / specific value

  if _fscores is None:
    readFragmentScores()

  # fragment score
  sfp = mfpgen.GetSparseCountFingerprint(m)

  score1 = 0.
  nf = 0
  nze = sfp.GetNonzeroElements()
  if not nze: # Handle case with no fragments if sfp is empty
      return 5.0 # Neutral score if no fragments (or other appropriate default)


  for id_val, count in nze.items(): # Renamed id to id_val to avoid conflict
    nf += count
    score1 += _fscores.get(id_val, -4) * count
  
  if nf == 0: # Avoid division by zero if no fragments were found / counted
      return 5.0 # Or another default, this indicates an issue or very small molecule

  score1 /= nf

  # features score
  nAtoms = m.GetNumAtoms()
  nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
  ri = m.GetRingInfo()
  nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
  nMacrocycles = 0
  for x in ri.AtomRings():
    if len(x) > 8:
      nMacrocycles += 1

  sizePenalty = nAtoms**1.005 - nAtoms
  stereoPenalty = math.log10(nChiralCenters + 1)
  spiroPenalty = math.log10(nSpiro + 1)
  bridgePenalty = math.log10(nBridgeheads + 1)
  macrocyclePenalty = 0.
  # ---------------------------------------
  # This differs from the paper, which defines:
  #  macrocyclePenalty = math.log10(nMacrocycles+1)
  # This form generates better results when 2 or more macrocycles are present
  if nMacrocycles > 0:
    macrocyclePenalty = math.log10(2)

  score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

  # correction for the fingerprint density
  # not in the original publication, added in version 1.1
  # to make highly symmetrical molecules easier to synthetise
  score3 = 0.
  if nAtoms > 0 and len(nze) > 0 : # Check nAtoms and numBits (len(nze)) to avoid math errors
    numBits = len(nze)
    if nAtoms > numBits :
        score3 = math.log(float(nAtoms) / numBits) * .5

  sascore = score1 + score2 + score3

  # need to transform "raw" value into scale between 1 and 10
  min_val = -4.0 # Renamed min to min_val
  max_val = 2.5  # Renamed max to max_val
  sascore = 11. - (sascore - min_val + 1) / (max_val - min_val) * 9.
  # Ensure sascore does not exceed bounds due to floating point issues before smoothing
  sascore = min(max(sascore, 1.0), 10.0)


  # smooth the 10-end
  if sascore > 8.:
    sascore = 8. + math.log(sascore + 1. - 9.) if (sascore + 1. - 9.) > 0 else 8.0 # Avoid math domain error
  
  # Final cap after smoothing
  if sascore > 10.:
    sascore = 10.0
  elif sascore < 1.:
    sascore = 1.0
  
  return sascore


def processMols(mols):
  # This function is for command-line use, not directly used by the agent
  print('smiles\tName\tsa_score')
  for i, m in enumerate(mols):
    if m is None:
      continue

    s = calculateScore(m)

    smiles = Chem.MolToSmiles(m)
    name_prop = m.GetProp('_Name') if m.HasProp('_Name') else f"Mol_{i}"
    if s is None:
      print(f"{smiles}\t{name_prop}\t{s}")
    else:
      print(f"{smiles}\t{name_prop}\t{s:.3f}")


if __name__ == '__main__':
  # This main block is for command-line testing of sascore.py
  # It's not essential for the agent's functionality but good for testing the SAscore script.
  import sys
  import time

  if len(sys.argv) < 2:
      print("Usage: python sascore.py <molFile.smi/.sdf> [fpscores.pkl.gz]")
      sys.exit(1)

  t1 = time.time()
  score_file = "fpscores.pkl.gz"
  if len(sys.argv) == 3: # User provided a path to fpscores
      score_file = sys.argv[2]
  
  try:
      readFragmentScores(score_file)
  except FileNotFoundError as e:
      print(e, file=sys.stderr)
      print("Please ensure 'fpscores.pkl.gz' is accessible or provide its path as the second argument.", file=sys.stderr)
      sys.exit(1)

  t2 = time.time()

  molFile = sys.argv[1]
  if molFile.endswith(".smi"):
    suppl = Chem.SmilesMolSupplier(molFile, titleLine=False) # Assuming typical SMILES file
  elif molFile.endswith(".sdf"):
    suppl = Chem.SDMolSupplier(molFile)
  else:
    print(f"Unrecognized file extension for {molFile}", file=sys.stderr)
    sys.exit(1)

  # Prepare a list of Mol objects to pass to processMols
  mols_to_process = []
  print("Loading molecules...")
  for i, mol in enumerate(suppl):
      if mol is None:
          print(f"Warning: Could not parse molecule at index {i} from {molFile}", file=sys.stderr)
          continue
      if not mol.HasProp('_Name'):
          mol.SetProp('_Name', f"Mol_{i+1}")
      mols_to_process.append(mol)
  
  if not mols_to_process:
      print(f"No molecules loaded from {molFile}", file=sys.stderr)
      sys.exit(1)

  t3 = time.time()
  print(f"Processing {len(mols_to_process)} molecules...")
  processMols(mols_to_process)
  t4 = time.time()

  print('Reading fragment scores took %.2f seconds. Loading and calculating SAscores took %.2f seconds' % ((t2 - t1), (t4 - t3)),
        file=sys.stderr)

#
#  Copyright (c) 2013, Novartis Institutes for BioMedical Research Inc.
#  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
#     * Neither the name of Novartis Institutes for BioMedical Research Inc.
#       nor the names of its contributors may be used to endorse or promote
#       products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#