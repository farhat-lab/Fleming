from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

try:
    import selfies as sf
except ImportError:  # pragma: no cover - optional dependency
    sf = None


COMMON_SELFIES_TOKENS = (
    "[C]",
    "[=C]",
    "[#C]",
    "[N]",
    "[=N]",
    "[O]",
    "[=O]",
    "[S]",
    "[=S]",
    "[P]",
    "[F]",
    "[Cl]",
    "[Br]",
    "[Ring1]",
    "[Branch1]",
)


@dataclass
class ChemicalSpaceSearchResult:
    strategy: str
    candidates: list[str]
    metadata: dict


def canonicalize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)


def tanimoto_similarity(smiles_a: str, smiles_b: str) -> float:
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        return 0.0
    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, nBits=2048)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, nBits=2048)
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def _selfies_token_set(smiles: str) -> list[str]:
    if sf is None:
        return list(COMMON_SELFIES_TOKENS)
    try:
        encoded = sf.encoder(smiles)
        alphabet = set(sf.split_selfies(encoded))
        alphabet.update(COMMON_SELFIES_TOKENS)
        return sorted(alphabet)
    except Exception:
        return list(COMMON_SELFIES_TOKENS)


def _stoned_like_neighbors(smiles: str, max_variants: int = 96) -> ChemicalSpaceSearchResult:
    if sf is None:
        return ChemicalSpaceSearchResult(
            strategy="immediate_fallback",
            candidates=_rdkit_local_neighbors(smiles, max_variants=max_variants),
            metadata={"reason": "selfies_not_installed"},
        )

    encoded = sf.encoder(smiles)
    tokens = list(sf.split_selfies(encoded))
    alphabet = _selfies_token_set(smiles)
    variants: set[str] = set()

    def maybe_add(candidate_selfies: Iterable[str]) -> None:
        if len(variants) >= max_variants:
            return
        try:
            decoded = sf.decoder("".join(candidate_selfies))
        except Exception:
            return
        canon = canonicalize_smiles(decoded)
        if canon and canon != smiles:
            variants.add(canon)

    for idx in range(len(tokens)):
        original = tokens[idx]
        for replacement in alphabet:
            if replacement == original:
                continue
            trial = list(tokens)
            trial[idx] = replacement
            maybe_add(trial)
            if len(variants) >= max_variants:
                break
        if len(variants) >= max_variants:
            break

    if len(tokens) > 1:
        for idx in range(len(tokens)):
            trial = tokens[:idx] + tokens[idx + 1 :]
            maybe_add(trial)
            if len(variants) >= max_variants:
                break

    insertion_positions = range(len(tokens) + 1)
    for idx in insertion_positions:
        for addition in alphabet[:8]:
            trial = tokens[:idx] + [addition] + tokens[idx:]
            maybe_add(trial)
            if len(variants) >= max_variants:
                break
        if len(variants) >= max_variants:
            break

    ranked = sorted(
        variants,
        key=lambda candidate: (
            -tanimoto_similarity(smiles, candidate),
            abs(len(candidate) - len(smiles)),
            candidate,
        ),
    )
    return ChemicalSpaceSearchResult(
        strategy="immediate",
        candidates=ranked[:max_variants],
        metadata={"generator": "stoned_like", "raw_count": len(variants)},
    )


def _rdkit_local_neighbors(smiles: str, max_variants: int = 96) -> list[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    replacements = {
        6: [7, 8, 16],
        7: [6, 8, 16],
        8: [6, 7, 16],
        16: [6, 7, 8],
        9: [17, 35],
        17: [9, 35],
        35: [9, 17],
    }
    variants: set[str] = set()

    for atom_idx, atom in enumerate(mol.GetAtoms()):
        atomic_num = atom.GetAtomicNum()
        for replacement in replacements.get(atomic_num, []):
            new_mol = Chem.RWMol(mol)
            new_mol.GetAtomWithIdx(atom_idx).SetAtomicNum(replacement)
            try:
                candidate = Chem.MolToSmiles(new_mol, isomericSmiles=True)
            except Exception:
                continue
            canon = canonicalize_smiles(candidate)
            if canon and canon != smiles:
                variants.add(canon)
            if len(variants) >= max_variants:
                break
        if len(variants) >= max_variants:
            break

    ranked = sorted(
        variants,
        key=lambda candidate: (-tanimoto_similarity(smiles, candidate), candidate),
    )
    return ranked[:max_variants]


def explore_immediate_chemical_space(smiles: str, max_variants: int = 96) -> ChemicalSpaceSearchResult:
    canonical = canonicalize_smiles(smiles)
    if canonical is None:
        return ChemicalSpaceSearchResult(
            strategy="immediate",
            candidates=[],
            metadata={"reason": "invalid_smiles"},
        )
    return _stoned_like_neighbors(canonical, max_variants=max_variants)


def explore_synspace(smiles: str, max_variants: int = 150) -> ChemicalSpaceSearchResult:
    try:
        from synspace import synspace
    except ImportError:  # pragma: no cover - optional dependency
        return ChemicalSpaceSearchResult(
            strategy="synspace",
            candidates=[],
            metadata={"reason": "synspace_not_installed"},
        )

    canonical = canonicalize_smiles(smiles)
    if canonical is None:
        return ChemicalSpaceSearchResult(
            strategy="synspace",
            candidates=[],
            metadata={"reason": "invalid_smiles"},
        )

    try:
        mols, sibling_props = synspace.chemical_space(canonical, num_samples=max_variants)
    except Exception as exc:  # pragma: no cover - external dependency
        return ChemicalSpaceSearchResult(
            strategy="synspace",
            candidates=[],
            metadata={"reason": f"synspace_error: {exc}"},
        )

    candidates = []
    for mol in mols:
        try:
            candidate = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        except Exception:
            continue
        if candidate and candidate != canonical:
            candidates.append(candidate)

    deduped = list(dict.fromkeys(candidates))
    return ChemicalSpaceSearchResult(
        strategy="synspace",
        candidates=deduped[:max_variants],
        metadata={"raw_count": len(deduped), "sibling_props_available": sibling_props is not None},
    )
