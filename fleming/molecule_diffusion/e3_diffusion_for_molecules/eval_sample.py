# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass

import sys
sys.path.append("./molecule_diffusion/e3_diffusion_for_molecules/")

import utils
import argparse
from configs.datasets_config import qm9_with_h, qm9_without_h
from qm9 import dataset
from qm9.models import get_model

from equivariant_diffusion.utils import assert_correctly_masked
import torch
import pickle
import qm9.visualizer as vis
from qm9.analyze import check_stability
from os.path import join
from qm9.sampling import sample_chain, sample
from configs.datasets_config import get_dataset_info
import os
import sys
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# now you can import sascore!
import sascorer
from rdkit import Chem
from rdkit.Chem import AllChem
from qm9.rdkit_functions import build_molecule, mol2smiles
from qm9.analyze import analyze_stability_for_molecules
from qm9.utils import compute_mean_mad


def check_mask_correct(variables, node_mask):
    for variable in variables:
        assert_correctly_masked(variable, node_mask)


def save_and_sample_chain(args, eval_args, device, flow,
                          n_tries, n_nodes, dataset_info, id_from=0,
                          num_chains=100):

    for i in range(num_chains):
        target_path = f'eval/chain_{i}/'

        one_hot, charges, x = sample_chain(
            args, device, flow, n_tries, dataset_info)

        vis.save_xyz_file(
            join(eval_args.model_path, target_path), one_hot, charges, x,
            dataset_info, id_from, name='chain')

        vis.visualize_chain_uncertainty(
            join(eval_args.model_path, target_path), dataset_info,
            spheres_3d=True)

    return one_hot, charges, x
#SBATCH --array=0-20%4
def sample_different_sizes_and_save(args, eval_args, device, generative_model,
                                    nodes_dist, dataset_info, n_samples=10, full_sample=10000, 
                                    mean = None, mad = None, return_single_molecule=False):
    res_mols = []
    if return_single_molecule is False:
        resultat = open('sampled_inhibitory_molecules_fixed_large_new_4', 'a')
    else:  
        full_sample = 1

    while len(res_mols) < full_sample:
        print(f"Generated molecule: {len(res_mols)}")
        nodesxsample = nodes_dist.sample(n_samples)
        molecules = {'one_hot': [], 'x': [], 'node_mask': []}
        
        #Samples n_samples number of nodes from random distribution 
        one_hot, charges, x, node_mask = sample(
            args, device, generative_model, dataset_info, prop_dist = None,
            nodesxsample=nodesxsample, mean = mean, mad = mad)

        molecules['one_hot'].append(one_hot.detach().cpu())
        molecules['x'].append(x.detach().cpu())
        molecules['node_mask'].append(node_mask.detach().cpu())

        molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
        validity_dict, rdkit_tuple = analyze_stability_for_molecules(molecules, dataset_info)

        for i in range(n_samples):
            num_atoms = int(node_mask[i].sum().item())
            atom_type = one_hot[i, :num_atoms].argmax(1).cpu().detach().numpy()
            # atom_decoder = dataset_info['atom_decoder']
            # atom_type = [atom_decoder[atom] for atom in atom_type]
            #get atomic number for each atom in atom_type
            # pt = Chem.GetPeriodicTable()
            # atom_type = [pt.GetAtomicNumber(atom) for atom in atom_type]
            x_squeeze = x[i, :num_atoms].squeeze(0).cpu().detach().numpy()
            mol = build_molecule(torch.tensor(x_squeeze), torch.tensor(atom_type), dataset_info, already_decoded = False)
            smiles = mol2smiles(mol)
            if smiles is not None:
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
                largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                smiles = mol2smiles(largest_mol)

                print(f"Smiles {smiles}")
                print(f"Synthesizability {sascorer.calculateScore(Chem.MolFromSmiles(smiles))}")
                if return_single_molecule is True:
                    return smiles
                else:
                    res_mols.append(smiles)
                    resultat.write(smiles + '\n')
                    resultat.flush()

    vis.save_xyz_file(
        join(eval_args.model_path, 'eval/molecules/'), one_hot, charges, x,
        id_from=0, name='molecule', dataset_info=dataset_info,
        node_mask=node_mask)


def sample_only_stable_different_sizes_and_save(
        args, eval_args, device, flow, nodes_dist,
        dataset_info, n_samples=10, n_tries=50):
    assert n_tries > n_samples

    nodesxsample = nodes_dist.sample(n_tries)
    one_hot, charges, x, node_mask = sample(
        args, device, flow, dataset_info,
        nodesxsample=nodesxsample)

    counter = 0
    for i in range(n_tries):
        num_atoms = int(node_mask[i:i+1].sum().item())
        atom_type = one_hot[i:i+1, :num_atoms].argmax(2).squeeze(0).cpu().detach().numpy()
        x_squeeze = x[i:i+1, :num_atoms].squeeze(0).cpu().detach().numpy()
        mol_stable = check_stability(x_squeeze, atom_type, dataset_info)[0]

        num_remaining_attempts = n_tries - i - 1
        num_remaining_samples = n_samples - counter

        if mol_stable or num_remaining_attempts <= num_remaining_samples:
            if mol_stable:
                print('Found stable mol.')
            vis.save_xyz_file(
                join(eval_args.model_path, 'eval/molecules/'),
                one_hot[i:i+1], charges[i:i+1], x[i:i+1],
                id_from=counter, name='molecule_stable',
                dataset_info=dataset_info,
                node_mask=node_mask[i:i+1])
            counter += 1

            if counter >= n_samples:
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default="outputs/edm_1",
                        help='Specify model path')
    parser.add_argument(
        '--n_tries', type=int, default=10,
        help='N tries to find stable molecule for gif animation')
    parser.add_argument('--n_nodes', type=int, default=19,
                        help='number of atoms in molecule for gif animation')

    eval_args, unparsed_args = parser.parse_known_args()

    assert eval_args.model_path is not None

    with open(join(eval_args.model_path, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    # CAREFUL with this -->
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = 1
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = 'sum'

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    dtype = torch.float32
    utils.create_folders(args)
    print(args)

    dataset_info = get_dataset_info(args.dataset, args.remove_h)

    dataloaders, charge_scale = dataset.retrieve_dataloaders(args)
    property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
    mean, mad = property_norms['tb_inhibition']['mean'], property_norms['tb_inhibition']['mad'] 

    flow, nodes_dist, prop_dist = get_model(
        args, device, dataset_info, dataloaders['train'])
    flow.to(device)
    

    fn = 'generative_model_ema.npy' if args.ema_decay > 0 else 'generative_model.npy'
    flow_state_dict = torch.load(join(eval_args.model_path, fn),
                                 map_location=device)

    flow.load_state_dict(flow_state_dict)

    print('Sampling handful of molecules.')
    sample_different_sizes_and_save(
        args, eval_args, device, flow, nodes_dist,
        dataset_info=dataset_info, n_samples=30, full_sample=10000, mean = mean, mad = mad, return_single_molecule = False)

    # print('Sampling stable molecules.')
    # sample_only_stable_different_sizes_and_save(
    #     args, eval_args, device, flow, nodes_dist,
    #     dataset_info=dataset_info, n_samples=10, n_tries=2*10)
    # print('Visualizing molecules.')
    # vis.visualize(
    #     join(eval_args.model_path, 'eval/molecules/'), dataset_info,
    #     max_num=100, spheres_3d=True)

    # print('Sampling visualization chain.')
    # save_and_sample_chain(
    #     args, eval_args, device, flow,
    #     n_tries=eval_args.n_tries, n_nodes=eval_args.n_nodes,
    #     dataset_info=dataset_info)


# if __name__ == "__main__":
#     main()
