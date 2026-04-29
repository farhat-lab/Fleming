import pandas as pd
from rdkit import Chem

graph_pos_list = Chem.SDMolSupplier('data/tb_data/raw_data/TB_Inhibition_3d_data_with_labels.sdf')

neg_mol = graph_pos_list[0]

for mol in graph_pos_list:
    if int(mol.GetProp('_tb_inhibition')) == 1:
        pos_mol = mol
        break
    else:
        print(mol.GetProp('_tb_inhibition'))

toy_list = [pos_mol,neg_mol]

writer = Chem.SDWriter('data/tb_data/raw_data/TOY_TB_Inhibition_3d_data_with_labels.sdf')

for n in toy_list:
    writer.write(n)
    
writer.close()