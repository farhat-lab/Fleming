from dataset.dataset import TB_Inhibition_Dataset
import torch
import numpy as np

if __name__ == '__main__':

    # dataset = TB_Inhibition_Dataset(raw_data_dir='data/tb_data/raw_data/Batch1and2_data.csv',
    #                                 processed_data_dir='data/tb_data/processed_data/',
    #                                 raw_data=True)
    
    # dataset = TB_Inhibition_Dataset(processed_data_dir='data/tb_data/processed_data/tb_processed_data_61896_molecules.pt',
    #                                 raw_data=False)
    


    ## Full Data
    # embed_fail_index = np.load('data/ML_Drug_Discovery/3d_conversion_failure_index.npy')

    # dataset = TB_Inhibition_Dataset(raw_data_dir='data/tb_data/raw_data/Batch1and2_data.csv',
    #                                 sdf_data_dir='data/tb_data/raw_data/Batch1and2_3d_data.csv',
    #                                 processed_data_dir='data/tb_data/processed_data/',
    #                                 raw_data=True,
    #                                 three_D_Channel=True,
    #                                 embed_fail_index=embed_fail_index)

    ## Toy Data
    # embed_fail_index = []

    # dataset = TB_Inhibition_Dataset(raw_data_dir='data/tb_data/raw_data/Batch1and2_data.csv',
    #                                 sdf_data_dir='data/tb_data/raw_data/TB_Inhibition_3d_data_with_labels.sdf',
    #                                 processed_data_dir='data/tb_data/processed_data/',
    #                                 raw_data=True,
    #                                 three_D_Channel=True,
    #                                 embed_fail_index=embed_fail_index)
    # print(dataset[0])
    
    ## Regression Data
    ## Normalized
    # dataset = TB_Inhibition_Dataset(raw_data_dir='data/tb_data/raw_data/regression_data/normalized_TB_inhibition_resazurin_62498_molecules.csv',
    #                                 sdf_data_dir='data/tb_data/raw_data/regression_data/normalized_TB_inhibition_resazurin_62498_molecules_3d.sdf',
    #                                 processed_data_dir='data/tb_data/processed_data/',
    #                                 raw_data=True,
    #                                 three_D_Channel=True,)

    ## Unnormalized
    # dataset = TB_Inhibition_Dataset(raw_data_dir='data/tb_data/raw_data/regression_data/normalized_TB_inhibition_resazurin_62498_molecules.csv',
    #                                 sdf_data_dir='data/tb_data/raw_data/regression_data/normalized_TB_inhibition_resazurin_62498_molecules_3d.sdf',
    #                                 processed_data_dir='data/tb_data/processed_data/',
    #                                 raw_data=True,
    #                                 three_D_Channel=True,)

    dataset = TB_Inhibition_Dataset(raw_data_dir='data/inhibition_training_data/tb_inhibition_A&M/final_data/CompoundFragment_Inhibition_114933_mols_inclusive_with_antibiotics.csv',
                                    sdf_data_dir='data/inhibition_training_data/tb_inhibition_A&M/final_data/final_tb_data_114933_3d.sdf',
                                    processed_data_dir='data/inhibition_training_data/tb_inhibition_A&M/final_data/',
                                    raw_data=True,
                                    three_D_Channel=True,)
    
    # negative_sample = dataset[0]

    # for item in dataset:
    #     if item.y == 1:
    #         item.idx = 1
    #         positive_sample = item
     #        break

    # torch.save([negative_sample, positive_sample], 
    #            'data/tb_data/processed_data/tb_processed_data_2_molecules.pt')

    # print(len(dataset))