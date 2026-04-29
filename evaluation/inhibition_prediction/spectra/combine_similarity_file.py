import numpy as np
from tqdm import tqdm

pre_calculated_similarity_0 = np.load('data/inhibition_training_data/tb_inhibition_A&M/pairwise_similarity_114933_0.npy')
pre_calculated_similarity_1 = np.load('data/inhibition_training_data/tb_inhibition_A&M/pairwise_similarity_114933_1.npy')
pre_calculated_similarity_2 = np.load('data/inhibition_training_data/tb_inhibition_A&M/pairwise_similarity_114933_2.npy')
pre_calculated_similarity_3 = np.load('data/inhibition_training_data/tb_inhibition_A&M/pairwise_similarity_114933_3.npy')
pre_calculated_similarity_4 = np.load('data/inhibition_training_data/tb_inhibition_A&M/pairwise_similarity_114933_4.npy')
pre_calculated_similarity_5 = np.load('data/inhibition_training_data/tb_inhibition_A&M/pairwise_similarity_114933_5.npy')

full_data = np.concatenate([pre_calculated_similarity_0,
                pre_calculated_similarity_1,
                pre_calculated_similarity_2,
                pre_calculated_similarity_3,
                pre_calculated_similarity_4,
                pre_calculated_similarity_5])

np.save('evaluation/inhibition_prediction/spectra/output/pairwise_similarity_114933.npy', full_data)


