import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyEMD import EMD
from PyEMD import CEEMDAN
from tqdm import tqdm
import math
#
file_path = '../Result-file/WF4-66-denoised.csv'
data = pd.read_csv(file_path, dtype={'denoised_power': np.float64})
denoised_data = data['denoised_power'].values
num_ensembles = 100
noise_std = 0.2 * np.std(denoised_data)
imf_list = []
# for _ in tqdm(range(num_ensembles), desc="CEEMD Processing"):
#     noise = np.random.normal(0, noise_std, size=len(denoised_data))
#     emd = EMD()
#     imfs_pos = emd(denoised_data + noise)
#     imfs_neg = emd(denoised_data - noise)
#     min_imfs = min(len(imfs_pos), len(imfs_neg))
#     imfs_avg = (imfs_pos[:min_imfs] + imfs_neg[:min_imfs]) / 2
#     imf_list.append(imfs_avg)
# imfs_ceemd = np.mean(np.array(imf_list), axis=0)
# num_imfs = imfs_ceemd.shape[0]

# ceemd = CEEMDAN(trials=10)
# ceemd_imfs = ceemd(denoised_data)
# imf_df = pd.DataFrame(np.array(ceemd_imfs).T, columns=[f'IMF_{i + 1}' for i in range(len(ceemd_imfs))])
# imf_df.to_csv('WF4-denoised-IMFs.csv', index=False, chunksize=5000)
#
p = 0.5
imf_df = pd.read_csv('../Result-file/WF4-denoised-IMFs.csv')
# Extracting raw data
file_path = '../Result-file/WF4-66-denoised.csv'
data = pd.read_csv(file_path)
original_data = data['Power (MW)'].values
# The results of grey correlation degree are stored
gray_relativity_results = []
# The grey correlation degree with the original data is calculated
for index, col_name in enumerate(imf_df.columns):
    imf_data = imf_df[col_name].values

    # Step 1: Calculate the absolute difference
    Gray_step1 = [math.fabs(imf_data[i] - original_data[i]) for i in range(len(original_data))]

    # Find the maximum and minimum
    Gray_max = max(Gray_step1)
    Gray_min = min(Gray_step1)

    # Step 2: Calculate the grey correlation degree
    Gray_relative = [(Gray_min + p * Gray_max) / (Gray_step1[i] + p * Gray_max) for i in range(len(original_data))]

    # The grey correlation degree of this IMF component is calculated
    Gray_relative_data = sum(Gray_relative) / len(Gray_relative)

    # Save the results
    gray_relativity_results.append(Gray_relative_data)
    print(f"{col_name} -GRA: {Gray_relative_data}")

gray_relativity_df = pd.DataFrame({
    'IMF': imf_df.columns,
    'Gray_Relativity': gray_relativity_results
})

# Sort by grey correlation
gray_relativity_df = gray_relativity_df.sort_values(by='Gray_Relativity', ascending=False)
gray_relativity_df.to_csv('../Result-file/WF4-CEEMDAN_Gray_Relativity.csv', index=False)

# 1. Merge
def merge_imfs(gray_relativity_df, imf_df, threshold=0.001):
    # Sorting
    gray_relativity_df = gray_relativity_df.sort_values(by='Gray_Relativity', ascending=False)

    # The initialization list stores the merged components
    merged_imfs = []
    merged_indices = []
    merged_groups = []

    # 遍历灰色关联度值
    for i in range(len(gray_relativity_df)):
        if i in merged_indices:
            continue
        # Grey correlation value for the current component
        current_relativity = gray_relativity_df.iloc[i]['Gray_Relativity']
        # Initialize the current merge group
        current_group = [gray_relativity_df.iloc[i]['IMF']]
        current_indices = [i]

        for j in range(i + 1, len(gray_relativity_df)):
            if j in merged_indices:
                continue
            next_relativity = gray_relativity_df.iloc[j]['Gray_Relativity']
            if abs(current_relativity - next_relativity) < threshold:
                current_group.append(gray_relativity_df.iloc[j]['IMF'])
                current_indices.append(j)

        merged_imf = imf_df[current_group].sum(axis=1)
        merged_imfs.append(merged_imf)
        merged_indices.extend(current_indices)
        merged_groups.append(current_group)

    for i in range(len(gray_relativity_df)):
        if i not in merged_indices:
            merged_imfs.append(imf_df[gray_relativity_df.iloc[i]['IMF']])
            merged_groups.append([gray_relativity_df.iloc[i]['IMF']])

    # Convert the merged components to a DataFrame
    merged_imfs_df = pd.concat(merged_imfs, axis=1)
    merged_imfs_df.columns = [f'Merged_IMF_{i + 1}' for i in range(len(merged_imfs))]

    return merged_imfs_df, merged_groups

# 2. Calling the merge function
merged_imfs_df, merged_groups = merge_imfs(gray_relativity_df, imf_df)

# 3. Output the recombined signal
merged_imfs_df.to_csv('../Result-file/WF4-Merged_IMFs.csv', index=False)

# 4. Output
print(f"The number of components after reorganization: {len(merged_imfs_df.columns)}")
print("The merged component groups are as follows：")
for i, group in enumerate(merged_groups):
    print(f"Merged_IMF_{i + 1}: {group}")