import pyswcloader
import urllib
import pandas as pd
import numpy as np
import nrrd
import os
from tqdm import tqdm
from collections import defaultdict
import networkx as nx
from pyswcloader import brain
from pyswcloader import swc
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from collections import Counter
import re
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import deque
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM, Bidirectional, Concatenate, Reshape, TimeDistributed
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Embedding, Reshape, Concatenate, LSTM, Dense, TimeDistributed, Bidirectional, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.layers import PReLU  # 导入PReLU层
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from scipy import stats
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Concatenate, LSTM, Dense, Reshape, TimeDistributed, Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from heapq import heappush, heappop
import ast
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



#############
#####对allen实验的数据进行修改，这个数据来自/public/home/sangchl/毕业课题/allen脑图谱/投射实验/合并result
#####这一步主要是吧有相同投射区域的位置合并去均值

file = pd.read_csv('/public/home/sangchl/毕业课题/2025.5.12模型/allen_data/merged_results.csv')

# 选择数值列，排除 'experiment_id', 'injection_position', '_meta'
numeric_cols = file.columns.difference(["experiment_id", "injection_position", "_meta"])

# 按 'experiment_id' 进行分组，并计算数值列的均值，同时保留 'injection_position'
merged_df = file.groupby(["experiment_id", "injection_position"])[numeric_cols].mean().reset_index()

merged_df = merged_df[merged_df['injection_position'] != 'unknown']

# 筛选以 '_left' 结尾的 injection_position
left_positions = merged_df[merged_df['injection_position'].str.endswith('_left')]
left_matrix = left_positions.set_index('injection_position')

# 筛选以 '_right' 结尾的 injection_position
right_positions = merged_df[merged_df['injection_position'].str.endswith('_right')]
right_matrix = right_positions.set_index('injection_position')

def split_columns_by_suffix(matrix, suffix):
    """筛选列名以指定后缀结尾的列"""
    return matrix.loc[:, matrix.columns.str.endswith(suffix)]


# 从 left_matrix 中提取 _left 列（目标脑区左）
left_matrix_left_columns = split_columns_by_suffix(left_matrix, '_left')

# 从 left_matrix 中提取 _right 列（目标脑区右）
left_matrix_right_columns = split_columns_by_suffix(left_matrix, '_right')

# 从 right_matrix 中提取 _left 列（目标脑区左）
right_matrix_left_columns = split_columns_by_suffix(right_matrix, '_left')

# 从 right_matrix 中提取 _right 列（目标脑区右）
right_matrix_right_columns = split_columns_by_suffix(right_matrix, '_right')

# 去除列明，行名（索引）的 _right 后缀
right_matrix_right_columns.columns = right_matrix_right_columns.columns.str.replace('_right', '')
right_matrix_right_columns.index = right_matrix_right_columns.index.str.replace('_right', '')

left_matrix_left_columns.columns = left_matrix_left_columns.columns.str.replace('_left', '')
left_matrix_left_columns.index = left_matrix_left_columns.index.str.replace('_left', '')

###先整理一个同侧投射的
ipsi_matrix = pd.concat([left_matrix_left_columns, right_matrix_right_columns])

# 按行名分组并计算均值（相同行名的数据会自动合并），不需要合并了
ipsi_matrix_result = ipsi_matrix


# 去除列明，行名（索引）的 _right 后缀
right_matrix_left_columns.columns = right_matrix_left_columns.columns.str.replace('_left', '')
right_matrix_left_columns.index = right_matrix_left_columns.index.str.replace('_right', '')

left_matrix_right_columns.columns = left_matrix_right_columns.columns.str.replace('_right', '')
left_matrix_right_columns.index = left_matrix_right_columns.index.str.replace('_left', '')

###这是对侧投射
contra_matrix = pd.concat([right_matrix_left_columns, left_matrix_right_columns])

# 按行名分组并计算均值（相同行名的数据会自动合并）
contra_matrix_result = contra_matrix

keys_str = [str(k) for k in keys]
# 删除行和列
ipsi_matrix_result = ipsi_matrix_result.drop(index=keys_str, errors='ignore') \
                                   .drop(columns=keys_str, errors='ignore')

contra_matrix_result = contra_matrix_result.drop(index=keys_str, errors='ignore') \
                                   .drop(columns=keys_str, errors='ignore')

# 替换行名
ipsi_matrix_result.index = ipsi_matrix_result.index.map(
    lambda x: stl_acro_dict.get(int(x), x) if str(x).isdigit() else x
)

# 替换列名
ipsi_matrix_result.columns = ipsi_matrix_result.columns.map(
    lambda x: stl_acro_dict.get(int(x), x) if str(x).isdigit() else x
)


# 替换行名
contra_matrix_result.index = contra_matrix_result.index.map(
    lambda x: stl_acro_dict.get(int(x), x) if str(x).isdigit() else x
)

# 替换列名
contra_matrix_result.columns = contra_matrix_result.columns.map(
    lambda x: stl_acro_dict.get(int(x), x) if str(x).isdigit() else x
)

######保存下来
ipsi_matrix_result.to_csv("/public/home/sangchl/毕业课题/2025.5.12模型/data/ipsi_matrix_result.csv")


def get_leaf_nodes(tree):
    leaf_nodes = []
    
    # 获取树中所有节点的ID
    for node in tree.all_nodes():
        if len(tree.children(node.identifier)) == 0:  # 如果没有子节点，说明是叶子节点
            leaf_nodes.append(node.identifier)  # 获取叶子节点的标识符或其他属性
    
    return leaf_nodes

# 假设 allen_brain_tree 是一个 treelib 树对象
leaf_nodes = get_leaf_nodes(allen_brain_tree)

def get_leaf_nodes_with_parents(tree):
    leaf_nodes_data = []
    
    # 遍历所有节点
    for node in tree.all_nodes():
        if len(tree.children(node.identifier)) == 0:  # 如果是叶子节点
            parent_node = tree.parent(node.identifier)  # 获取父节点
            parent_node_id = parent_node.identifier if parent_node else None  # 获取父节点ID，如果没有父节点则为None
            leaf_nodes_data.append([node.identifier, parent_node_id])  # 将节点ID和父节点ID添加到数据中
    
    return leaf_nodes_data

# 假设 allen_brain_tree 是一个 treelib 树对象
leaf_nodes_with_parents = get_leaf_nodes_with_parents(allen_brain_tree)

node_parent_node = pd.DataFrame(leaf_nodes_with_parents, columns=['node', 'parentnode'])

node_parent_node['node'] = node_parent_node['node'].replace(stl_acro_dict)
node_parent_node['parentnode'] = node_parent_node['parentnode'].replace(stl_acro_dict)

# 步骤1：从 node_parent_node 创建列名到父节点的映射字典
column_to_parent = dict(zip(node_parent_node['node'], node_parent_node['parentnode']))

# 步骤2：为每个列名获取对应的父节点（未找到的设为"root"）
column_names = ipsi_matrix_result.columns.tolist()
parent_names = [column_to_parent.get(col, "root") for col in column_names]

# 步骤3：创建多级列索引 [父节点, 原始列名]
multi_columns = pd.MultiIndex.from_arrays(
    [parent_names, column_names],
    names=["Parent", "Region"]
)

# 步骤4：应用新列索引
ipsi_matrix_result_with_parent_columns = ipsi_matrix_result.copy()
ipsi_matrix_result_with_parent_columns.columns = multi_columns

ipsi_matrix_result_sorted = ipsi_matrix_result_with_parent_columns.sort_index(
    axis=1,          # 对列排序
    level="Parent",  # 按父节点层级排序
    sort_remaining=True  # 如果父节点相同，继续按子节点排序
)

# 从 node_parent_node 创建 {节点: 父节点} 的字典
row_to_parent = dict(zip(node_parent_node['node'], node_parent_node['parentnode']))

# 为每一行获取父节点（未找到的设为 "root"）
row_parents = [
    row_to_parent.get(row, "root") 
    for row in ipsi_matrix_result_sorted.index
]

ipsi_matrix_result_sorted.insert(0, "parent_node", row_parents)

ipsi_matrix_result_final = ipsi_matrix_result_sorted.sort_values(
    by="parent_node",  # 按父节点排序
    ascending=True,    # 升序
    kind='mergesort'   # 稳定排序（保持相同父节点的行原始顺序）
)

ipsi_matrix_result_final.to_csv('/public/home/sangchl/毕业课题/2025.5.12模型/data/ipsi_matrix_result.csv')


##########保留前75%表达值
def keep_above_percentile(row, percentile=75):
    """保留大于指定分位数的值"""
    threshold = np.percentile(row, percentile)
    new_row = row.copy()
    new_row[row < threshold] = 0
    return new_row

# 应用处理
result_filtered = ipsi_matrix_result_final.iloc[:, 1:].apply(
    keep_above_percentile, 
    axis=1, 
    percentile=75
)

# 假设 result_filtered 是你的数据框
def normalize_row_nonzero_min(row):
    # 排除零值后找到每行的非零最小值
    nonzero_min = row[row > 0].min()
    
    # 如果该行所有值为0，则返回原始行（或全零）
    if nonzero_min == np.inf:  # 如果没有非零值
        return row  # 或者 return np.zeros_like(row) 
    # 归一化：每个元素减去非零最小值再除以最大值与非零最小值的差
    row_max = row.max()
    return (row - nonzero_min) / (row_max - nonzero_min)

# 对每行应用归一化
result_filtered_normalized = result_filtered.apply(normalize_row_nonzero_min, axis=1)

result_filtered_normalized[result_filtered_normalized < 0] = 0

result_filtered.insert(0, "parent_node", ipsi_matrix_result_final["parent_node"])

result_filtered_mean = result_filtered.groupby('injection_position').mean()

result_filtered_mean.to_csv('/public/home/sangchl/毕业课题/2025.5.12模型/中间文件查看/result_filtered_mean.csv')












#########将两者数据结合

unique_start_node_values = unique_pairs['start_node'].unique()

def replace_node_id(node_id, acronym_dict):
    if node_id == '':  # 处理空字符串
        return 'Unknown'
    try:
        int_id = int(node_id)  # 转换为整数
        return acronym_dict.get(int_id, node_id)  # 存在则替换，否则保留原值
    except ValueError:  # 非数字字符串（如 'ABC'）
        return node_id

# 应用替换
replaced_nodes = np.array([
    replace_node_id(node, stl_acro_dict) 
    for node in unique_start_node_values
])


valid_injection_positions = set(replaced_nodes) - {'Unknown'}
filtered_df = result_filtered_mean.loc[
    result_filtered_mean.index.isin(valid_injection_positions)
]

result_matrix = []

# 遍历filtered_df的每一行（每个实验）
for exp_idx, exp_row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="处理实验中"):
    # 获取注射位置（假设是索引的第一部分）
    if isinstance(exp_idx, tuple):
        injection_position = exp_idx[0]
        experiment_id = f"{exp_idx[0]}_{exp_idx[1]}"  # 创建唯一实验ID
    else:
        injection_position = exp_idx
        experiment_id = str(exp_idx)
    
    # 在unique_pairs中匹配起始点
    matched_paths = unique_pairs[unique_pairs['replaced_start_node'] == injection_position]
    
    # 如果没有匹配路径，跳过
    if matched_paths.empty:
        continue
    
    # 处理每条匹配路径
    for _, path_row in matched_paths.iterrows():
        # 获取路径中的脑区序列
        path_regions = path_row['replaced_path'].split('→')
        
        # 存储路径强度信息（包含首节点）
        path_info = {
            'experiment_id': experiment_id,
            'injection_position': injection_position,
            'path': path_row['replaced_path'],
            'path_length': len(path_regions) - 1,
            'region_intensities': [],
            'intensity_sequence': []  # 单独存储强度序列用于计算
        }
        
        valid_path = True
        
        # 首先添加首节点强度（注射位置自身）
        injection_cols = [col for col in filtered_df.columns 
                         if col[1] == injection_position]
        if injection_cols:
            injection_intensity = max(exp_row[col] for col in injection_cols)
            path_info['region_intensities'].append(f"{injection_position}:{injection_intensity:.6f}")
            path_info['intensity_sequence'].append(injection_intensity)
        else:
            valid_path = False
        
        # 计算路径上其他脑区的强度（从第二个开始）
        for region in path_regions[1:]:
            # 查找该脑区的强度列
            region_cols = [col for col in filtered_df.columns 
                          if col[1] == region]  # 假设列是(Parent, Region)格式
            
            if region_cols:
                # 取所有匹配列中的最大强度值
                intensity = max(exp_row[col] for col in region_cols)
                path_info['region_intensities'].append(f"{region}:{intensity:.6f}")
                path_info['intensity_sequence'].append(intensity)
            else:
                valid_path = False
                break
        
        # 如果路径有效，计算总强度并保存结果
        if valid_path and len(path_info['intensity_sequence']) == len(path_regions):
            # 计算总强度（所有节点强度的乘积，包括首节点）
            path_info['total_intensity'] = np.prod(path_info['intensity_sequence'])
            
            # 将强度信息转为字符串
            path_info['region_intensities'] = " → ".join(path_info['region_intensities'])
            
            # 添加到结果矩阵
            result_matrix.append(path_info)

if result_matrix:
    results_df = pd.DataFrame(result_matrix)
    
    # 重新排列列顺序
    results_df = results_df[['experiment_id', 'injection_position', 'path',
                            'path_length', 'region_intensities', 
                            'total_intensity']]
    
    # 保存结果
    results_df.to_csv('/public/home/sangchl/毕业课题/2025.5.12模型/中间文件查看/path_intensity_results_with_source.csv', 
                     index=False)

filtered_results_df = results_df[results_df['path_length'] >= 5]

filtered_results_df['strength'] = filtered_results_df['region_intensities'].apply(
    lambda x: [float(i.split(':')[1]) for i in x.split('→')]  # 提取每个节点后的数字并转换为 float
)



filtered_results_df = filtered_results_df[filtered_results_df['strength'].apply(lambda x: all(val != 0 for val in x))]

filtered_results_df = filtered_results_df.drop(filtered_results_df.columns[0], axis=1)

filtered_results_df = filtered_results_df.drop(['region_intensities', 'total_intensity'], axis=1)


# 添加起始点和终止点列
filtered_results_df['start'] = filtered_results_df['path'].apply(
    lambda x: x.split('→')[0] if '→' in x else x
)

filtered_results_df['end'] = filtered_results_df['path'].apply(
    lambda x: x.split('→')[-1] if '→' in x else x
)

filtered_results_df.to_csv('/public/home/sangchl/毕业课题/2025.5.12模型/中间文件查看/path_intensity_results_with_source.csv', 
                    index=False,
                    encoding='utf-8')
