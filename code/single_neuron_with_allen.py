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


file1 = pd.read_csv('/public/home/sangchl/毕业课题/2025.5.12模型/data/海马.csv')
file2 = pd.read_csv('/public/home/sangchl/毕业课题/2025.5.12模型/data/前额叶皮层.csv')
file3 = pd.read_csv('/public/home/sangchl/毕业课题/2025.5.12模型/data/下丘脑.csv')

allen_brain_tree = pyswcloader.brain.allen_brain_tree('/public/home/sangchl/毕业课题/2025.5.12模型/data/1.json')
stl_acro_dict = pyswcloader.brain.acronym_dict('/public/home/sangchl/毕业课题/2025.5.12模型/data/1.json')


######下面不是每次都要实行，这一步是对哪些区域以及其子区域都在的区域进行删除
combined_df = pd.concat([file1, file2, file3], ignore_index=True)

def annotate_tree_with_dict(allen_brain_tree, stl_acro_dict):
    # 遍历 allen_brain_tree 中的所有节点
    for node in allen_brain_tree.all_nodes():
        region_name = node.identifier  # 获取节点的标识符，即区域名称
        
        # 检查字典中是否有该区域的注释
        if region_name in stl_acro_dict:
            # 获取注释
            annotation = stl_acro_dict[region_name]
            
            # 将注释添加到节点的自定义属性中
            allen_brain_tree.get_node(region_name).tag = annotation  # 使用tag属性保存注释
    return allen_brain_tree

def has_descendant_in_matrix(tree, node_id, matrix_nodes):
    """检查当前节点的任意后代是否在矩阵中（递归）"""
    for child in tree.children(node_id):
        child_node = tree.get_node(child.identifier)
        # 如果子节点有 tag 且在矩阵中，返回 True
        if hasattr(child_node, 'tag') and child_node.tag in matrix_nodes:
            return True
        # 否则递归检查子节点的后代
        if has_descendant_in_matrix(tree, child.identifier, matrix_nodes):
            return True
    return False

allen_brain_tree_anno=annotate_tree_with_dict(allen_brain_tree, stl_acro_dict)

tag_to_id = {}

for node in allen_brain_tree_anno.all_nodes():
    if hasattr(node, 'tag') and node.tag:  # 确保节点有 tag
        tag_to_id[node.tag] = node.identifier  # 例如 {"PMv": 123}

all_matrix_nodes = set(directed_df.index) | set(directed_df.columns)

result_set = set()

for tag in all_matrix_nodes:
    if tag not in tag_to_id:
        continue  # 跳过无对应 identifier 的 tag
    
    node_id = tag_to_id[tag]
    if has_descendant_in_matrix(allen_brain_tree_anno, node_id, all_matrix_nodes):
        result_set.add(tag)


keys = [k for k, v in keys_for_values.items() if v in result_set]
keys.append(0)

# 将keys转换为集合以提高查找效率
keys_set = set(keys)
################
###在路径当中删除这些父节点
combined_df['processed_compressed_path'] = combined_df['compressed_path'].apply(
    lambda x: process_compressed_path(x, keys_set)
)

#########在上一步可能产生一些区域自连通，把这些节点合并一下
combined_df["merged_compressed_path"] = combined_df["processed_compressed_path"].apply(merge_consecutive_nodes)

def remove_weights(path):
    # 使用正则表达式移除所有括号及其中的内容
    return re.sub(r'\(\d+\)', '', path)

# 创建新列'clean_path'，其中移除了括号和权重
combined_df['clean_path'] = combined_df['merged_compressed_path'].apply(remove_weights)

combined_df['repalce_path'] = combined_df['clean_path'].apply(
    lambda x: replace_nodes_with_acronyms(x, stl_acro_dict)
)

def split_path_to_columns(df):
    """
    将clean_path拆分为三列：
    - start_node: 起始节点（字符串）
    - end_node: 终止节点（字符串）
    - middle_nodes: 中间节点的集合（set类型）
    """
    # 向量化操作提取首尾节点（高效）
    split_paths = df['clean_path'].str.split('→')
    df['start_node'] = split_paths.str[0]
    df['end_node'] = split_paths.str[-1]
    
    # 处理中间节点为集合
    def get_middle_set(path):
        if not isinstance(path, str):
            return set()
        nodes = path.split('→')
        return set(nodes[1:-1]) if len(nodes) > 2 else set()
    
    df['middle_nodes'] = df['clean_path'].apply(get_middle_set)
    return df

combined_df = split_path_to_columns(combined_df)

unique_pairs = combined_df[['start_node', 'end_node']].drop_duplicates()


unique_pairs['path'] = ""  # 初始化为空字符串
unique_pairs['path_length'] = 0  # 路径长度
unique_pairs['edges_used'] = 0  # 使用的边数

# 定义路径构建函数（修复版）
def build_path_with_minimal_data(df, start, end):
    """
    使用最少的数据行构建从起点到终点的路径
    """
    # 按频率降序排序
    sorted_df = df.sort_values('Frequency', ascending=False).reset_index(drop=True)
    
    # 记录起始点和终止点是否被添加
    start_added = False
    end_added = False
    
    # 逐步增加使用的数据行
    for n in range(1, len(sorted_df) + 1):
        # 创建有向图
        G = nx.DiGraph()
        
        # 添加前n行作为边
        for i in range(n):
            source, target = sorted_df.iloc[i]['Node Pair'].split('→')
            G.add_edge(source, target)
            
            # 检查起始点和终止点是否被添加
            if source == start or target == start:
                start_added = True
            if source == end or target == end:
                end_added = True
        
        # 只有当起始点和终止点都存在时才检查路径
        if start_added and end_added:
            try:
                if nx.has_path(G, start, end):
                    # 找到最短路径
                    path = nx.shortest_path(G, source=start, target=end)
                    return "→".join(path), n
            except nx.NodeNotFound:
                # 如果节点不存在，继续添加更多边
                continue
    
    return "", 0  # 未找到路径

# 主处理循环,这里面的combined_df，test_data也要考虑一下
for index, row in unique_pairs.iterrows():
    start_node = str(row['start_node'])
    end_node = str(row['end_node'])
    
    print(f"处理对: {start_node} → {end_node} ({index+1}/{len(unique_pairs)})")
    
    # 步骤1: 筛选目标路径
    target_paths = combined_df[
        (combined_df['start_node'] == start_node) & 
        (combined_df['end_node'] == end_node)
    ]
    
    # 如果没有找到任何路径，跳过
    if len(target_paths) == 0:
        unique_pairs.at[index, 'path'] = "No paths found"
        continue
    
    # 步骤2: 准备中间节点并去重
    target_paths['middle_nodes'] = target_paths['middle_nodes'].apply(
        lambda x: tuple(sorted(x)) if x else ()
    )
    
    deduped_df = target_paths.drop_duplicates(
        subset=['neuron_id', 'middle_nodes'],
        keep='first'
    )
    
    # 步骤3: 提取所有连续节点对并统计频率
    node_pairs = []
    for path in deduped_df['clean_path']:
        nodes = path.split('→')
        pairs = [f"{nodes[i]}→{nodes[i+1]}" for i in range(len(nodes) - 1)]
        node_pairs.extend(pairs)
    
    # 统计节点对频率
    if len(node_pairs) == 0:
        unique_pairs.at[index, 'path'] = "No valid edges"
        continue
    
    pair_counter = Counter(node_pairs)
    pair_freq_df = pd.DataFrame(pair_counter.most_common(), 
                                columns=['Node Pair', 'Frequency'])
    
    # 步骤4: 构建路径
    path_str, edges_used = build_path_with_minimal_data(pair_freq_df, start_node, end_node)
    
    # 保存结果
    unique_pairs.at[index, 'path'] = path_str
    unique_pairs.at[index, 'edges_used'] = edges_used
    unique_pairs.at[index, 'path_length'] = len(path_str.split('→')) - 1 if path_str else 0
    
    # 每100个对保存一次进度
    if (index + 1) % 100 == 0:
        unique_pairs.to_csv('/public/home/sangchl/毕业课题/2025.5.12模型/中间文件查看/unique_pairs_with_paths_partial.csv', index=False)
        print(f"已保存前 {index+1} 个对的进度")

def replace_nodes_with_acronyms(path_str, acronym_dict):
    """将路径中的节点ID替换为缩写名称
    Args:
        path_str (str): 原始路径字符串（如 '382→466→971'）
        acronym_dict (dict): 节点ID到缩写名称的映射字典
    Returns:
        str: 替换后的路径字符串
    """
    # 分割路径为节点列表
    nodes = path_str.split('→')
    # 遍历每个节点：若在字典中则替换，否则保留原值（避免缺失键错误）
    replaced_nodes = [
        str(acronym_dict.get(int(node), node))  # 处理非数字节点（如484682470）
        if node.isdigit() else node
        for node in nodes
    ]
    # 重新连接为路径字符串
    return '→'.join(replaced_nodes)


# 应用函数到整个DataFrame的path列
unique_pairs['replaced_path'] = unique_pairs['path'].apply(
    lambda x: replace_nodes_with_acronyms(x, stl_acro_dict)
)

unique_pairs['replaced_start_node'] = unique_pairs['start_node'].apply(
    lambda x: replace_nodes_with_acronyms(x, stl_acro_dict)
)








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


