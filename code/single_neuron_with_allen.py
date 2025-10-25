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












