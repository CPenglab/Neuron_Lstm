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

#subprocess.check_call([sys.executable, "-m", "pip", "freeze", ">", "requirements.txt"])

anno = pyswcloader.brain.read_nrrd('annotation_25.nrrd')

def get_regional_paths_optimized(G):
    """拓扑排序+动态规划优化"""
    # 获取拓扑序并确定根节点
    topo_order = list(nx.topological_sort(G))
    root = topo_order[0]
    
    # 预缓存节点区域信息
    region_cache = {n: G.nodes[n]['region'] for n in G.nodes}
    
    # 预缓存边长度信息
    edge_length_cache = {(u, v): d['length'] for u, v, d in G.edges(data=True)}
    
    # DP数据结构：每个节点对应的所有路径
    # 结构：{node: [{'node_path': [], 'regional_path': [], 'length': int}, ...]}
    dp = defaultdict(list)
    
    # 初始化根节点
    dp[root] = [{
        'node_path': [root],
        'regional_path': [region_cache[root]],
        'length': 0
    }]
    
    # 按拓扑序处理后续节点
    for node in topo_order[1:]:
        predecessors = list(G.predecessors(node))
        
        # 合并所有前驱节点的路径
        for pred in predecessors:
            for path in dp[pred]:
                new_path = {
                    'node_path': path['node_path'] + [node],
                    'regional_path': path['regional_path'] + [region_cache[node]],
                    'length': path['length'] + edge_length_cache[(pred, node)]
                }
                dp[node].append(new_path)
    
    # 收集所有叶子节点的路径
    leaves = [n for n in G.nodes if G.out_degree(n) == 0]
    return [path for leaf in leaves for path in dp[leaf]]

def compress_path(path):
    """压缩连续相同区域的路径"""
    from itertools import groupby
    compressed = []
    for region, group in groupby(path):
        count = len(list(group))
        compressed.append(f"{region}({count})" if count > 1 else str(region))
    return "→".join(compressed)

def build_connection_graph(data_path, annotation, resolution, save=False, save_path=os.getcwd()):
    """
    构建神经元区域连接有向图
    
    参数：
    data_path: SWC文件路径
    annotation: 脑区标注数据
    resolution: 分辨率参数
    save: 是否保存结果（默认False）
    save_path: 结果保存路径
    
    返回：
    (networkx.DiGraph, edge_df) 元组：
    - 有向图对象
    - 包含详细连接信息的DataFrame
    """
    # 预处理数据
    data = swc.swc_preprocess(data_path)
    neuron_name = data_path.split('/')[-1].split('.')[0]
    
    # 添加区域信息
    data['region'] = data.apply(
        lambda x: brain.find_region(x[['x','y','z']], annotation, resolution),
        axis=1
    )
    
    # 创建有向图和边列表
    G = nx.DiGraph()
    edge_data = []
    
    # 找到根节点（parent = -1）
    root_node = data[data['parent'] == -1].index[0]
    G.add_node(root_node, 
              region=data.loc[root_node, 'region'],
              pos=(data.loc[root_node, 'x'],
                   data.loc[root_node, 'y'],
                   data.loc[root_node, 'z']))
    for idx in data.index:
        G.add_node(idx,
                region=data.loc[idx, 'region'],
                pos=(data.loc[idx, 'x'],
                    data.loc[idx, 'y'],
                    data.loc[idx, 'z']))
    # 构建连接关系
    for idx in data.index:
        if idx == root_node:
            continue
            
        parent_idx = data.loc[idx, 'parent']
        if parent_idx not in data.index:
            continue
            
        # 获取区域信息
        child_region = data.loc[idx, 'region']
        parent_region = data.loc[parent_idx, 'region']
        
        # 计算几何特征
        distance = math.dist(data.loc[idx, 'x':'z'],
                           data.loc[parent_idx, 'x':'z'])
        
        # 添加边属性
        edge_attrs = {
            'from_region': parent_region,
            'to_region': child_region,
            'length': distance,
            'direction_vector': (
                data.loc[idx, 'x'] - data.loc[parent_idx, 'x'],
                data.loc[idx, 'y'] - data.loc[parent_idx, 'y'],
                data.loc[idx, 'z'] - data.loc[parent_idx, 'z']
            )
        }
        
        # 更新图结构
        G.add_edge(parent_idx, idx, **edge_attrs)
        
        # 记录边信息
        edge_data.append({
            'parent_id': parent_idx,
            'child_id': idx,
            'parent_region': parent_region,
            'child_region': child_region,
            'length': distance,
            'x1': data.loc[parent_idx, 'x'],
            'y1': data.loc[parent_idx, 'y'],
            'z1': data.loc[parent_idx, 'z'],
            'x2': data.loc[idx, 'x'],
            'y2': data.loc[idx, 'y'],
            'z2': data.loc[idx, 'z']
        })
    
    # 转换为DataFrame
    edge_df = pd.DataFrame(edge_data)
    
    # 保存结果
    if save:
        edge_path = os.path.join(save_path, f"{neuron_name}_connection_edges.csv")
        edge_df.to_csv(edge_path, index=False)
        
        graph_path = os.path.join(save_path, f"{neuron_name}_connection_graph.gml")
        nx.write_gml(G, graph_path)
    
    return G, edge_df

def process_single_folder(folder_path, annotation, resolution):
    """
    处理单个文件夹内的所有SWC文件（优化版）
    """
    folder_results = []
    swc_files = [f for f in os.listdir(folder_path) if f.endswith('.swc')]
    
    for swc_file in tqdm(swc_files, desc=f'Processing {os.path.basename(folder_path)}', leave=False):
        try:
            swc_path = os.path.join(folder_path, swc_file)
            neuron_id = swc_file.split('.')[0]
            
            # 检查是否已有该神经元的结果（在部分处理中断的情况下）
            existing_results = [r for r in folder_results if r['neuron_id'] == neuron_id]
            if existing_results:
                continue
                
            G, edges = build_connection_graph(
                data_path=swc_path,
                annotation=annotation,
                resolution=resolution,
                save=False
            )
            regional_paths = get_regional_paths_optimized(G)
            for path_id, item in enumerate(regional_paths):
                compressed = compress_path(item['regional_path'])
                folder_results.append({
                    'neuron_folder': os.path.basename(os.path.dirname(folder_path)),
                    'neuron_id': neuron_id,
                    'path_id': path_id,
                    'compressed_path': compressed,
                    'path_length': len(item['regional_path']),
                    'unique_regions': len(set(item['regional_path'])),
                    'is_pure': len(set(item['regional_path'])) == 1
                })
                
        except Exception as e:
            print(f"处理文件 {swc_file} 时出错: {str(e)}")
            continue
    
    # 保存结果前按神经元ID排序
    if folder_results:
        folder_df = pd.DataFrame(folder_results).sort_values('neuron_id')
        folder_output = os.path.join(folder_path, 'result.csv')
        folder_df.to_csv(folder_output, index=False)
    
    return folder_results

def process_all_neurons(root_path, annotation, resolution):
    """
    处理所有神经元SWC文件并合并结果
    
    参数：
    root_path: 包含所有神经元文件夹的根路径
    annotation: 脑区标注数据
    resolution: 分辨率参数
    """
    all_results = []
    neuron_folders = [d for d in os.listdir(root_path)]
    
    for folder in tqdm(neuron_folders, desc='Processing neuron folders'):
        folder_path = os.path.join(root_path, folder, 'swc_allen_space')
        result_file = os.path.join(folder_path, 'result.csv')
        
        # 检查结果文件是否存在
        if os.path.exists(result_file):
            try:
                # 直接加载已有结果
                existing_df = pd.read_csv(result_file)
                all_results.extend(existing_df.to_dict('records'))
                continue  # 跳过处理
            except Exception as e:
                print(f"加载已有结果文件 {result_file} 失败，将重新处理: {str(e)}")
        
        if not os.path.exists(folder_path):
            continue
            
        # 处理单个文件夹
        folder_results = process_single_folder(folder_path, annotation, resolution)
        all_results.extend(folder_results)
    
    # 保存最终合并结果
    final_output = os.path.join(root_path, 'combined_results.csv')
    pd.DataFrame(all_results).to_csv(final_output, index=False)
    print(f"所有结果已合并保存到: {final_output}")
 

