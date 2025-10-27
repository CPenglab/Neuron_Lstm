# import pyswcloader
# import urllib
# import pandas as pd
# import numpy as np
# import nrrd
# import os
# from tqdm import tqdm
# from collections import defaultdict
# import networkx as nx
# from pyswcloader import brain
# from pyswcloader import swc
# import math
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import numpy as np
# from collections import Counter
# import re
# import matplotlib.pyplot as plt
# from itertools import groupby

import pandas as pd
import numpy as np
import math
import os
from collections import defaultdict
from itertools import groupby
import networkx as nx
from tqdm import tqdm
from pyswcloader import swc, brain

class SWCProcessor:
    """SWC文件处理器 - 用于原始SWC数据处理和特征提取"""
    
    def __init__(self, annotation, resolution):
        """
        初始化处理器
        
        Args:
            annotation: 脑区标注数据
            resolution: 分辨率参数
        """
        self.annotation = annotation
        self.resolution = resolution
        self.graph = None
        self.edge_data = None
    
    def process_swc_file(self, file_path):
        """
        处理单个SWC文件
        
        Args:
            file_path: SWC文件路径
            
        Returns:
            tuple: (graph对象, 边数据DataFrame, 区域路径列表)
        """
        # 预处理SWC数据
        data = swc.swc_preprocess(file_path)
        
        # 添加区域信息
        data['region'] = data.apply(
            lambda x: brain.find_region(x[['x','y','z']], self.annotation, self.resolution),
            axis=1
        )
        
        # 构建连接图
        G, edge_df = self._build_connection_graph(data, file_path)
        self.graph = G
        self.edge_data = edge_df
        
        # 提取区域路径
        regional_paths = self._get_regional_paths_optimized(G)
        
        return G, edge_df, regional_paths
    
    def _build_connection_graph(self, data, file_path):
        """构建神经元区域连接有向图"""
        G = nx.DiGraph()
        edge_data = []
        neuron_name = os.path.basename(file_path).split('.')[0]
        
        # 添加节点
        for idx in data.index:
            G.add_node(
                idx,
                region=data.loc[idx, 'region'],
                pos=(data.loc[idx, 'x'], data.loc[idx, 'y'], data.loc[idx, 'z'])
            )
        
        # 找到根节点
        root_node = data[data['parent'] == -1].index[0]
        
        # 构建连接关系
        for idx in data.index:
            if idx == root_node:
                continue
                
            parent_idx = data.loc[idx, 'parent']
            if parent_idx not in data.index:
                continue
            
            # 计算几何特征
            child_coords = data.loc[idx, ['x', 'y', 'z']].values
            parent_coords = data.loc[parent_idx, ['x', 'y', 'z']].values
            distance = math.dist(child_coords, parent_coords)
            
            # 区域信息
            child_region = data.loc[idx, 'region']
            parent_region = data.loc[parent_idx, 'region']
            
            # 添加边
            edge_attrs = {
                'from_region': parent_region,
                'to_region': child_region,
                'length': distance,
                'direction_vector': tuple(child_coords - parent_coords)
            }
            G.add_edge(parent_idx, idx, **edge_attrs)
            
            # 记录边信息
            edge_data.append({
                'parent_id': parent_idx,
                'child_id': idx,
                'parent_region': parent_region,
                'child_region': child_region,
                'length': distance,
                'x1': parent_coords[0], 'y1': parent_coords[1], 'z1': parent_coords[2],
                'x2': child_coords[0], 'y2': child_coords[1], 'z2': child_coords[2]
            })
        
        return G, pd.DataFrame(edge_data)
    
    def _get_regional_paths_optimized(self, G):
        """拓扑排序+动态规划优化获取区域路径"""
        if not G:
            return []
        
        topo_order = list(nx.topological_sort(G))
        root = topo_order[0]
        
        # 预缓存信息
        region_cache = {n: G.nodes[n]['region'] for n in G.nodes}
        edge_length_cache = {(u, v): d['length'] for u, v, d in G.edges(data=True)}
        
        # 动态规划
        dp = defaultdict(list)
        dp[root] = [{
            'node_path': [root],
            'regional_path': [region_cache[root]],
            'length': 0
        }]
        
        for node in topo_order[1:]:
            predecessors = list(G.predecessors(node))
            for pred in predecessors:
                for path in dp[pred]:
                    new_path = {
                        'node_path': path['node_path'] + [node],
                        'regional_path': path['regional_path'] + [region_cache[node]],
                        'length': path['length'] + edge_length_cache[(pred, node)]
                    }
                    dp[node].append(new_path)
        
        # 收集叶子节点路径
        leaves = [n for n in G.nodes if G.out_degree(n) == 0]
        return [path for leaf in leaves for path in dp[leaf]]
    
    @staticmethod
    def compress_path(path):
        """压缩连续相同区域的路径"""
        compressed = []
        for region, group in groupby(path):
            count = len(list(group))
            compressed.append(f"{region}({count})" if count > 1 else str(region))
        return "→".join(compressed)
    
    def extract_path_features(self, regional_paths, neuron_id, folder_name):
        """
        从区域路径中提取特征
        
        Args:
            regional_paths: 区域路径列表
            neuron_id: 神经元ID
            folder_name: 文件夹名称
            
        Returns:
            list: 路径特征字典列表
        """
        features = []
        for path_id, path_data in enumerate(regional_paths):
            compressed = self.compress_path(path_data['regional_path'])
            features.append({
                'neuron_folder': folder_name,
                'neuron_id': neuron_id,
                'path_id': path_id,
                'compressed_path': compressed,
                'path_length': len(path_data['regional_path']),
                'unique_regions': len(set(path_data['regional_path'])),
                'is_pure': len(set(path_data['regional_path'])) == 1
            })
        return features

class BatchSWCProcessor:
    """批量SWC文件处理器"""
    
    def __init__(self, annotation, resolution):
        self.annotation = annotation
        self.resolution = resolution
        self.processor = SWCProcessor(annotation, resolution)
    
    def process_folder(self, folder_path, save_results=True):
        """
        处理文件夹中的所有SWC文件
        
        Args:
            folder_path: 文件夹路径
            save_results: 是否保存结果
            
        Returns:
            list: 所有神经元的路径特征
        """
        folder_results = []
        swc_files = [f for f in os.listdir(folder_path) if f.endswith('.swc')]
        folder_name = os.path.basename(folder_path)
        
        for swc_file in tqdm(swc_files, desc=f'Processing {folder_name}'):
            try:
                swc_path = os.path.join(folder_path, swc_file)
                neuron_id = swc_file.split('.')[0]
                
                # 处理单个SWC文件
                G, edges, regional_paths = self.processor.process_swc_file(swc_path)
                
                # 提取特征
                path_features = self.processor.extract_path_features(
                    regional_paths, neuron_id, folder_name
                )
                folder_results.extend(path_features)
                
            except Exception as e:
                print(f"处理文件 {swc_file} 时出错: {str(e)}")
                continue
        
        # 保存结果
        if save_results and folder_results:
            df = pd.DataFrame(folder_results)
            output_path = os.path.join(folder_path, 'regional_paths.csv')
            df.to_csv(output_path, index=False)
        
        return folder_results
    
    def process_batch_folders(self, root_path):
        """
        批量处理多个文件夹
        
        Args:
            root_path: 根目录路径
            
        Returns:
            pd.DataFrame: 合并的结果
        """
        all_results = []
        folders = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
        
        for folder in tqdm(folders, desc='Processing folders'):
            folder_path = os.path.join(root_path, folder, 'swc_allen_space')
            
            if not os.path.exists(folder_path):
                continue
                
            # 检查是否已有结果
            result_file = os.path.join(folder_path, 'regional_paths.csv')
            if os.path.exists(result_file):
                try:
                    df = pd.read_csv(result_file)
                    all_results.extend(df.to_dict('records'))
                    continue
                except:
                    pass
            
            # 处理文件夹
            folder_results = self.process_folder(folder_path, save_results=True)
            all_results.extend(folder_results)
        
        # 保存合并结果
        if all_results:
            final_df = pd.DataFrame(all_results)
            final_output = os.path.join(root_path, 'all_regional_paths.csv')
            final_df.to_csv(final_output, index=False)
            print(f"所有结果已保存到: {final_output}")
        
        return pd.DataFrame(all_results) if all_results else pd.DataFrame()

