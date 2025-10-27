import pandas as pd
import numpy as np
import re
from collections import defaultdict, Counter
import networkx as nx
from tqdm import tqdm
from pyswcloader import brain

class SWCPathProcessor:
    """SWC路径处理器 - 专门处理神经元路径数据"""
    
    def __init__(self, allen_brain_tree, stl_acro_dict):
        """
        初始化处理器
        
        Args:
            allen_brain_tree: 脑区树结构
            stl_acro_dict: 缩写字典
        """
        self.allen_brain_tree = allen_brain_tree
        self.stl_acro_dict = stl_acro_dict
        self.annotated_tree = self._annotate_tree_with_dict()
    
    def _annotate_tree_with_dict(self):
        """为脑区树添加注释"""
        allen_brain_tree_anno = self.allen_brain_tree
        for node in allen_brain_tree_anno.all_nodes():
            region_name = node.identifier
            if region_name in self.stl_acro_dict:
                annotation = self.stl_acro_dict[region_name]
                allen_brain_tree_anno.get_node(region_name).tag = annotation
        return allen_brain_tree_anno
    
    def _has_descendant_in_matrix(self, tree, node_id, matrix_nodes):
        """检查当前节点的任意后代是否在矩阵中（递归）"""
        for child in tree.children(node_id):
            child_node = tree.get_node(child.identifier)
            if hasattr(child_node, 'tag') and child_node.tag in matrix_nodes:
                return True
            if self._has_descendant_in_matrix(tree, child.identifier, matrix_nodes):
                return True
        return False
    
    def filter_problematic_nodes(self, directed_df, keys_for_values):
        """
        过滤有问题的节点
        
        Args:
            directed_df: 有向图数据
            keys_for_values: 键值映射字典
            
        Returns:
            set: 需要过滤的节点集合
        """
        tag_to_id = {}
        for node in self.annotated_tree.all_nodes():
            if hasattr(node, 'tag') and node.tag:
                tag_to_id[node.tag] = node.identifier

        all_matrix_nodes = set(directed_df.index) | set(directed_df.columns)
        result_set = set()

        for tag in all_matrix_nodes:
            if tag not in tag_to_id:
                continue
            node_id = tag_to_id[tag]
            if self._has_descendant_in_matrix(self.annotated_tree, node_id, all_matrix_nodes):
                result_set.add(tag)

        keys = [k for k, v in keys_for_values.items() if v in result_set]
        keys.append(0)
        return set(keys)
    
    @staticmethod
    def process_compressed_path(path, keys_set):
        """在路径当中删除指定的父节点"""
        if not isinstance(path, str):
            return path
        
        nodes = path.split('→')
        filtered_nodes = []
        
        for node in nodes:
            # 提取节点ID（去掉括号中的数字）
            clean_node = re.sub(r'\(\d+\)', '', node)
            if clean_node not in keys_set:
                filtered_nodes.append(node)
        
        return '→'.join(filtered_nodes)
    
    @staticmethod
    def merge_consecutive_nodes(path):
        """合并连续相同节点（处理区域自连通）"""
        if not isinstance(path, str):
            return path
        
        nodes = path.split('→')
        if not nodes:
            return path
        
        merged = []
        current_node = nodes[0]
        count = 1
        
        for i in range(1, len(nodes)):
            if nodes[i] == current_node:
                count += 1
            else:
                if count > 1:
                    merged.append(f"{current_node}({count})")
                else:
                    merged.append(current_node)
                current_node = nodes[i]
                count = 1
        
        # 添加最后一个节点
        if count > 1:
            merged.append(f"{current_node}({count})")
        else:
            merged.append(current_node)
        
        return '→'.join(merged)
    
    @staticmethod
    def remove_weights(path):
        """移除路径中的权重信息（括号和数字）"""
        if not isinstance(path, str):
            return path
        return re.sub(r'\(\d+\)', '', path)
    
    def replace_nodes_with_acronyms(self, path_str):
        """将路径中的节点ID替换为缩写名称"""
        if not isinstance(path_str, str):
            return path_str
        
        nodes = path_str.split('→')
        replaced_nodes = []
        
        for node in nodes:
            if node.isdigit():
                replaced_nodes.append(str(self.stl_acro_dict.get(int(node), node)))
            else:
                replaced_nodes.append(node)
        
        return '→'.join(replaced_nodes)
    
    @staticmethod
    def split_path_to_columns(df, path_column='clean_path'):
        """
        将路径拆分为起始节点、终止节点和中间节点
        
        Args:
            df: 包含路径的DataFrame
            path_column: 路径列名
            
        Returns:
            DataFrame: 添加了新列的DataFrame
        """
        df = df.copy()
        
        # 向量化操作提取首尾节点
        split_paths = df[path_column].str.split('→')
        df['start_node'] = split_paths.str[0]
        df['end_node'] = split_paths.str[-1]
        
        # 处理中间节点为集合
        def get_middle_set(path):
            if not isinstance(path, str):
                return set()
            nodes = path.split('→')
            return set(nodes[1:-1]) if len(nodes) > 2 else set()
        
        df['middle_nodes'] = df[path_column].apply(get_middle_set)
        return df
    
    @staticmethod
    def build_path_with_minimal_data(df, start, end):
        """
        使用最少的数据行构建从起点到终点的路径
        
        Args:
            df: 包含节点对和频率的DataFrame
            start: 起始节点
            end: 终止节点
            
        Returns:
            tuple: (路径字符串, 使用的边数)
        """
        sorted_df = df.sort_values('Frequency', ascending=False).reset_index(drop=True)
        start_added = False
        end_added = False
        
        for n in range(1, len(sorted_df) + 1):
            G = nx.DiGraph()
            
            for i in range(n):
                source, target = sorted_df.iloc[i]['Node Pair'].split('→')
                G.add_edge(source, target)
                
                if source == start or target == start:
                    start_added = True
                if source == end or target == end:
                    end_added = True
            
            if start_added and end_added:
                try:
                    if nx.has_path(G, start, end):
                        path = nx.shortest_path(G, source=start, target=end)
                        return "→".join(path), n
                except nx.NodeNotFound:
                    continue
        
        return "", 0
    
    def build_representative_paths(self, combined_df, save_progress_path=None):
        """
        为所有唯一的起点-终点对构建代表性路径
        
        Args:
            combined_df: 包含所有路径数据的DataFrame
            save_progress_path: 进度保存路径
            
        Returns:
            DataFrame: 包含代表性路径的结果
        """
        # 拆分路径为列
        df_with_nodes = self.split_path_to_columns(combined_df)
        unique_pairs = df_with_nodes[['start_node', 'end_node']].drop_duplicates()
        
        # 初始化结果列
        unique_pairs['path'] = ""
        unique_pairs['path_length'] = 0
        unique_pairs['edges_used'] = 0
        
        # 处理每个起点-终点对
        for index, row in tqdm(unique_pairs.iterrows(), total=len(unique_pairs), desc="构建代表性路径"):
            start_node = str(row['start_node'])
            end_node = str(row['end_node'])
            
            # 筛选目标路径
            target_paths = df_with_nodes[
                (df_with_nodes['start_node'] == start_node) & 
                (df_with_nodes['end_node'] == end_node)
            ]
            
            if len(target_paths) == 0:
                unique_pairs.at[index, 'path'] = "No paths found"
                continue
            
            # 准备中间节点并去重
            target_paths['middle_nodes'] = target_paths['middle_nodes'].apply(
                lambda x: tuple(sorted(x)) if x else ()
            )
            
            deduped_df = target_paths.drop_duplicates(
                subset=['neuron_id', 'middle_nodes'],
                keep='first'
            )
            
            # 提取所有连续节点对并统计频率
            node_pairs = []
            for path in deduped_df['clean_path']:
                nodes = path.split('→')
                pairs = [f"{nodes[i]}→{nodes[i+1]}" for i in range(len(nodes) - 1)]
                node_pairs.extend(pairs)
            
            if len(node_pairs) == 0:
                unique_pairs.at[index, 'path'] = "No valid edges"
                continue
            
            pair_counter = Counter(node_pairs)
            pair_freq_df = pd.DataFrame(pair_counter.most_common(), 
                                        columns=['Node Pair', 'Frequency'])
            
            # 构建路径
            path_str, edges_used = self.build_path_with_minimal_data(pair_freq_df, start_node, end_node)
            
            # 保存结果
            unique_pairs.at[index, 'path'] = path_str
            unique_pairs.at[index, 'edges_used'] = edges_used
            unique_pairs.at[index, 'path_length'] = len(path_str.split('→')) - 1 if path_str else 0
            
            # 定期保存进度
            if save_progress_path and (index + 1) % 100 == 0:
                unique_pairs.to_csv(save_progress_path, index=False)
                
        # 最终确保保存一次（无论是否达到整百）
        if save_progress_path:
            unique_pairs.to_csv(save_progress_path, index=False)        

        return unique_pairs