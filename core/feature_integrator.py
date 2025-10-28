import pandas as pd
import numpy as np
import re
from collections import defaultdict, Counter
import networkx as nx
from tqdm import tqdm
from pyswcloader import brain
import os

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
        self.annotated_tree = self.annotate_tree_with_dict()
    
    def annotate_tree_with_dict(self):
        """为脑区树添加注释"""
        allen_brain_tree_anno = self.allen_brain_tree
        for node in allen_brain_tree_anno.all_nodes():
            region_name = node.identifier
            if region_name in self.stl_acro_dict:
                annotation = self.stl_acro_dict[region_name]
                allen_brain_tree_anno.get_node(region_name).tag = annotation
        return allen_brain_tree_anno
    
    def has_descendant_in_matrix(self, tree, node_id, matrix_nodes):
        """检查当前节点的任意后代是否在矩阵中（递归）"""
        for child in tree.children(node_id):
            child_node = tree.get_node(child.identifier)
            if hasattr(child_node, 'tag') and child_node.tag in matrix_nodes:
                return True
            if self.has_descendant_in_matrix(tree, child.identifier, matrix_nodes):
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
            if self.has_descendant_in_matrix(self.annotated_tree, node_id, all_matrix_nodes):
                result_set.add(tag)

        keys = [k for k, v in keys_for_values.items() if v in result_set]
        keys.append(0)
        return set(keys)
    
    def merge_consecutive_nodes(self, path):
        """合并连续相同节点（处理区域自连通）"""
        if not path:  # 空路径直接返回
            return ""
        
        # 正则表达式提取区域ID和数值
        pattern = re.compile(r'(\d+)\((\d+)\)')
        nodes = []
        
        # 解析所有节点
        for part in path.split('→'):
            match = pattern.match(part)
            if match:
                region_id = int(match.group(1))
                count = int(match.group(2))
                nodes.append((region_id, count))
        
        # 合并连续相同区域ID的节点
        merged = []
        for region_id, count in nodes:
            if merged and merged[-1][0] == region_id:
                # 合并到前一个节点，累加数值
                merged[-1] = (region_id, merged[-1][1] + count)
            else:
                # 添加新节点
                merged.append((region_id, count))
        
        # 重新生成路径字符串
        return '→'.join([f"{r}({c})" for r, c in merged])
    
    def process_compressed_path(self, path, keys_to_remove):
        """在路径当中删除指定的父节点"""
        # 将路径分割成节点列表
        nodes = path.split('→')
        filtered_nodes = []
        for node in nodes:
            # 提取区域ID（括号前的部分）
            region_id_str = node.split('(')[0].strip()
            # 转换为整数
            try:
                region_id = int(region_id_str)
            except:
                # 如果格式错误，跳过该节点（根据实际情况调整）
                continue
            # 如果区域ID不在要删除的keys中，则保留该节点
            if region_id not in keys_to_remove:
                filtered_nodes.append(node)
        # 重新连接剩余节点
        return '→'.join(filtered_nodes) if filtered_nodes else ''
    
    def remove_weights(self, path):
        """移除路径中的权重信息（括号和数字）"""
        # 使用正则表达式移除所有括号及其中的内容
        return re.sub(r'\(\d+\)', '', path)
    
    def replace_nodes_with_acronyms(self, path_str):
        """将路径中的节点ID替换为缩写名称"""
        # 分割路径为节点列表
        nodes = path_str.split('→')
        # 遍历每个节点：若在字典中则替换，否则保留原值（避免缺失键错误）
        replaced_nodes = [
            str(self.stl_acro_dict.get(int(node), node))  # 处理非数字节点（如484682470）
            if node.isdigit() else node
            for node in nodes
        ]
        # 重新连接为路径字符串
        return '→'.join(replaced_nodes)
    
    def split_path_to_columns(self, df):
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
            combined_df: 已经拆分好的DataFrame（包含start_node, end_node, middle_nodes等列）
            save_progress_path: 进度保存路径
            
        Returns:
            DataFrame: 包含代表性路径的结果
        """
        if save_progress_path is None:
            save_progress_path = 'data/progress.csv'
        
        # 如果进度文件不存在，先创建并初始化
        if not os.path.exists(save_progress_path):
            print("创建新的进度文件...")
            # 直接从combined_df中提取唯一对
            unique_pairs = combined_df[['start_node', 'end_node']].drop_duplicates().reset_index(drop=True)
            
            # 确保节点类型为字符串，处理空值
            unique_pairs['start_node'] = unique_pairs['start_node'].apply(
                lambda x: str(x) if pd.notna(x) else ""
            )
            unique_pairs['end_node'] = unique_pairs['end_node'].apply(
                lambda x: str(x) if pd.notna(x) else ""
            )
            
            # 初始化结果列
            unique_pairs['path'] = ""
            unique_pairs['path_length'] = 0
            unique_pairs['edges_used'] = 0
            
            # 保存初始化结果
            unique_pairs.to_csv(save_progress_path, index=False)
            print(f"初始化完成: 共找到 {len(unique_pairs)} 个路径对")
        else:
            # 如果进度文件存在，直接加载
            print(f"从断点继续: {save_progress_path}")
            unique_pairs = pd.read_csv(save_progress_path, keep_default_na=False)
            # 确保节点类型为字符串，并去除小数点，处理空值
            unique_pairs['start_node'] = unique_pairs['start_node'].apply(
                lambda x: str(x).replace('.0', '') if pd.notna(x) and x != "" else ""
            )
            unique_pairs['end_node'] = unique_pairs['end_node'].apply(
                lambda x: str(x).replace('.0', '') if pd.notna(x) and x != "" else ""
            )
        
        # 确保combined_df中的节点类型也是字符串，处理空值
        combined_df['start_node'] = combined_df['start_node'].apply(
            lambda x: str(x) if pd.notna(x) else ""
        )
        combined_df['end_node'] = combined_df['end_node'].apply(
            lambda x: str(x) if pd.notna(x) else ""
        )
        
        total_pairs = len(unique_pairs)
        
        # 计算已处理的数量 - 只要path不是空字符串就认为是已处理
        processed_count = 0
        for i in range(total_pairs):
            if pd.notna(unique_pairs.loc[i, 'path']) and unique_pairs.loc[i, 'path'] != "":
                processed_count += 1
        
        print(f"已处理 {processed_count} 个路径对，剩余 {total_pairs - processed_count} 个待处理")
        
        # 如果没有需要处理的路径对，直接返回
        if processed_count == total_pairs:
            print("所有路径对已处理完成")
            return unique_pairs
        
        # 创建需要处理的索引列表
        indices_to_process = []
        for i in range(total_pairs):
            if pd.isna(unique_pairs.loc[i, 'path']) or unique_pairs.loc[i, 'path'] == "":
                indices_to_process.append(i)
        
        # 使用tqdm进度条处理需要处理的行
        for i in tqdm(indices_to_process, desc="构建代表性路径", total=len(indices_to_process)):
            start_node = unique_pairs.loc[i, 'start_node']
            end_node = unique_pairs.loc[i, 'end_node']
            
            # 如果节点为空，跳过
            if start_node == "" or end_node == "":
                unique_pairs.at[i, 'path'] = "Invalid nodes"
                processed_count += 1
                continue
                
            # 筛选目标路径 - 直接使用已经拆分好的combined_df
            target_paths = combined_df[
                (combined_df['start_node'] == start_node) & 
                (combined_df['end_node'] == end_node)
            ].copy()
            
            if len(target_paths) == 0:
                unique_pairs.at[i, 'path'] = "No paths found"
                processed_count += 1
            else:
                # 准备中间节点并去重
                target_paths.loc[:, 'middle_nodes'] = target_paths['middle_nodes'].apply(
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
                    unique_pairs.at[i, 'path'] = "No valid edges"
                    processed_count += 1
                else:
                    pair_counter = Counter(node_pairs)
                    pair_freq_df = pd.DataFrame(pair_counter.most_common(), 
                                                columns=['Node Pair', 'Frequency'])
                    
                    # 构建路径
                    path_str, edges_used = self.build_path_with_minimal_data(pair_freq_df, start_node, end_node)
                    
                    # 保存结果
                    unique_pairs.at[i, 'path'] = path_str
                    unique_pairs.at[i, 'edges_used'] = edges_used
                    unique_pairs.at[i, 'path_length'] = len(path_str.split('→')) - 1 if path_str else 0
                    processed_count += 1
            
            # 定期保存进度
            if save_progress_path and (processed_count % 10 == 0 or processed_count == total_pairs):
                # 确保保存时节点类型为字符串，处理空值
                save_df = unique_pairs.copy()
                save_df['start_node'] = save_df['start_node'].apply(
                    lambda x: str(x) if pd.notna(x) else ""
                )
                save_df['end_node'] = save_df['end_node'].apply(
                    lambda x: str(x) if pd.notna(x) else ""
                )
                save_df.to_csv(save_progress_path, index=False)
                print(f"进度已保存: 已处理 {processed_count}/{total_pairs} 个路径对")
        
        # 最终保存一次
        if save_progress_path:
            # 确保保存时节点类型为字符串，处理空值
            save_df = unique_pairs.copy()
            save_df['start_node'] = save_df['start_node'].apply(
                lambda x: str(x) if pd.notna(x) else ""
            )
            save_df['end_node'] = save_df['end_node'].apply(
                lambda x: str(x) if pd.notna(x) else ""
            )
            save_df.to_csv(save_progress_path, index=False)
            print(f"最终结果已保存: 共处理 {total_pairs} 个路径对")
        
        return unique_pairs