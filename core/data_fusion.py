import pandas as pd
import numpy as np
import re
import requests
import os
import time
from tqdm import tqdm
from collections import defaultdict
from pyswcloader import brain
import nrrd

class AllenDataFusion:
    """Allen实验数据融合器 - 处理Allen实验数据与路径数据的融合"""
    
    def __init__(self, anno, allen_brain_tree, stl_acro_dict):
        """
        初始化融合器
        
        Args:
            allen_brain_tree: 脑区树结构
            stl_acro_dict: 缩写字典
        """
        self.allen_brain_tree = allen_brain_tree
        self.stl_acro_dict = stl_acro_dict
        self.anno = anno
    
    def load_and_preprocess_allen_data(self, file_path):
        """
        加载和预处理Allen实验数据
        
        Args:
            file_path: Allen数据文件路径
            
        Returns:
            DataFrame: 预处理后的数据
        """
        file = pd.read_csv(file_path)
        
        # 选择数值列，排除非数值列
        numeric_cols = file.columns.difference(["experiment_id", "injection_position", "_meta"])
        
        # 按实验ID和注射位置分组求均值
        merged_df = file.groupby(["experiment_id", "injection_position"])[numeric_cols].mean().reset_index()
        merged_df = merged_df[merged_df['injection_position'] != 'unknown']
        
        return merged_df
    
    def create_ipsi_contra_matrices(self, merged_df):
        """
        创建同侧和对侧投射矩阵
        
        Args:
            merged_df: 预处理后的Allen数据
            
        Returns:
            tuple: (同侧矩阵, 对侧矩阵)
        """
        # 分离左侧和右侧注射位置
        left_positions = merged_df[merged_df['injection_position'].str.endswith('_left')]
        right_positions = merged_df[merged_df['injection_position'].str.endswith('_right')]
        
        left_matrix = left_positions.set_index('injection_position')
        right_matrix = right_positions.set_index('injection_position')
        
        def split_columns_by_suffix(matrix, suffix):
            """筛选列名以指定后缀结尾的列"""
            return matrix.loc[:, matrix.columns.str.endswith(suffix)]
        
        # 分离不同侧别的列
        left_matrix_left = split_columns_by_suffix(left_matrix, '_left')
        left_matrix_right = split_columns_by_suffix(left_matrix, '_right')
        right_matrix_left = split_columns_by_suffix(right_matrix, '_left')
        right_matrix_right = split_columns_by_suffix(right_matrix, '_right')
        
        # 清理列名和索引
        right_matrix_right.columns = right_matrix_right.columns.str.replace('_right', '')
        right_matrix_right.index = right_matrix_right.index.str.replace('_right', '')
        left_matrix_left.columns = left_matrix_left.columns.str.replace('_left', '')
        left_matrix_left.index = left_matrix_left.index.str.replace('_left', '')
        right_matrix_left.columns = right_matrix_left.columns.str.replace('_left', '')
        right_matrix_left.index = right_matrix_left.index.str.replace('_right', '')
        left_matrix_right.columns = left_matrix_right.columns.str.replace('_right', '')
        left_matrix_right.index = left_matrix_right.index.str.replace('_left', '')
        
        # 合并同侧和对侧矩阵
        ipsi_matrix = pd.concat([left_matrix_left, right_matrix_right])
        contra_matrix = pd.concat([right_matrix_left, left_matrix_right])
        
        return ipsi_matrix, contra_matrix
    
    def filter_matrix_nodes(self, matrix, keys_to_remove):
        """
        过滤矩阵中的指定节点
        
        Args:
            matrix: 要过滤的矩阵
            keys_to_remove: 要移除的节点列表
            
        Returns:
            DataFrame: 过滤后的矩阵
        """
        keys_str = [str(k) for k in keys_to_remove]
        filtered_matrix = matrix.drop(index=keys_str, errors='ignore') \
                              .drop(columns=keys_str, errors='ignore')
        
        # 替换行列名为缩写名称
        filtered_matrix.index = filtered_matrix.index.map(
            lambda x: self.stl_acro_dict.get(int(x), x) if str(x).isdigit() else x
        )
        filtered_matrix.columns = filtered_matrix.columns.map(
            lambda x: self.stl_acro_dict.get(int(x), x) if str(x).isdigit() else x
        )
        
        return filtered_matrix
    
    def create_hierarchical_matrix(self, matrix):
        """
        创建层次化矩阵（按父节点分组）
        
        Args:
            matrix: 原始矩阵
            
        Returns:
            DataFrame: 层次化矩阵
        """
        # 获取叶子节点及其父节点
        leaf_nodes_data = self._get_leaf_nodes_with_parents()
        node_parent_df = pd.DataFrame(leaf_nodes_data, columns=['node', 'parentnode'])
        
        # 替换为缩写名称
        node_parent_df['node'] = node_parent_df['node'].replace(self.stl_acro_dict)
        node_parent_df['parentnode'] = node_parent_df['parentnode'].replace(self.stl_acro_dict)
        
        # 创建列层次结构
        column_to_parent = dict(zip(node_parent_df['node'], node_parent_df['parentnode']))
        column_names = matrix.columns.tolist()
        parent_names = [column_to_parent.get(col, "root") for col in column_names]
        
        multi_columns = pd.MultiIndex.from_arrays(
            [parent_names, column_names],
            names=["Parent", "Region"]
        )
        
        matrix_with_parents = matrix.copy()
        matrix_with_parents.columns = multi_columns
        
        # 排序
        matrix_sorted = matrix_with_parents.sort_index(
            axis=1, level="Parent", sort_remaining=True
        )
        
        # 添加行父节点并排序
        row_to_parent = dict(zip(node_parent_df['node'], node_parent_df['parentnode']))
        row_parents = [row_to_parent.get(row, "root") for row in matrix_sorted.index]
        matrix_sorted.insert(0, "parent_node", row_parents)
        
        matrix_final = matrix_sorted.sort_values(by="parent_node", ascending=True, kind='mergesort')
        
        return matrix_final
    
    def _get_leaf_nodes_with_parents(self):
        """获取叶子节点及其父节点信息"""
        leaf_nodes_data = []
        
        for node in self.allen_brain_tree.all_nodes():
            if len(self.allen_brain_tree.children(node.identifier)) == 0:  # 叶子节点
                parent_node = self.allen_brain_tree.parent(node.identifier)
                parent_node_id = parent_node.identifier if parent_node else None
                leaf_nodes_data.append([node.identifier, parent_node_id])
        
        return leaf_nodes_data
    
    def filter_and_normalize_matrix(self, matrix, percentile=75):
        """
        过滤和归一化矩阵
        
        Args:
            matrix: 原始矩阵
            percentile: 百分位数阈值
            
        Returns:
            DataFrame: 处理后的矩阵
        """
        # 提取数值部分（排除父节点列）
        numeric_matrix = matrix.iloc[:, 1:] if 'parent_node' in matrix.columns else matrix
        
        def keep_above_percentile(row, percentile):
            """保留大于指定分位数的值"""
            threshold = np.percentile(row, percentile)
            new_row = row.copy()
            new_row[row < threshold] = 0
            return new_row
        
        # 过滤低值
        result_filtered = numeric_matrix.apply(
            keep_above_percentile, axis=1, percentile=percentile
        )
        
        # 相同注射点取均值
        # result_filtered.insert(0, "parent_node", matrix["parent_node"])

        result_filtered_mean = result_filtered.groupby('injection_position').mean()
        
        return result_filtered_mean
    
    def integrate_paths_with_intensity(self, unique_pairs_df, intensity_matrix, min_path_length=5):
        """
        将路径数据与强度矩阵融合
        
        Args:
            unique_pairs_df: 代表性路径数据
            intensity_matrix: 强度矩阵
            min_path_length: 最小路径长度
            
        Returns:
            DataFrame: 融合后的数据
        """
        # 获取有效的注射位置
        unique_start_nodes = unique_pairs_df['replaced_start_node'].unique()
        
        def replace_node_id(node_id):
            if node_id == '':
                return 'Unknown'
            try:
                int_id = int(node_id)
                return self.stl_acro_dict.get(int_id, node_id)
            except ValueError:
                return node_id
        
        replaced_nodes = np.array([replace_node_id(node) for node in unique_start_nodes])
        valid_injection_positions = set(replaced_nodes) - {'Unknown'}
        
        # 过滤强度矩阵
        filtered_matrix = intensity_matrix.loc[
            intensity_matrix.index.isin(valid_injection_positions)
        ]
        
        # 按注射位置分组求均值
        filtered_matrix_mean = filtered_matrix.groupby(filtered_matrix.index).mean()
        
        result_matrix = []
        
        # 处理每个实验
        for exp_idx, exp_row in tqdm(filtered_matrix_mean.iterrows(), 
                                total=len(filtered_matrix_mean), 
                                desc="处理实验中"):
            
            injection_position = exp_idx[0] if isinstance(exp_idx, tuple) else exp_idx
            experiment_id = f"{exp_idx[0]}_{exp_idx[1]}" if isinstance(exp_idx, tuple) else str(exp_idx)
            
            # 匹配路径
            matched_paths = unique_pairs_df[unique_pairs_df['replaced_start_node'] == injection_position]
            
            if matched_paths.empty:
                continue
            
            # 处理每条匹配路径
            for _, path_row in matched_paths.iterrows():
                path_regions = path_row['replaced_path'].split('→')
                
                path_info = {
                    'experiment_id': experiment_id,
                    'injection_position': injection_position,
                    'path': path_row['replaced_path'],
                    'path_length': len(path_regions) - 1,
                    'region_intensities': [],
                    'intensity_sequence': []
                }
                
                valid_path = True
                
                # 添加注射位置自身强度
                injection_cols = [col for col in filtered_matrix_mean.columns 
                                if col[1] == injection_position]
                if injection_cols:
                    injection_intensity = max(exp_row[col] for col in injection_cols)
                    path_info['region_intensities'].append(f"{injection_position}:{injection_intensity:.6f}")
                    path_info['intensity_sequence'].append(injection_intensity)
                else:
                    valid_path = False
                
                # 计算路径上其他脑区的强度
                for region in path_regions[1:]:
                    region_cols = [col for col in filtered_matrix_mean.columns 
                                if col[1] == region]
                    
                    if region_cols:
                        intensity = max(exp_row[col] for col in region_cols)
                        path_info['region_intensities'].append(f"{region}:{intensity:.6f}")
                        path_info['intensity_sequence'].append(intensity)
                    else:
                        valid_path = False
                        break
                
                # 如果路径有效，计算总强度
                if valid_path and len(path_info['intensity_sequence']) == len(path_regions):
                    path_info['total_intensity'] = np.prod(path_info['intensity_sequence'])
                    path_info['region_intensities'] = " → ".join(path_info['region_intensities'])
                    result_matrix.append(path_info)
        
        if not result_matrix:
            return pd.DataFrame()
        
        # 创建DataFrame时明确指定数据类型
        results_df = pd.DataFrame(result_matrix, copy=True)
        results_df = results_df[['experiment_id', 'injection_position', 'path',
                                'path_length', 'region_intensities', 'total_intensity']]
        
        # 进一步过滤
        filtered_results = self._filter_final_results(results_df, min_path_length)
        
        return filtered_results
    
    def _filter_final_results(self, results_df, min_path_length):
        """过滤最终结果 - 修复版本"""
        # 按路径长度过滤，使用.copy()避免SettingWithCopyWarning
        filtered = results_df[results_df['path_length'] >= min_path_length].copy()
        
        # 使用.loc添加新列
        filtered.loc[:, 'strength'] = filtered['region_intensities'].apply(
            lambda x: [float(i.split(':')[1]) for i in x.split('→')]
        )
        
        # 移除包含零强度的路径，再次使用.copy()
        zero_mask = filtered['strength'].apply(lambda x: all(val != 0 for val in x))
        filtered = filtered[zero_mask].copy()
        
        # 清理列
        columns_to_drop = [col for col in ['experiment_id', 'region_intensities', 'total_intensity'] 
                        if col in filtered.columns]
        filtered = filtered.drop(columns_to_drop, axis=1)
        
        # 添加起始点和终止点
        filtered.loc[:, 'start'] = filtered['path'].apply(
            lambda x: x.split('→')[0] if '→' in x else x
        )
        filtered.loc[:, 'end'] = filtered['path'].apply(
            lambda x: x.split('→')[-1] if '→' in x else x
        )
        
        return filtered
    
    def download_Allen_files(
        self,
        csv_file_path,
        download_dir,
        id_column='id',
        max_retries=3,
        min_file_size=1024,
        base_url="http://api.brain-map.org/grid_data/download_file",
        image_type="injection_fraction", 
        resolution=25):
        """
        从Allen Brain Atlas API批量下载injection_fraction文件
        
        Args:
            csv_file_path (str): 包含实验ID的CSV文件路径
            download_dir (str): 文件下载保存目录
            id_column (str): CSV文件中包含实验ID的列名，默认为'id'
            max_retries (int): 最大重试次数，默认为3
            min_file_size (int): 最小文件大小阈值（字节），默认为1024
            base_url (str): API基础URL，默认为"http://api.brain-map.org/grid_data/download_file"
            image_type (str): 图像类型，默认为'injection_fraction',或选择'projection_density'
            resolution (int): 分辨率，默认为25
            
        Returns:
            tuple: (成功下载的ID列表, 失败下载的ID列表)
        """
        # 读取CSV文件
        try:
            df = pd.read_csv(csv_file_path)
            experiment_ids = df[id_column].tolist()
            print(f"成功读取CSV文件，找到 {len(experiment_ids)} 个实验ID")
        except Exception as e:
            print(f"读取CSV文件失败: {str(e)}")
            return [], []
        
        # 创建保存目录
        os.makedirs(download_dir, exist_ok=True)
        
        def download_single_file(exp_id, retries=max_retries):
            """下载单个文件"""
            url = f"{base_url}/{exp_id}?image={image_type}&resolution={resolution}"
            file_path = os.path.join(download_dir, f"{exp_id}_{image_type}.nrrd")
            
            for attempt in range(retries):
                try:
                    # 第一次请求获取文件大小
                    with requests.get(url, stream=True) as r:
                        r.raise_for_status()
                        total_size = int(r.headers.get('content-length', 0))
                        
                        # 检查文件是否已存在且大小匹配
                        if os.path.exists(file_path):
                            existing_size = os.path.getsize(file_path)
                            if total_size > 0 and existing_size == total_size:
                                print(f"✓ 文件已存在且完整: {exp_id}")
                                return True
                        
                        # 开始下载
                        downloaded_size = 0
                        with open(file_path, "wb") as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    downloaded_size += len(chunk)
                        
                        # 下载后验证
                        if total_size > 0 and downloaded_size != total_size:
                            raise Exception(f"文件大小不匹配: 期望 {total_size} 字节, 实际 {downloaded_size} 字节")
                        
                        # 检查文件是否大于最小阈值
                        if os.path.getsize(file_path) < min_file_size:
                            raise Exception(f"文件过小，可能下载不完整: 仅 {os.path.getsize(file_path)} 字节")
                        
                        print(f"✓ 下载成功: {exp_id} -> {file_path}")
                        return True
                        
                except Exception as e:
                    print(f"✗ 尝试 {attempt + 1}/{retries} 失败 {exp_id}: {str(e)}")
                    if os.path.exists(file_path):
                        os.remove(file_path)  # 删除不完整的文件
                    if attempt < retries - 1:
                        time.sleep(2)  # 等待2秒后重试
                    continue
            
            return False
        
        # 记录成功和失败的ID
        success_ids = []
        failed_ids = []
        
        # 批量下载
        for exp_id in tqdm(experiment_ids, desc="下载{image_type}文件"):
            if download_single_file(exp_id):
                success_ids.append(exp_id)
            else:
                failed_ids.append(exp_id)
        
        # 打印总结报告
        print("\n下载总结:")
        print(f"成功下载: {len(success_ids)} 个文件")
        print(f"失败下载: {len(failed_ids)} 个文件")
        if failed_ids:
            print("失败的ID:", failed_ids)
        
        return success_ids, failed_ids
    
    def preprocess_annotation_data(self):
        """
        安全优化的注解数据预处理，确保结果完全一致
        """
        print("开始预处理注解数据...")
        
        # 创建左右半脑标记（保持不变）
        z_mid = self.anno.shape[2] // 2
        annot_labeled = np.where(
            np.arange(self.anno.shape[2]) < z_mid,
            np.char.add(self.anno.astype(str), "_left"),
            np.char.add(self.anno.astype(str), "_right")
        )
        
        # 修复不对称区域（保持原有逻辑）
        unique_elements = np.unique(annot_labeled)
        
        print("修复不对称区域...")
        for area in tqdm(unique_elements, desc="修复不对称区域"):
            if area.endswith('_right') and area.replace('_right', '_left') not in unique_elements:
                x, y, z = np.where(annot_labeled == area)
                if len(z) > 0:
                    symmetric_z = self.anno.shape[2] - 1 - z
                    for i in range(len(x)):
                        if 0 <= symmetric_z[i] < self.anno.shape[2]:
                            annot_labeled[x[i], y[i], symmetric_z[i]] = area.replace('_right', '_left')
        
        # 更新唯一元素列表
        unique_elements = np.unique(annot_labeled)
        
        # 优化掩码计算部分，添加进度条
        print("预计算区域掩码...")
        area_masks = {}
        
        for area in tqdm(unique_elements, desc="计算区域掩码"):
            mask = (annot_labeled == area)
            if mask.sum() > 0:  # 只保存有体素的区域
                area_masks[area] = mask
        
        valid_areas = list(area_masks.keys())
        print(f"预处理完成，共有 {len(valid_areas)} 个有效区域")
        
        return annot_labeled, area_masks, valid_areas
    
    def process_experiment_fast(self, experiment_id, annot_labeled, area_masks, valid_areas, 
                            base_dir, output_dir=None, use_projection_density=True):
        """
        处理单个实验数据
        
        参数:
            experiment_id: 实验ID (如 100140756)
            annot_labeled: 3D标注数组
            area_masks: 预计算的脑区掩码字典
            valid_areas: 有效脑区列表
            base_dir: 数据基础目录
            output_dir: 结果保存路径（可选）
            use_projection_density: 是否使用投影密度，False则使用投影能量
            
        返回:
            dict: 包含实验结果的字典
        """
        # 1. 动态生成文件路径
        if use_projection_density:
            data_file = os.path.join(base_dir, f"projection_density/{experiment_id}_projection_density.nrrd")
            data_type = "projection_density"
        else:
            data_file = os.path.join(base_dir, f"projection_energy/{experiment_id}_projection_energy.nrrd")
            data_type = "projection_energy"
        
        injection_file = os.path.join(base_dir, f"injection_fraction/{experiment_id}_injection_fraction.nrrd")
        
        print(f"\n▶ 开始处理实验 {experiment_id}...")
        
        # 2. 加载数据（带错误处理）
        try:
            print("├─ 加载数据文件中...")
            data_array, _ = nrrd.read(data_file)
            inf, _ = nrrd.read(injection_file)
        except FileNotFoundError as e:
            error_msg = f"└─ 文件不存在: {str(e)}"
            print(error_msg)
            return {'experiment_id': experiment_id, 'error': 'file_not_found'}
        except Exception as e:
            error_msg = f"└─ 读取数据时出错: {str(e)}"
            print(error_msg)
            return {'experiment_id': experiment_id, 'error': 'read_error'}
        
        # 3. 计算注射中心
        print("├─ 计算注射中心...")
        injection_mask = inf >= 1
        if not injection_mask.any():
            print("└─ 未检测到有效注射区域")
            return {'experiment_id': experiment_id, 'injection_position': 'unknown'}
        
        weights = data_array[injection_mask]
        centroid = [np.average(coords, weights=weights) 
                for coords in np.where(injection_mask)]
        
        # 4. 获取注射位置
        try:
            inj_pos = annot_labeled[tuple(np.round(centroid).astype(int))]
            print(f"├─ 注射位置: {inj_pos}")
        except IndexError:
            inj_pos = 'unknown'
            print("├─ 注射位置: 未知（坐标越界）")
        
        # 5. 计算各脑区数据的均值
        print("├─ 计算脑区数据均值...")
        data_means = {}
        for area in tqdm(valid_areas, desc=f"处理{data_type}", leave=False):
            mask = area_masks.get(area)
            if mask is not None and mask.shape == data_array.shape:
                masked_data = data_array[mask]
                nonzero_data = masked_data[masked_data > 0]  # 忽略零值
                data_means[area] = float(nonzero_data.mean()) if len(nonzero_data) > 0 else 0.0
            else:
                data_means[area] = 0.0
        
        # 构建结果字典
        result = {
            'experiment_id': experiment_id,
            'injection_position': inj_pos,
            **data_means,
            '_meta': {
                'stat_method': f'mean_of_nonzero_{data_type}',
                'data_type': data_type
            }
        }
        
        # 6. 保存结果（如果指定了输出目录）
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{experiment_id}_results.csv")
            try:
                pd.DataFrame([result]).to_csv(output_path, index=False)
                print(f"└─ 结果已保存至: {output_path}")
                # 输出成功标识文件
                flag_path = os.path.join(output_dir, f"{experiment_id}.COMPLETED")
                open(flag_path, 'w').close()
                print(f"✓ 标识文件已生成: {flag_path}")
            except Exception as e:
                print(f"└─ 保存失败: {str(e)}")
        else:
            print("└─ 处理完成（未保存文件）")
        
        return result
    
    def batch_process_experiments_sequential(self, experiment_ids, annot_labeled, area_masks, valid_areas,
                                            base_dir, output_dir, use_projection_density=True):
        """
        批量处理实验数据（顺序版本，不使用并行）
        
        Args:
            experiment_ids: 实验ID列表
            annot_labeled: 标记后的注解数据
            area_masks: 区域掩码字典
            valid_areas: 有效区域列表
            base_dir: 数据基础目录
            output_dir: 输出目录
            use_projection_density: 是否使用投影密度
            
        Returns:
            DataFrame: 合并的结果
        """
        print("开始批量处理实验数据（顺序版本）...")
        print(f"共需处理 {len(experiment_ids)} 个实验")
        
        all_results = []
        successful_count = 0
        failed_count = 0
        skipped_count = 0
        
        for i, exp_id in enumerate(tqdm(experiment_ids, desc="处理实验")):
            # 检查是否已处理
            flag_path = os.path.join(output_dir, f"{exp_id}.COMPLETED")
            if os.path.exists(flag_path):
                print(f"\n[{i+1}/{len(experiment_ids)}] 跳过已处理实验 {exp_id}")
                skipped_count += 1
                continue
                
            print(f"\n[{i+1}/{len(experiment_ids)}] 处理实验 {exp_id}")
            
            result = self.process_experiment_fast(
                experiment_id=exp_id,
                annot_labeled=annot_labeled,
                area_masks=area_masks,
                valid_areas=valid_areas,
                base_dir=base_dir,
                output_dir=output_dir,
                use_projection_density=use_projection_density
            )
            
            # 统计成功和失败
            if 'error' in result or result.get('injection_position') == 'unknown':
                failed_count += 1
            else:
                successful_count += 1
                all_results.append(result)
            
            # 每处理10个实验显示一次进度
            if (i + 1) % 10 == 0:
                print(f"\n=== 进度: {i+1}/{len(experiment_ids)} ===")
                print(f"成功: {successful_count}, 失败: {failed_count}, 跳过: {skipped_count}")
        
        # 保存合并结果
        if all_results:
            combined_df = pd.DataFrame(all_results)
            combined_output = os.path.join(output_dir, "combined_results.csv")
            combined_df.to_csv(combined_output, index=False)
            print(f"\n批量处理完成!")
            print(f"成功处理: {successful_count} 个实验")
            print(f"失败: {failed_count} 个实验")
            print(f"跳过: {skipped_count} 个已处理实验")
            print(f"合并结果已保存至: {combined_output}")
            
            return combined_df
        else:
            print("\n批量处理完成，但没有成功处理的结果")
            return pd.DataFrame()
    
    def save_single_result(self, experiment_id, annot_labeled, area_masks, valid_areas, base_dir, output_dir):
        """
        处理单个实验并直接保存为CSV
        
        Args:
            experiment_id: 实验ID
            annot_labeled: 标记后的注解数据
            area_masks: 区域掩码字典
            valid_areas: 有效区域列表
            base_dir: 数据基础目录
            output_dir: 输出目录
            
        Returns:
            str: 保存的文件路径
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 处理实验数据
        result = self.process_experiment_fast(
            experiment_id=experiment_id,
            annot_labeled=annot_labeled,
            area_masks=area_masks,
            valid_areas=valid_areas,
            base_dir=base_dir,
            output_dir=output_dir
        )
        
        # 转换为DataFrame并保存
        df = pd.DataFrame([result])
        output_path = os.path.join(output_dir, f"{experiment_id}_results.csv")
        df.to_csv(output_path, index=False)
        
        return output_path
