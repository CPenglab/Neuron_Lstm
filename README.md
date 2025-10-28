# 🧠 Neuron Connectivity Analyzer

## 📋 四阶段工作流程

### 1. 🎯 原始SWC数据处理
从单神经元SWC形态学数据中提取有向连接路径

**功能特点**：
- SWC格式解析与预处理
- 脑区标注映射
- 有向图网络构建
- 拓扑排序优化

```python
# 加载脑区注解数据
anno = pyswcloader.brain.read_nrrd('data/annotation_25.nrrd')
resolution = 25  # 图像分辨率，单位：微米

# 创建SWC批量处理器
batch_processor = BatchSWCProcessor(anno, resolution)

# 处理单神经元SWC数据
# 该路径下存有部分示例数据
# 完整数据集已整理并放置在 data\neuron_path_data\zip_fold 目录下
root_path = "data/orig_swc_data/test/unzip/"

# 执行批量处理
results = batch_processor.process_batch_folders(root_path)
```



### 2. 🔍 代表性特征路径整合
从大量连接路径中识别关键特征路径

**功能特点**：
- 路径频率统计分析
- 贪心算法特征路径识别
- 连续区域路径压缩
- 路径模式可视化



```python
加载数据
directed_df=pd.read_csv('data/小鼠邻接矩阵_filted_anno.csv',index_col=0)
file = pd.read_csv('data/orig_swc_data/test/unzip/all_regional_paths.csv')

combined_df = pd.concat([file], ignore_index=True)
加载脑区数据
allen_brain_tree = pyswcloader.brain.allen_brain_tree('data/1.json')
stl_acro_dict = pyswcloader.brain.acronym_dict('data/1.json')

创建处理器
processor = SWCPathProcessor(allen_brain_tree, stl_acro_dict)
处理路径数据
keys_set = processor.filter_problematic_nodes(directed_df, stl_acro_dict)
combined_df['processed_compressed_path'] = combined_df['compressed_path'].apply(
    lambda x: processor.process_compressed_path(x, keys_set)
)
combined_df["merged_compressed_path"] = combined_df["processed_compressed_path"].apply(
    processor.merge_consecutive_nodes
)
combined_df['clean_path'] = combined_df['merged_compressed_path'].apply(
    processor.remove_weights
)
combined_df['replace_path'] = combined_df['clean_path'].apply(
    processor.replace_nodes_with_acronyms
)
combined_df = processor.split_path_to_columns(combined_df)

构建代表性路径
unique_pairs = processor.build_representative_paths(
    combined_df, 
    save_progress_path='data/neuron_path_data/example/progress.csv'
)

添加替换后的路径
unique_pairs['replaced_path'] = unique_pairs['path'].apply(
    processor.replace_nodes_with_acronyms
)

unique_pairs['replaced_start_node'] = unique_pairs['start_node'].apply(
    processor.replace_nodes_with_acronyms
)

unique_pairs.to_csv('data/neuron_path_data/example/neuron_path_dataunique_pairs_with_paths_partial.csv', index=False)
```



### 3. 📊 实验数据融合与训练集构建
将特征路径与实验投射强度数据结合

**功能特点**：
- 基因表达数据整合
- 投射强度标签对齐
- 数据标准化处理
- 训练/验证集分割



```python
下载数据
AllenData = AllenDataFusion(allen_brain_tree, stl_acro_dict)

AllenData.download_Allen_files(  
    csv_file_path='data/experiment/url.csv',
    download_dir='data/experiment/injection_fraction',
    image_type="injection_fraction"
)

AllenData.download_Allen_files(  
    csv_file_path='data/experiment/url.csv',
    download_dir='data/experiment/projection_density',
    image_type="projection_density"
)


# 初始化融合器
fusion_processor = AllenDataFusion(anno, allen_brain_tree, stl_acro_dict)

批量处理实验数据（顺序版本）
results_df = fusion_processor.batch_process_experiments_sequential(
    experiment_ids=id_list,
    annot_labeled=annot_labeled,
    area_masks=area_masks,
    valid_areas=valid_areas,
    base_dir="data/experiment/example",
    output_dir="data/experiment/example/result",
    use_projection_density=True  # 使用投影密度
)




加载和预处理Allen数据
allen_data = fusion_processor.load_and_preprocess_allen_data(
    'data/experiment/merged_results.csv'
)

与路径数据融合
final_results = fusion_processor.integrate_paths_with_intensity(
    unique_pairs,  # 来自前一步的代表性路径
    ipsi_processed,
    min_path_length=5
)

```






### 4. 🧠 LSTM模型训练与预测
构建序列模型预测脑区连接强度

**功能特点**：
- LSTM/RNN序列建模
- 五折交叉验证
- 梯度重要性分析
- 连接强度预测


