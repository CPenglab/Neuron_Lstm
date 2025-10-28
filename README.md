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
加载注解数据
anno = pyswcloader.brain.read_nrrd('data/annotation_25.nrrd')
resolution = 25  

创建批量处理器
batch_processor = BatchSWCProcessor(anno, resolution)

处理所有数据
root_path = "data/orig_swc_data/test/unzip/"
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

### 4. 🧠 LSTM模型训练与预测
构建序列模型预测脑区连接强度

**功能特点**：
- LSTM/RNN序列建模
- 五折交叉验证
- 梯度重要性分析
- 连接强度预测


