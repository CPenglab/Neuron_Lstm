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
resolution = 25  # 根据实际情况调整
allen_brain_tree = pyswcloader.brain.allen_brain_tree('data/1.json')
stl_acro_dict = pyswcloader.brain.acronym_dict('data/1.json')

# 创建SWC批量处理器
batch_processor = BatchSWCProcessor(anno, resolution)

# 处理单神经元SWC数据
# 该路径下存有部分原始swc示例数据
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

```python
#加载数据,下面展示的为示例数据集
#directed_df为小鼠脑物理邻接矩阵
#all_regional_paths.csv为上一步单神经元整理出的路径信息
directed_df=pd.read_csv('data/Mouse_brain_adjacency_matrix.csv',index_col=0)
file = pd.read_csv('data/orig_swc_data/test/unzip/all_regional_paths.csv')

combined_df = pd.concat([file], ignore_index=True)


# 创建处理器
processor_swc = SWCPathProcessor(allen_brain_tree, stl_acro_dict)
# 处理路径数据
keys_set = processor_swc.filter_problematic_nodes(directed_df, stl_acro_dict)

combined_df = processor_swc.process_path_pipeline(combined_df,keys_set)

# 构建代表性路径
unique_pairs = processor_swc.build_representative_paths(
    combined_df, 
    save_progress_path='data/neuron_path_data/example/progress.csv'
)

#对代表性路径进行注释
unique_pairs['replaced_path'] = unique_pairs['path'].apply(
    processor_swc.replace_nodes_with_acronyms
)

unique_pairs['replaced_start_node'] = unique_pairs['replaced_path'].str.split('→').str[0]

unique_pairs.to_csv('data/neuron_path_data/example/result.csv', index=False)


```



### 3. 📊 实验数据融合与训练集构建
将特征路径与实验投射强度数据结合

**功能特点**：
- Allen数据下载
- 投射强度标签对齐
- 基因表达数据整合
- 数据标准化处理
- 训练/验证集分割



```python
#下载数据,这里只下载了10例数据左右作为示范，如果您想要下载更多数据，可以通过我们提供的整体实验数据url.csv
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



all_experiments = pd.read_csv('data/experiment/url.csv')
# 初始化融合器
fusion_processor = AllenDataFusion(anno, allen_brain_tree, stl_acro_dict)

# 预处理注解数据（只做一次），极其耗时以及占用内存
annot_labeled, area_masks, valid_areas = fusion_processor.preprocess_annotation_data()

# 实验ID列表
id_list = all_experiments['id'].tolist()

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



######数据整合，在这里我们提供了7319544单神经元路径统计完全的特征路径
unique_pairs = pd.read_csv('data/neuron_path_data/zip_fold/result.csv')

#加载和预处理Allen数据，这里我们提供了2992完整的处理后数据
allen_data = fusion_processor.load_and_preprocess_allen_data(
    'data/experiment/merged_results.csv'
)

创建同侧和对侧矩阵
ipsi_matrix, contra_matrix = fusion_processor.create_ipsi_contra_matrices(allen_data)

过滤节点（需要传入keys_set）
ipsi_filtered = fusion_processor.filter_matrix_nodes(ipsi_matrix, keys_set)
contra_filtered = fusion_processor.filter_matrix_nodes(contra_matrix, keys_set)

创建层次化矩阵,这一步对应生成ipsi_matrix_result
ipsi_hierarchical = fusion_processor.create_hierarchical_matrix(ipsi_filtered)

过滤和归一化
ipsi_processed = fusion_processor.filter_and_normalize_matrix(ipsi_hierarchical, percentile=75)

#将实验数据与神经元路径信息数据融合
final_results = fusion_processor.integrate_paths_with_intensity(
    unique_pairs,  # 来自前一步的代表性路径
    ipsi_processed,
    min_path_length=5
)

#基因表达数据通过整理digtal brain的空转数据放在了data/gene目录下
#横坐标是区域，纵坐标是基因名，如果您有更优质数据集可按照这个格式进行编辑

```






### 4. 🧠 LSTM模型训练与预测
构建序列模型预测脑区连接强度

**功能特点**：
- 训练数据集构建
- LSTM序列建模
- 梯度重要性分析
- 连接强度预测

```python
# 初始化处理器
processor_SequenceDataProcessor = SequenceDataProcessor(stl_acro_dict, 'data/gene/gene_filled_result.csv')

# 加载和准备数据
X, y_log, max_len, pca = processor_SequenceDataProcessor.load_and_prepare_data('data/model/final_results.csv', window_size=3)

# 分割数据
node_train, node_test, strength_train, strength_test = processor_SequenceDataProcessor.split_data(X, y_log)

# 准备最终数据
gene_train, gene_test, init_strength_train, init_strength_test, strength_train_shift, strength_test_shift = processor_SequenceDataProcessor.prepare_final_data(
    node_train, node_test, strength_train, strength_test, max_len
)

# 构建模型
model = processor_SequenceDataProcessor.build_true_autoregressive_model_with_k(max_len=3,gene_embed_dim=64)

# 训练模型（使用内置回调）
r2_callback = processor_SequenceDataProcessor.MultiInputR2ScoreCallback(
    validation_data=([gene_test, init_strength_test], strength_test_shift)
)

history = model.fit(
    [gene_train, init_strength_train],
    strength_train_shift,
    validation_data=([gene_test, init_strength_test], strength_test_shift),
    epochs=50,
    batch_size=32,
    callbacks=[r2_callback]
)

gene_all = np.concatenate([gene_train, gene_test], axis=0)
init_strength_all = np.concatenate([init_strength_train, init_strength_test], axis=0)
strength_shift_all = np.concatenate([strength_train_shift, strength_test_shift], axis=0)


# 计算基因以及位置重要性
position_imp, dim_imp = processor_SequenceDataProcessor.compute_gene_importance(
    model=model,
    dataset=(gene_all, init_strength_all),
    target_timestep=-1,
    n_samples=20000
)

# 获取原始基因重要性
gene_importance, gene_importance_df = processor_SequenceDataProcessor.get_gene_importance_from_pca(
    dimension_importance=dim_imp
)

```



