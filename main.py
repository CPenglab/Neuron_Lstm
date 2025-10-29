from core.swc_processor import BatchSWCProcessor
from core.feature_integrator import SWCPathProcessor
from core.data_fusion import AllenDataFusion
from core.model import SequenceDataProcessor
import pandas as pd
import pyswcloader
import numpy as np


#############swc_processor
# 加载注解数据
anno = pyswcloader.brain.read_nrrd('data/annotation_25.nrrd')
resolution = 25  # 根据实际情况调整
allen_brain_tree = pyswcloader.brain.allen_brain_tree('data/1.json')
stl_acro_dict = pyswcloader.brain.acronym_dict('data/1.json')

# 创建批量处理器
batch_processor = BatchSWCProcessor(anno, resolution)

# 处理所有数据
root_path = "data/orig_swc_data/test/unzip/"
results = batch_processor.process_batch_folders(root_path)
print(f"处理完成，共得到 {len(results)} 条路径记录")

#############feature_integrator

# 在你的主程序中这样使用


# 加载数据
directed_df=pd.read_csv('data/Mouse_brain_adjacency_matrix.csv',index_col=0)
file = pd.read_csv('data/orig_swc_data/test/unzip/all_regional_paths.csv')
# file1 = pd.read_csv('data/neuron_path_data/zip_fold/海马.zip')
# file2 = pd.read_csv('data/neuron_path_data/zip_fold/前额叶皮层.zip')
# file3 = pd.read_csv('data/neuron_path_data/zip_fold/下丘脑.zip')


combined_df = pd.concat([file], ignore_index=True)
# combined_df = pd.concat([file1,file2,file3], ignore_index=True)


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

# unique_pairs = processor.build_representative_paths(
#     combined_df, 
#     save_progress_path='data/neuron_path_data/zip_fold/progress.csv'
# )


# 添加替换后的路径
unique_pairs['replaced_path'] = unique_pairs['path'].apply(
    processor_swc.replace_nodes_with_acronyms
)

unique_pairs['replaced_start_node'] = unique_pairs['replaced_path'].str.split('→').str[0]

unique_pairs.to_csv('data/neuron_path_data/example/result.csv', index=False)


######下载数据
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

# 预处理注解数据（只做一次），及其耗时以及占用内存
annot_labeled, area_masks, valid_areas = fusion_processor.preprocess_annotation_data()

# 实验ID列表
id_list = all_experiments['id'].tolist()

# 批量处理实验数据（顺序版本）
results_df = fusion_processor.batch_process_experiments_sequential(
    experiment_ids=id_list,
    annot_labeled=annot_labeled,
    area_masks=area_masks,
    valid_areas=valid_areas,
    base_dir="data/experiment/example",
    output_dir="data/experiment/example/result",
    use_projection_density=True  # 使用投影密度
)




######数据整合
unique_pairs = pd.read_csv('data/neuron_path_data/zip_fold/result.csv')


# 1. 加载和预处理Allen数据
allen_data = fusion_processor.load_and_preprocess_allen_data(
    'data/experiment/merged_results.csv'
)

# 2. 创建同侧和对侧矩阵
ipsi_matrix, contra_matrix = fusion_processor.create_ipsi_contra_matrices(allen_data)

# 3. 过滤节点（需要传入keys_set）
ipsi_filtered = fusion_processor.filter_matrix_nodes(ipsi_matrix, keys_set)
contra_filtered = fusion_processor.filter_matrix_nodes(contra_matrix, keys_set)

# 4. 创建层次化矩阵,这一步对应生成ipsi_matrix_result
ipsi_hierarchical = fusion_processor.create_hierarchical_matrix(ipsi_filtered)

# 5. 过滤和归一化
ipsi_processed = fusion_processor.filter_and_normalize_matrix(ipsi_hierarchical, percentile=75)




# 6. 与路径数据融合
final_results = fusion_processor.integrate_paths_with_intensity(
    unique_pairs,  # 来自前一步的代表性路径
    ipsi_processed,
    min_path_length=5
)

# 保存结果
final_results.to_csv('data/model/final_results.csv', index=False)

# final_results = pd.read_csv('data/model/final_results.csv')




###model
# 使用示例
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


# 计算基因重要性
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

gene_importance_df.to_csv('data/model/gene_importance.csv')
















