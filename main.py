from core.swc_processor import BatchSWCProcessor
from core.feature_integrator import SWCPathProcessor
import pandas as pd
import pyswcloader

#############swc_processor
# 加载注解数据
anno = pyswcloader.brain.read_nrrd('data/annotation_25.nrrd')
resolution = 25  # 根据实际情况调整

# 创建批量处理器
batch_processor = BatchSWCProcessor(anno, resolution)

# 处理所有数据
root_path = "data/orig_swc_data/test/unzip/"
results = batch_processor.process_batch_folders(root_path)
print(f"处理完成，共得到 {len(results)} 条路径记录")

#############feature_integrator

# 在你的主程序中这样使用

# 加载数据
# file1 = pd.read_csv('data/neuron_path_data/海马.csv')
# file2 = pd.read_csv('data/neuron_path_data/前额叶皮层.csv')
# file3 = pd.read_csv('data/neuron_path_data/下丘脑.csv')
# combined_df = pd.concat([file1, file2, file3], ignore_index=True)
directed_df=pd.read_csv('data/小鼠邻接矩阵_filted_anno.csv',index_col=0)
file = pd.read_csv('data/orig_swc_data/test/unzip/all_regional_paths.csv')
combined_df = pd.concat([file], ignore_index=True)


# 加载脑区数据
allen_brain_tree = pyswcloader.brain.allen_brain_tree('data/1.json')
stl_acro_dict = pyswcloader.brain.acronym_dict('data/1.json')
# keys_for_values = {k: v for k, v in stl_acro_dict.items()}

# 创建处理器
processor = SWCPathProcessor(allen_brain_tree, stl_acro_dict)


# 处理路径数据
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

combined_df['repalce_path'] = combined_df['clean_path'].apply(
    processor.replace_nodes_with_acronyms
)

# 构建代表性路径
unique_pairs = processor.build_representative_paths(
    combined_df, 
    save_progress_path='data/neuron_path_data/progress.csv'
)

# 添加替换后的路径
unique_pairs['replaced_path'] = unique_pairs['path'].apply(
    processor.replace_nodes_with_acronyms
)

unique_pairs['replaced_start_node'] = unique_pairs['start_node'].apply(
    processor.replace_nodes_with_acronyms
)

unique_pairs.to_csv('data/neuron_path_data/neuron_path_dataunique_pairs_with_paths_partial_test.csv', index=False)





