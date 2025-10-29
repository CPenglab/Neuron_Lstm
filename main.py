from core.swc_processor import BatchSWCProcessor
from core.feature_integrator import SWCPathProcessor
from core.data_fusion import AllenDataFusion
from core.model import SequenceDataProcessor
import pandas as pd
import pyswcloader
import numpy as np


############# swc_processor
# Load annotation data
anno = pyswcloader.brain.read_nrrd('data/annotation_25.nrrd')
resolution = 25  # Adjust according to actual situation
allen_brain_tree = pyswcloader.brain.allen_brain_tree('data/1.json')
stl_acro_dict = pyswcloader.brain.acronym_dict('data/1.json')

# Create batch processor
batch_processor = BatchSWCProcessor(anno, resolution)

# Process all data
root_path = "data/orig_swc_data/test/unzip/"
results = batch_processor.process_batch_folders(root_path)
print(f"Processing completed, obtained {len(results)} path records")




############# feature_integrator
# Load data
directed_df = pd.read_csv('data/Mouse_brain_adjacency_matrix.csv', index_col=0)
file = pd.read_csv('data/orig_swc_data/test/unzip/all_regional_paths.csv')
# file1 = pd.read_csv('data/neuron_path_data/zip_fold/hippocampus.zip')
# file2 = pd.read_csv('data/neuron_path_data/zip_fold/prefrontal_cortex.zip')
# file3 = pd.read_csv('data/neuron_path_data/zip_fold/hypothalamus.zip')

combined_df = pd.concat([file], ignore_index=True)
# combined_df = pd.concat([file1, file2, file3], ignore_index=True)

# Create processor
processor_swc = SWCPathProcessor(allen_brain_tree, stl_acro_dict)

# Process path data
keys_set = processor_swc.filter_problematic_nodes(directed_df, stl_acro_dict)
combined_df = processor_swc.process_path_pipeline(combined_df, keys_set)
 
# Build representative paths
unique_pairs = processor_swc.build_representative_paths(
    combined_df, 
    save_progress_path='data/neuron_path_data/example/progress.csv'
)

# unique_pairs = processor.build_representative_paths(
#     combined_df, 
#     save_progress_path='data/neuron_path_data/zip_fold/progress.csv'
# )


# Add annotated paths
unique_pairs['replaced_path'] = unique_pairs['path'].apply(
    processor_swc.replace_nodes_with_acronyms
)

unique_pairs['replaced_start_node'] = unique_pairs['replaced_path'].str.split('→').str[0]

unique_pairs.to_csv('data/neuron_path_data/example/result.csv', index=False)




############# data_fusion
###### Download data
AllenData = AllenDataFusion(anno,allen_brain_tree, stl_acro_dict)

AllenData.download_Allen_files(  
    csv_file_path='data/experiment/url.csv',
    download_dir='data/experiment/example/injection_fraction',
    image_type="injection_fraction"
)

AllenData.download_Allen_files(  
    csv_file_path='data/experiment/url.csv',
    download_dir='data/experiment/example/projection_density',
    image_type="projection_density"
)

all_experiments = pd.read_csv('data/experiment/url.csv')

# Initialize fusion processor
fusion_processor = AllenDataFusion(anno, allen_brain_tree, stl_acro_dict)

# Preprocess annotation data (do only once), extremely time-consuming and memory-intensive
annot_labeled, area_masks, valid_areas = fusion_processor.preprocess_annotation_data()

# Experimental ID list
id_list = all_experiments['id'].tolist()

# Batch process experimental data (sequential version)
results_df = fusion_processor.batch_process_experiments_sequential(
    experiment_ids=id_list,
    annot_labeled=annot_labeled,
    area_masks=area_masks,
    valid_areas=valid_areas,
    base_dir="data/experiment/example",
    output_dir="data/experiment/example/result",
    use_projection_density=True  # Use projection density
)




###### Data integration
unique_pairs = pd.read_csv('data/neuron_path_data/zip_fold/result.csv')

# Load and preprocess Allen data
allen_data = fusion_processor.load_and_preprocess_allen_data(
    'data/experiment/merged_results.csv'
)

# Create ipsilateral and contralateral matrices
ipsi_matrix, contra_matrix = fusion_processor.create_ipsi_contra_matrices(allen_data)

# Filter nodes (need to pass keys_set)
ipsi_filtered = fusion_processor.filter_matrix_nodes(ipsi_matrix, keys_set)
contra_filtered = fusion_processor.filter_matrix_nodes(contra_matrix, keys_set)

# Create hierarchical matrix, this step corresponds to generating ipsi_matrix_result
ipsi_hierarchical = fusion_processor.create_hierarchical_matrix(ipsi_filtered)

# Filter and normalize
ipsi_processed = fusion_processor.filter_and_normalize_matrix(ipsi_hierarchical, percentile=75)




# Integrate with path data
final_results = fusion_processor.integrate_paths_with_intensity(
    unique_pairs,  # Representative paths from previous step
    ipsi_processed,
    min_path_length=5
)

# Save results
final_results.to_csv('data/model/final_results.csv', index=False)




############# model
# Initialize processor
processor_SequenceDataProcessor = SequenceDataProcessor(stl_acro_dict, 'data/gene/gene_filled_result.csv')

# Load and prepare data
X, y_log, max_len, pca = processor_SequenceDataProcessor.load_and_prepare_data('data/model/final_results.csv', window_size=5)

# Split data
node_train, node_test, strength_train, strength_test = processor_SequenceDataProcessor.split_data(X, y_log)

# Prepare final data
gene_train, gene_test, init_strength_train, init_strength_test, strength_train_shift, strength_test_shift = processor_SequenceDataProcessor.prepare_final_data(
    node_train, node_test, strength_train, strength_test, max_len
)

# Build model
model = processor_SequenceDataProcessor.build_true_autoregressive_model_with_k(max_len=5, gene_embed_dim=64)

# Train model (using built-in callback)
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


# Calculate gene importance
position_imp, dim_imp = processor_SequenceDataProcessor.compute_gene_importance(
    model=model,
    dataset=(gene_all, init_strength_all),
    target_timestep=-1,
    n_samples=20000
)

# Get original gene importance
gene_importance, gene_importance_df = processor_SequenceDataProcessor.get_gene_importance_from_pca(
    dimension_importance=dim_imp
)

gene_importance_df.to_csv('data/model/gene_importance.csv')