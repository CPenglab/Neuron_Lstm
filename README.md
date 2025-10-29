# 🧠 Neuron Connectivity Analyzer

## 📋 Four-Stage Workflow

### 1. 🎯 Raw SWC Data Processing
Extract directed connection paths from single neuron SWC morphological data

**Features**:
- SWC format parsing and preprocessing
- Brain region annotation mapping
- Directed graph network construction
- Topological sorting optimization

```python
# Load brain region annotation data
anno = pyswcloader.brain.read_nrrd('data/annotation_25.nrrd')
resolution = 25  # Adjust according to actual situation
allen_brain_tree = pyswcloader.brain.allen_brain_tree('data/1.json')
stl_acro_dict = pyswcloader.brain.acronym_dict('data/1.json')

# Create SWC batch processor
batch_processor = BatchSWCProcessor(anno, resolution)

# Process single neuron SWC data
# This path contains some example raw SWC data
# Complete dataset for the paper has been organized and placed in data\neuron_path_data\zip_fold directory
root_path = "data/orig_swc_data/test/unzip/"

# Execute batch processing
results = batch_processor.process_batch_folders(root_path)
```



### 2. 🔍 Representative Feature Path Integration
Identify key feature paths from large numbers of connection paths

**Features**：
- Path frequency statistical analysis
- Greedy algorithm feature path identification

```python
# Load data
# directed_df is the mouse brain physical adjacency matrix
# all_regional_paths.csv contains single neuron path information organized from example dataset
directed_df=pd.read_csv('data/Mouse_brain_adjacency_matrix.csv',index_col=0)
file = pd.read_csv('data/orig_swc_data/test/unzip/all_regional_paths.csv')
combined_df = pd.concat([file], ignore_index=True)


# Create processor
processor_swc = SWCPathProcessor(allen_brain_tree, stl_acro_dict)
# Node preprocessing
keys_set = processor_swc.filter_problematic_nodes(directed_df, stl_acro_dict)
# Path preprocessing
combined_df = processor_swc.process_path_pipeline(combined_df,keys_set)

# Build representative paths
unique_pairs = processor_swc.build_representative_paths(
    combined_df, 
    save_progress_path='data/neuron_path_data/example/progress.csv'
)

# Annotate representative paths
unique_pairs['replaced_path'] = unique_pairs['path'].apply(
    processor_swc.replace_nodes_with_acronyms
)
unique_pairs['replaced_start_node'] = unique_pairs['replaced_path'].str.split('→').str[0]

unique_pairs.to_csv('data/neuron_path_data/example/result.csv', index=False)
```



### 3. 📊 Experimental Data Download, Processing and Training Set Construction
Download and process raw data from Allen Institute, and integrate feature paths with experimental projection intensity data

**Features**：
- Allen data download and processing
- Feature path and intensity information integration


```python
# Download data, here only about 10 examples are downloaded as demonstration
# If you want to download more data, you can use the complete experimental data url.csv we provided
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
# Initialize fusion processor
fusion_processor = AllenDataFusion(anno, allen_brain_tree, stl_acro_dict)
# Preprocess annotation data, extremely time-consuming and memory-intensive
annot_labeled, area_masks, valid_areas = fusion_processor.preprocess_annotation_data()
# Experimental ID list
id_list = all_experiments['id'].tolist()
# Batch process experimental data
results_df = fusion_processor.batch_process_experiments_sequential(
    experiment_ids=id_list,
    annot_labeled=annot_labeled,
    area_masks=area_masks,
    valid_areas=valid_areas,
    base_dir="data/experiment/example",
    output_dir="data/experiment/example/result",
    use_projection_density=True  # 使用投影密度
)



# Here we provide 7319544 single neuron paths statistically organized feature paths for use
unique_pairs = pd.read_csv('data/neuron_path_data/zip_fold/result.csv')
# Load and preprocess Allen data, here we provide 2992 complete processed examples
allen_data = fusion_processor.load_and_preprocess_allen_data(
    'data/experiment/merged_results.csv'
)

# Create ipsilateral and contralateral matrices
ipsi_matrix, contra_matrix = fusion_processor.create_ipsi_contra_matrices(allen_data)

# Filter nodes
ipsi_filtered = fusion_processor.filter_matrix_nodes(ipsi_matrix, keys_set)
contra_filtered = fusion_processor.filter_matrix_nodes(contra_matrix, keys_set)

# Create hierarchical matrix
ipsi_hierarchical = fusion_processor.create_hierarchical_matrix(ipsi_filtered)

#　Filter and normalize
ipsi_processed = fusion_processor.filter_and_normalize_matrix(ipsi_hierarchical, percentile=75)

#　Integrate feature paths with intensity information
final_results = fusion_processor.integrate_paths_with_intensity(
    unique_pairs,  # 来自读取的代表性路径
    ipsi_processed,
    min_path_length=5
)

# Gene expression data is placed in data/gene directory by organizing spatial transcriptomics data from digital brain
# Rows are regions, columns are gene names. If you have better datasets, you can edit them in this format
```






### 4. 🧠 LSTM Model Training and Prediction
Build sequence models to predict brain region connection strength

**Features**：
- Training dataset construction
- LSTM modeling
- Gradient importance analysis
- Connection strength prediction

```python
# Initialize processor
processor_SequenceDataProcessor = SequenceDataProcessor(stl_acro_dict, 'data/gene/gene_filled_result.csv')

# Load and prepare data input, here we have completed removal of completely duplicate data, mainly to extract sequence information and corresponding intensity information
X, y_log, max_len, pca = processor_SequenceDataProcessor.load_and_prepare_data('data/model/final_results.csv', window_size=5)

# Split data
node_train, node_test, strength_train, strength_test = processor_SequenceDataProcessor.split_data(X, y_log)

# Prepare model training data, here we add gene embeddings to sequence information
gene_train, gene_test, init_strength_train, init_strength_test, strength_train_shift, strength_test_shift = processor_SequenceDataProcessor.prepare_final_data(
    node_train, node_test, strength_train, strength_test, max_len
)

# Build LSTM model
model = processor_SequenceDataProcessor.build_true_autoregressive_model_with_k(max_len=5,gene_embed_dim=64)

# Train model, here we use callback functions to verify model performance in test set
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


# Subsequent gradient importance analysis
gene_all = np.concatenate([gene_train, gene_test], axis=0)
init_strength_all = np.concatenate([init_strength_train, init_strength_test], axis=0)
strength_shift_all = np.concatenate([strength_train_shift, strength_test_shift], axis=0)


# Calculate gene and position importance
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

```
