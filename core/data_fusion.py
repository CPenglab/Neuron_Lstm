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
    """Allen Experimental Data Fusion - Processes Allen experimental data and path data fusion"""
    
    def __init__(self, anno, allen_brain_tree, stl_acro_dict):
        """
        Initialize fusion processor
        
        Args:
            allen_brain_tree: Brain region tree structure
            stl_acro_dict: Abbreviation dictionary
        """
        self.allen_brain_tree = allen_brain_tree
        self.stl_acro_dict = stl_acro_dict
        self.anno = anno
    
    def load_and_preprocess_allen_data(self, file_path):
        """
        Load and preprocess Allen experimental data
        
        Args:
            file_path: Allen data file path
            
        Returns:
            DataFrame: Preprocessed data
        """
        file = pd.read_csv(file_path)
        
        # Select numeric columns, exclude non-numeric columns
        numeric_cols = file.columns.difference(["experiment_id", "injection_position", "_meta"])
        
        # Group by experiment ID and injection position to calculate mean
        merged_df = file.groupby(["experiment_id", "injection_position"])[numeric_cols].mean().reset_index()
        merged_df = merged_df[merged_df['injection_position'] != 'unknown']
        
        return merged_df
    
    def create_ipsi_contra_matrices(self, merged_df):
        """
        Create ipsilateral and contralateral projection matrices
        
        Args:
            merged_df: Preprocessed Allen data
            
        Returns:
            tuple: (ipsilateral matrix, contralateral matrix)
        """
        # Separate left and right injection positions
        left_positions = merged_df[merged_df['injection_position'].str.endswith('_left')]
        right_positions = merged_df[merged_df['injection_position'].str.endswith('_right')]
        
        left_matrix = left_positions.set_index('injection_position')
        right_matrix = right_positions.set_index('injection_position')
        
        def split_columns_by_suffix(matrix, suffix):
            """Filter columns ending with specified suffix"""
            return matrix.loc[:, matrix.columns.str.endswith(suffix)]
        
        # Separate columns by side
        left_matrix_left = split_columns_by_suffix(left_matrix, '_left')
        left_matrix_right = split_columns_by_suffix(left_matrix, '_right')
        right_matrix_left = split_columns_by_suffix(right_matrix, '_left')
        right_matrix_right = split_columns_by_suffix(right_matrix, '_right')
        
        # Clean column names and indices
        right_matrix_right.columns = right_matrix_right.columns.str.replace('_right', '')
        right_matrix_right.index = right_matrix_right.index.str.replace('_right', '')
        left_matrix_left.columns = left_matrix_left.columns.str.replace('_left', '')
        left_matrix_left.index = left_matrix_left.index.str.replace('_left', '')
        right_matrix_left.columns = right_matrix_left.columns.str.replace('_left', '')
        right_matrix_left.index = right_matrix_left.index.str.replace('_right', '')
        left_matrix_right.columns = left_matrix_right.columns.str.replace('_right', '')
        left_matrix_right.index = left_matrix_right.index.str.replace('_left', '')
        
        # Merge ipsilateral and contralateral matrices
        ipsi_matrix = pd.concat([left_matrix_left, right_matrix_right])
        contra_matrix = pd.concat([right_matrix_left, left_matrix_right])
        
        return ipsi_matrix, contra_matrix
    
    def filter_matrix_nodes(self, matrix, keys_to_remove):
        """
        Filter specified nodes from matrix
        
        Args:
            matrix: Matrix to filter
            keys_to_remove: List of nodes to remove
            
        Returns:
            DataFrame: Filtered matrix
        """
        keys_str = [str(k) for k in keys_to_remove]
        filtered_matrix = matrix.drop(index=keys_str, errors='ignore') \
                              .drop(columns=keys_str, errors='ignore')
        
        # Replace row and column names with abbreviation names
        filtered_matrix.index = filtered_matrix.index.map(
            lambda x: self.stl_acro_dict.get(int(x), x) if str(x).isdigit() else x
        )
        filtered_matrix.columns = filtered_matrix.columns.map(
            lambda x: self.stl_acro_dict.get(int(x), x) if str(x).isdigit() else x
        )
        
        return filtered_matrix
    
    def create_hierarchical_matrix(self, matrix):
        """
        Create hierarchical matrix (grouped by parent node)
        
        Args:
            matrix: Original matrix
            
        Returns:
            DataFrame: Hierarchical matrix
        """
        # Get leaf nodes and their parent nodes
        leaf_nodes_data = self._get_leaf_nodes_with_parents()
        node_parent_df = pd.DataFrame(leaf_nodes_data, columns=['node', 'parentnode'])
        
        # Replace with abbreviation names
        node_parent_df['node'] = node_parent_df['node'].replace(self.stl_acro_dict)
        node_parent_df['parentnode'] = node_parent_df['parentnode'].replace(self.stl_acro_dict)
        
        # Create column hierarchy
        column_to_parent = dict(zip(node_parent_df['node'], node_parent_df['parentnode']))
        column_names = matrix.columns.tolist()
        parent_names = [column_to_parent.get(col, "root") for col in column_names]
        
        multi_columns = pd.MultiIndex.from_arrays(
            [parent_names, column_names],
            names=["Parent", "Region"]
        )
        
        matrix_with_parents = matrix.copy()
        matrix_with_parents.columns = multi_columns
        
        # Sort
        matrix_sorted = matrix_with_parents.sort_index(
            axis=1, level="Parent", sort_remaining=True
        )
        
        # Add row parent nodes and sort
        row_to_parent = dict(zip(node_parent_df['node'], node_parent_df['parentnode']))
        row_parents = [row_to_parent.get(row, "root") for row in matrix_sorted.index]
        matrix_sorted.insert(0, "parent_node", row_parents)
        
        matrix_final = matrix_sorted.sort_values(by="parent_node", ascending=True, kind='mergesort')
        
        return matrix_final
    
    def _get_leaf_nodes_with_parents(self):
        """Get leaf nodes and their parent node information"""
        leaf_nodes_data = []
        
        for node in self.allen_brain_tree.all_nodes():
            if len(self.allen_brain_tree.children(node.identifier)) == 0:  # Leaf node
                parent_node = self.allen_brain_tree.parent(node.identifier)
                parent_node_id = parent_node.identifier if parent_node else None
                leaf_nodes_data.append([node.identifier, parent_node_id])
        
        return leaf_nodes_data
    
    def filter_and_normalize_matrix(self, matrix, percentile=75):
        """
        Filter and normalize matrix
        
        Args:
            matrix: Original matrix
            percentile: Percentile threshold
            
        Returns:
            DataFrame: Processed matrix
        """
        # Extract numeric part (exclude parent node column)
        numeric_matrix = matrix.iloc[:, 1:] if 'parent_node' in matrix.columns else matrix
        
        def keep_above_percentile(row, percentile):
            """Keep values above specified percentile"""
            threshold = np.percentile(row, percentile)
            new_row = row.copy()
            new_row[row < threshold] = 0
            return new_row
        
        # Filter low values
        result_filtered = numeric_matrix.apply(
            keep_above_percentile, axis=1, percentile=percentile
        )
        
        # Take mean for same injection points
        # result_filtered.insert(0, "parent_node", matrix["parent_node"])

        result_filtered_mean = result_filtered.groupby('injection_position').mean()
        
        return result_filtered_mean
    
    def integrate_paths_with_intensity(self, unique_pairs_df, intensity_matrix, min_path_length=5):
        """
        Integrate path data with intensity matrix
        
        Args:
            unique_pairs_df: Representative path data
            intensity_matrix: Intensity matrix
            min_path_length: Minimum path length
            
        Returns:
            DataFrame: Integrated data
        """
        # Get valid injection positions
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
        
        # Filter intensity matrix
        filtered_matrix = intensity_matrix.loc[
            intensity_matrix.index.isin(valid_injection_positions)
        ]
        
        # Calculate mean by injection position
        filtered_matrix_mean = filtered_matrix.groupby(filtered_matrix.index).mean()
        
        result_matrix = []
        
        # Process each experiment
        for exp_idx, exp_row in tqdm(filtered_matrix_mean.iterrows(), 
                                total=len(filtered_matrix_mean), 
                                desc="Processing experiments"):
            
            injection_position = exp_idx[0] if isinstance(exp_idx, tuple) else exp_idx
            experiment_id = f"{exp_idx[0]}_{exp_idx[1]}" if isinstance(exp_idx, tuple) else str(exp_idx)
            
            # Match paths
            matched_paths = unique_pairs_df[unique_pairs_df['replaced_start_node'] == injection_position]
            
            if matched_paths.empty:
                continue
            
            # Process each matched path
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
                
                # Add injection position's own intensity
                injection_cols = [col for col in filtered_matrix_mean.columns 
                                if col[1] == injection_position]
                if injection_cols:
                    injection_intensity = max(exp_row[col] for col in injection_cols)
                    path_info['region_intensities'].append(f"{injection_position}:{injection_intensity:.6f}")
                    path_info['intensity_sequence'].append(injection_intensity)
                else:
                    valid_path = False
                
                # Calculate intensity for other brain regions on the path
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
                
                # If path is valid, calculate total intensity
                if valid_path and len(path_info['intensity_sequence']) == len(path_regions):
                    path_info['total_intensity'] = np.prod(path_info['intensity_sequence'])
                    path_info['region_intensities'] = " → ".join(path_info['region_intensities'])
                    result_matrix.append(path_info)
        
        if not result_matrix:
            return pd.DataFrame()
        
        # Create DataFrame with explicit data types
        results_df = pd.DataFrame(result_matrix, copy=True)
        results_df = results_df[['experiment_id', 'injection_position', 'path',
                                'path_length', 'region_intensities', 'total_intensity']]
        
        # Further filtering
        filtered_results = self._filter_final_results(results_df, min_path_length)
        
        return filtered_results
    
    def _filter_final_results(self, results_df, min_path_length):
        """Filter final results - fixed version"""
        # Filter by path length, use .copy() to avoid SettingWithCopyWarning
        filtered = results_df[results_df['path_length'] >= min_path_length].copy()
        
        # Use .loc to add new column
        filtered.loc[:, 'strength'] = filtered['region_intensities'].apply(
            lambda x: [float(i.split(':')[1]) for i in x.split('→')]
        )
        
        # Remove paths containing zero intensity, use .copy() again
        zero_mask = filtered['strength'].apply(lambda x: all(val != 0 for val in x))
        filtered = filtered[zero_mask].copy()
        
        # Clean columns
        columns_to_drop = [col for col in ['experiment_id', 'region_intensities', 'total_intensity'] 
                        if col in filtered.columns]
        filtered = filtered.drop(columns_to_drop, axis=1)
        
        # Add start and end points
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
        Batch download injection_fraction files from Allen Brain Atlas API
        
        Args:
            csv_file_path (str): CSV file path containing experiment IDs
            download_dir (str): Directory to save downloaded files
            id_column (str): Column name in CSV containing experiment IDs, default 'id'
            max_retries (int): Maximum retry attempts, default 3
            min_file_size (int): Minimum file size threshold (bytes), default 1024
            base_url (str): API base URL, default "http://api.brain-map.org/grid_data/download_file"
            image_type (str): Image type, default 'injection_fraction', or choose 'projection_density'
            resolution (int): Resolution, default 25
            
        Returns:
            tuple: (List of successfully downloaded IDs, List of failed download IDs)
        """
        # Read CSV file
        try:
            df = pd.read_csv(csv_file_path)
            experiment_ids = df[id_column].tolist()
            print(f"Successfully read CSV file, found {len(experiment_ids)} experiment IDs")
        except Exception as e:
            print(f"Failed to read CSV file: {str(e)}")
            return [], []
        
        # Create save directory
        os.makedirs(download_dir, exist_ok=True)
        
        def download_single_file(exp_id, retries=max_retries):
            """Download single file"""
            url = f"{base_url}/{exp_id}?image={image_type}&resolution={resolution}"
            file_path = os.path.join(download_dir, f"{exp_id}_{image_type}.nrrd")
            
            for attempt in range(retries):
                try:
                    # First request to get file size
                    with requests.get(url, stream=True) as r:
                        r.raise_for_status()
                        total_size = int(r.headers.get('content-length', 0))
                        
                        # Check if file already exists and size matches
                        if os.path.exists(file_path):
                            existing_size = os.path.getsize(file_path)
                            if total_size > 0 and existing_size == total_size:
                                print(f"✓ File already exists and is complete: {exp_id}")
                                return True
                        
                        # Start download
                        downloaded_size = 0
                        with open(file_path, "wb") as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    downloaded_size += len(chunk)
                        
                        # Verify after download
                        if total_size > 0 and downloaded_size != total_size:
                            raise Exception(f"File size mismatch: expected {total_size} bytes, got {downloaded_size} bytes")
                        
                        # Check if file is larger than minimum threshold
                        if os.path.getsize(file_path) < min_file_size:
                            raise Exception(f"File too small, possibly incomplete: only {os.path.getsize(file_path)} bytes")
                        
                        print(f"✓ Download successful: {exp_id} -> {file_path}")
                        return True
                        
                except Exception as e:
                    print(f"✗ Attempt {attempt + 1}/{retries} failed {exp_id}: {str(e)}")
                    if os.path.exists(file_path):
                        os.remove(file_path)  # Delete incomplete file
                    if attempt < retries - 1:
                        time.sleep(2)  # Wait 2 seconds before retry
                    continue
            
            return False
        
        # Record successful and failed IDs
        success_ids = []
        failed_ids = []
        
        # Batch download
        for exp_id in tqdm(experiment_ids, desc=f"Downloading {image_type} files"):
            if download_single_file(exp_id):
                success_ids.append(exp_id)
            else:
                failed_ids.append(exp_id)
        
        # Print summary report
        print("\nDownload Summary:")
        print(f"Successfully downloaded: {len(success_ids)} files")
        print(f"Failed downloads: {len(failed_ids)} files")
        if failed_ids:
            print("Failed IDs:", failed_ids)
        
        return success_ids, failed_ids
    
    def preprocess_annotation_data(self):
        """
        Safely optimized annotation data preprocessing to ensure completely consistent results
        """
        print("Starting annotation data preprocessing...")
        
        # Create left/right hemisphere labels (keep unchanged)
        z_mid = self.anno.shape[2] // 2
        annot_labeled = np.where(
            np.arange(self.anno.shape[2]) < z_mid,
            np.char.add(self.anno.astype(str), "_left"),
            np.char.add(self.anno.astype(str), "_right")
        )
        
        # Fix asymmetric regions (keep original logic)
        unique_elements = np.unique(annot_labeled)
        
        print("Fixing asymmetric regions...")
        for area in tqdm(unique_elements, desc="Fixing asymmetric regions"):
            if area.endswith('_right') and area.replace('_right', '_left') not in unique_elements:
                x, y, z = np.where(annot_labeled == area)
                if len(z) > 0:
                    symmetric_z = self.anno.shape[2] - 1 - z
                    for i in range(len(x)):
                        if 0 <= symmetric_z[i] < self.anno.shape[2]:
                            annot_labeled[x[i], y[i], symmetric_z[i]] = area.replace('_right', '_left')
        
        # Update unique elements list
        unique_elements = np.unique(annot_labeled)
        
        # Optimize mask calculation part, add progress bar
        print("Precomputing region masks...")
        area_masks = {}
        
        for area in tqdm(unique_elements, desc="Calculating region masks"):
            mask = (annot_labeled == area)
            if mask.sum() > 0:  # Only save regions with voxels
                area_masks[area] = mask
        
        valid_areas = list(area_masks.keys())
        print(f"Preprocessing completed, total {len(valid_areas)} valid regions")
        
        return annot_labeled, area_masks, valid_areas
    
    def process_experiment_fast(self, experiment_id, annot_labeled, area_masks, valid_areas, 
                            base_dir, output_dir=None, use_projection_density=True):
        """
        Process single experiment data
        
        Parameters:
            experiment_id: Experiment ID (e.g., 100140756)
            annot_labeled: 3D annotation array
            area_masks: Precomputed brain region mask dictionary
            valid_areas: List of valid brain regions
            base_dir: Data base directory
            output_dir: Result save path (optional)
            use_projection_density: Whether to use projection density, False uses projection energy
            
        Returns:
            dict: Dictionary containing experiment results
        """
        # 1. Dynamically generate file paths
        if use_projection_density:
            data_file = os.path.join(base_dir, f"projection_density/{experiment_id}_projection_density.nrrd")
            data_type = "projection_density"
        else:
            data_file = os.path.join(base_dir, f"projection_energy/{experiment_id}_projection_energy.nrrd")
            data_type = "projection_energy"
        
        injection_file = os.path.join(base_dir, f"injection_fraction/{experiment_id}_injection_fraction.nrrd")
        
        print(f"\n▶ Starting to process experiment {experiment_id}...")
        
        # 2. Load data (with error handling)
        try:
            print("├─ Loading data files...")
            data_array, _ = nrrd.read(data_file)
            inf, _ = nrrd.read(injection_file)
        except FileNotFoundError as e:
            error_msg = f"└─ File does not exist: {str(e)}"
            print(error_msg)
            return {'experiment_id': experiment_id, 'error': 'file_not_found'}
        except Exception as e:
            error_msg = f"└─ Error reading data: {str(e)}"
            print(error_msg)
            return {'experiment_id': experiment_id, 'error': 'read_error'}
        
        # 3. Calculate injection center
        print("├─ Calculating injection center...")
        injection_mask = inf >= 1
        if not injection_mask.any():
            print("└─ No valid injection region detected")
            return {'experiment_id': experiment_id, 'injection_position': 'unknown'}
        
        weights = data_array[injection_mask]
        centroid = [np.average(coords, weights=weights) 
                for coords in np.where(injection_mask)]
        
        # 4. Get injection position
        try:
            inj_pos = annot_labeled[tuple(np.round(centroid).astype(int))]
            print(f"├─ Injection position: {inj_pos}")
        except IndexError:
            inj_pos = 'unknown'
            print("├─ Injection position: Unknown (coordinates out of bounds)")
        
        # 5. Calculate mean data for each brain region
        print("├─ Calculating brain region data means...")
        data_means = {}
        for area in tqdm(valid_areas, desc=f"Processing {data_type}", leave=False):
            mask = area_masks.get(area)
            if mask is not None and mask.shape == data_array.shape:
                masked_data = data_array[mask]
                nonzero_data = masked_data[masked_data > 0]  # Ignore zero values
                data_means[area] = float(nonzero_data.mean()) if len(nonzero_data) > 0 else 0.0
            else:
                data_means[area] = 0.0
        
        # Build result dictionary
        result = {
            'experiment_id': experiment_id,
            'injection_position': inj_pos,
            **data_means,
            '_meta': {
                'stat_method': f'mean_of_nonzero_{data_type}',
                'data_type': data_type
            }
        }
        
        # 6. Save results (if output directory specified)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{experiment_id}_results.csv")
            try:
                pd.DataFrame([result]).to_csv(output_path, index=False)
                print(f"└─ Results saved to: {output_path}")
                # Output success flag file
                flag_path = os.path.join(output_dir, f"{experiment_id}.COMPLETED")
                open(flag_path, 'w').close()
                print(f"✓ Flag file generated: {flag_path}")
            except Exception as e:
                print(f"└─ Save failed: {str(e)}")
        else:
            print("└─ Processing completed (file not saved)")
        
        return result
    
    def batch_process_experiments_sequential(self, experiment_ids, annot_labeled, area_masks, valid_areas,
                                            base_dir, output_dir, use_projection_density=True):
        """
        Batch process experimental data (sequential version, no parallel processing)
        
        Args:
            experiment_ids: List of experiment IDs
            annot_labeled: Labeled annotation data
            area_masks: Region mask dictionary
            valid_areas: List of valid regions
            base_dir: Data base directory
            output_dir: Output directory
            use_projection_density: Whether to use projection density
            
        Returns:
            DataFrame: Combined results
        """
        print("Starting batch processing of experimental data (sequential version)...")
        print(f"Total experiments to process: {len(experiment_ids)}")
        
        all_results = []
        successful_count = 0
        failed_count = 0
        skipped_count = 0
        
        for i, exp_id in enumerate(tqdm(experiment_ids, desc="Processing experiments")):
            # Check if already processed
            flag_path = os.path.join(output_dir, f"{exp_id}.COMPLETED")
            if os.path.exists(flag_path):
                print(f"\n[{i+1}/{len(experiment_ids)}] Skipping already processed experiment {exp_id}")
                skipped_count += 1
                continue
                
            print(f"\n[{i+1}/{len(experiment_ids)}] Processing experiment {exp_id}")
            
            result = self.process_experiment_fast(
                experiment_id=exp_id,
                annot_labeled=annot_labeled,
                area_masks=area_masks,
                valid_areas=valid_areas,
                base_dir=base_dir,
                output_dir=output_dir,
                use_projection_density=use_projection_density
            )
            
            # Count successes and failures
            if 'error' in result or result.get('injection_position') == 'unknown':
                failed_count += 1
            else:
                successful_count += 1
                all_results.append(result)
            
            # Show progress every 10 experiments
            if (i + 1) % 10 == 0:
                print(f"\n=== Progress: {i+1}/{len(experiment_ids)} ===")
                print(f"Success: {successful_count}, Failed: {failed_count}, Skipped: {skipped_count}")
        
        # Save combined results
        if all_results:
            combined_df = pd.DataFrame(all_results)
            combined_output = os.path.join(output_dir, "combined_results.csv")
            combined_df.to_csv(combined_output, index=False)
            print(f"\nBatch processing completed!")
            print(f"Successfully processed: {successful_count} experiments")
            print(f"Failed: {failed_count} experiments")
            print(f"Skipped: {skipped_count} already processed experiments")
            print(f"Combined results saved to: {combined_output}")
            
            return combined_df
        else:
            print("\nBatch processing completed, but no successful results")
            return pd.DataFrame()
    
    def save_single_result(self, experiment_id, annot_labeled, area_masks, valid_areas, base_dir, output_dir):
        """
        Process single experiment and directly save as CSV
        
        Args:
            experiment_id: Experiment ID
            annot_labeled: Labeled annotation data
            area_masks: Region mask dictionary
            valid_areas: List of valid regions
            base_dir: Data base directory
            output_dir: Output directory
            
        Returns:
            str: Saved file path
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Process experimental data
        result = self.process_experiment_fast(
            experiment_id=experiment_id,
            annot_labeled=annot_labeled,
            area_masks=area_masks,
            valid_areas=valid_areas,
            base_dir=base_dir,
            output_dir=output_dir
        )
        
        # Convert to DataFrame and save
        df = pd.DataFrame([result])
        output_path = os.path.join(output_dir, f"{experiment_id}_results.csv")
        df.to_csv(output_path, index=False)
        
        return output_path