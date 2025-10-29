import pandas as pd
import numpy as np
import math
import os
from collections import defaultdict
from itertools import groupby
import networkx as nx
from tqdm import tqdm
from pyswcloader import swc, brain

class SWCProcessor:
    """SWC file processor - for raw SWC data processing and feature extraction"""
    
    def __init__(self, annotation, resolution):
        """
        Initialize processor
        
        Args:
            annotation: Brain region annotation data
            resolution: Resolution parameters
        """
        self.annotation = annotation
        self.resolution = resolution
        self.graph = None
        self.edge_data = None
    
    def process_swc_file(self, file_path):
        """
        Process single SWC file
        
        Args:
            file_path: SWC file path
            
        Returns:
            tuple: (graph object, edge data DataFrame, regional path list)
        """
        # Preprocess SWC data
        data = swc.swc_preprocess(file_path)
        
        # Add region information
        data['region'] = data.apply(
            lambda x: brain.find_region(x[['x','y','z']], self.annotation, self.resolution),
            axis=1
        )
        
        # Build connection graph
        G, edge_df = self._build_connection_graph(data, file_path)
        self.graph = G
        self.edge_data = edge_df
        
        # Extract regional paths
        regional_paths = self._get_regional_paths_optimized(G)
        
        return G, edge_df, regional_paths
    
    def _build_connection_graph(self, data, file_path):
        """Build directed graph of neuronal region connections"""
        G = nx.DiGraph()
        edge_data = []
        neuron_name = os.path.basename(file_path).split('.')[0]
        
        # Add nodes
        for idx in data.index:
            G.add_node(
                idx,
                region=data.loc[idx, 'region'],
                pos=(data.loc[idx, 'x'], data.loc[idx, 'y'], data.loc[idx, 'z'])
            )
        
        # Find root node
        root_node = data[data['parent'] == -1].index[0]
        
        # Build connection relationships
        for idx in data.index:
            if idx == root_node:
                continue
                
            parent_idx = data.loc[idx, 'parent']
            if parent_idx not in data.index:
                continue
            
            # Calculate geometric features
            child_coords = data.loc[idx, ['x', 'y', 'z']].values
            parent_coords = data.loc[parent_idx, ['x', 'y', 'z']].values
            distance = math.dist(child_coords, parent_coords)
            
            # Region information
            child_region = data.loc[idx, 'region']
            parent_region = data.loc[parent_idx, 'region']
            
            # Add edge
            edge_attrs = {
                'from_region': parent_region,
                'to_region': child_region,
                'length': distance,
                'direction_vector': tuple(child_coords - parent_coords)
            }
            G.add_edge(parent_idx, idx, **edge_attrs)
            
            # Record edge information
            edge_data.append({
                'parent_id': parent_idx,
                'child_id': idx,
                'parent_region': parent_region,
                'child_region': child_region,
                'length': distance,
                'x1': parent_coords[0], 'y1': parent_coords[1], 'z1': parent_coords[2],
                'x2': child_coords[0], 'y2': child_coords[1], 'z2': child_coords[2]
            })
        
        return G, pd.DataFrame(edge_data)
    
    def _get_regional_paths_optimized(self, G):
        """Topological sorting + dynamic programming optimization to get regional paths"""
        if not G:
            return []
        
        topo_order = list(nx.topological_sort(G))
        root = topo_order[0]
        
        # Pre-cache information
        region_cache = {n: G.nodes[n]['region'] for n in G.nodes}
        edge_length_cache = {(u, v): d['length'] for u, v, d in G.edges(data=True)}
        
        # Dynamic programming
        dp = defaultdict(list)
        dp[root] = [{
            'node_path': [root],
            'regional_path': [region_cache[root]],
            'length': 0
        }]
        
        for node in topo_order[1:]:
            predecessors = list(G.predecessors(node))
            for pred in predecessors:
                for path in dp[pred]:
                    new_path = {
                        'node_path': path['node_path'] + [node],
                        'regional_path': path['regional_path'] + [region_cache[node]],
                        'length': path['length'] + edge_length_cache[(pred, node)]
                    }
                    dp[node].append(new_path)
        
        # Collect leaf node paths
        leaves = [n for n in G.nodes if G.out_degree(n) == 0]
        return [path for leaf in leaves for path in dp[leaf]]
    
    @staticmethod
    def compress_path(path):
        """Compress paths with consecutive identical regions"""
        compressed = []
        for region, group in groupby(path):
            count = len(list(group))
            compressed.append(f"{region}({count})" if count > 1 else str(region))
        return "→".join(compressed)
    
    def extract_path_features(self, regional_paths, neuron_id, folder_name):
        """
        Extract features from regional paths
        
        Args:
            regional_paths: Regional path list
            neuron_id: Neuron ID
            folder_name: Folder name
            
        Returns:
            list: Path feature dictionary list
        """
        features = []
        for path_id, path_data in enumerate(regional_paths):
            compressed = self.compress_path(path_data['regional_path'])
            features.append({
                'neuron_folder': folder_name,
                'neuron_id': neuron_id,
                'path_id': path_id,
                'compressed_path': compressed,
                'path_length': len(path_data['regional_path']),
                'unique_regions': len(set(path_data['regional_path'])),
                'is_pure': len(set(path_data['regional_path'])) == 1
            })
        return features

class BatchSWCProcessor:
    """Batch SWC file processor"""
    
    def __init__(self, annotation, resolution):
        self.annotation = annotation
        self.resolution = resolution
        self.processor = SWCProcessor(annotation, resolution)
    
    def process_folder(self, folder_path, save_results=True):
        """
        Process all SWC files in folder
        
        Args:
            folder_path: Folder path
            save_results: Whether to save results
            
        Returns:
            list: Path features for all neurons
        """
        folder_results = []
        swc_files = [f for f in os.listdir(folder_path) if f.endswith('.swc')]
        folder_name = os.path.basename(folder_path)
        
        for swc_file in tqdm(swc_files, desc=f'Processing {folder_name}'):
            try:
                swc_path = os.path.join(folder_path, swc_file)
                neuron_id = swc_file.split('.')[0]
                
                # Process single SWC file
                G, edges, regional_paths = self.processor.process_swc_file(swc_path)
                
                # Extract features
                path_features = self.processor.extract_path_features(
                    regional_paths, neuron_id, folder_name
                )
                folder_results.extend(path_features)
                
            except Exception as e:
                print(f"Error processing file {swc_file}: {str(e)}")
                continue
        
        # Save results
        if save_results and folder_results:
            df = pd.DataFrame(folder_results)
            output_path = os.path.join(folder_path, 'regional_paths.csv')
            df.to_csv(output_path, index=False)
        
        return folder_results
    
    def process_batch_folders(self, root_path):
        """
        Batch process multiple folders
        
        Args:
            root_path: Root directory path
            
        Returns:
            pd.DataFrame: Merged results
        """
        all_results = []
        folders = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
        
        for folder in tqdm(folders, desc='Processing folders'):
            folder_path = os.path.join(root_path, folder, 'swc_allen_space')
            
            if not os.path.exists(folder_path):
                continue
                
            # Check if results already exist
            result_file = os.path.join(folder_path, 'regional_paths.csv')
            if os.path.exists(result_file):
                try:
                    df = pd.read_csv(result_file)
                    all_results.extend(df.to_dict('records'))
                    continue
                except:
                    pass
            
            # Process folder
            folder_results = self.process_folder(folder_path, save_results=True)
            all_results.extend(folder_results)
        
        # Save merged results
        if all_results:
            final_df = pd.DataFrame(all_results)
            final_output = os.path.join(root_path, 'all_regional_paths.csv')
            final_df.to_csv(final_output, index=False)
            print(f"All results saved to: {final_output}")
        
        return pd.DataFrame(all_results) if all_results else pd.DataFrame()