import pandas as pd
import numpy as np
import re
from collections import defaultdict, Counter
import networkx as nx
from tqdm import tqdm
from pyswcloader import brain
import os

class SWCPathProcessor:
    """SWC Path Processor - Specialized for processing neuron path data"""

    def __init__(self, allen_brain_tree, stl_acro_dict):
        """
        Initialize processor
        
        Args:
            allen_brain_tree: Brain region tree structure
            stl_acro_dict: Abbreviation dictionary
        """
        self.allen_brain_tree = allen_brain_tree
        self.stl_acro_dict = stl_acro_dict
        self.annotated_tree = self.annotate_tree_with_dict()
    
    def annotate_tree_with_dict(self):
        """Add annotations to brain region tree"""
        allen_brain_tree_anno = self.allen_brain_tree
        for node in allen_brain_tree_anno.all_nodes():
            region_name = node.identifier
            if region_name in self.stl_acro_dict:
                annotation = self.stl_acro_dict[region_name]
                allen_brain_tree_anno.get_node(region_name).tag = annotation
        return allen_brain_tree_anno
    
    def has_descendant_in_matrix(self, tree, node_id, matrix_nodes):
        """Check if any descendant of current node is in matrix (recursive)"""
        for child in tree.children(node_id):
            child_node = tree.get_node(child.identifier)
            if hasattr(child_node, 'tag') and child_node.tag in matrix_nodes:
                return True
            if self.has_descendant_in_matrix(tree, child.identifier, matrix_nodes):
                return True
        return False
    
    def filter_problematic_nodes(self, directed_df, keys_for_values):
        """
        Filter problematic nodes
        
        Args:
            directed_df: Directed graph data
            keys_for_values: Key-value mapping dictionary
            
        Returns:
            set: Set of nodes to filter
        """
        tag_to_id = {}
        for node in self.annotated_tree.all_nodes():
            if hasattr(node, 'tag') and node.tag:
                tag_to_id[node.tag] = node.identifier

        all_matrix_nodes = set(directed_df.index) | set(directed_df.columns)
        result_set = set()

        for tag in all_matrix_nodes:
            if tag not in tag_to_id:
                continue
            node_id = tag_to_id[tag]
            if self.has_descendant_in_matrix(self.annotated_tree, node_id, all_matrix_nodes):
                result_set.add(tag)

        keys = [k for k, v in keys_for_values.items() if v in result_set]
        keys.append(0)
        return set(keys)
    
    def merge_consecutive_nodes(self, path):
        """Merge consecutive identical nodes (handle region self-connectivity)"""
        if not path:  # Return directly for empty path
            return ""
        
        # Regular expression to extract region ID and value
        pattern = re.compile(r'(\d+)\((\d+)\)')
        nodes = []
        
        # Parse all nodes
        for part in path.split('→'):
            match = pattern.match(part)
            if match:
                region_id = int(match.group(1))
                count = int(match.group(2))
                nodes.append((region_id, count))
        
        # Merge consecutive nodes with same region ID
        merged = []
        for region_id, count in nodes:
            if merged and merged[-1][0] == region_id:
                # Merge into previous node, accumulate values
                merged[-1] = (region_id, merged[-1][1] + count)
            else:
                # Add new node
                merged.append((region_id, count))
        
        # Regenerate path string
        return '→'.join([f"{r}({c})" for r, c in merged])
    
    def process_compressed_path(self, path, keys_to_remove):
        """Delete specified parent nodes in path"""
        # Split path into node list
        nodes = path.split('→')
        filtered_nodes = []
        for node in nodes:
            # Extract region ID (part before parentheses)
            region_id_str = node.split('(')[0].strip()
            # Convert to integer
            try:
                region_id = int(region_id_str)
            except:
                # Skip node if format error (adjust according to actual situation)
                continue
            # Keep node if region ID not in keys to remove
            if region_id not in keys_to_remove:
                filtered_nodes.append(node)
        # Reconnect remaining nodes
        return '→'.join(filtered_nodes) if filtered_nodes else ''
    
    def remove_weights(self, path):
        """Remove weight information from path (parentheses and numbers)"""
        # Use regex to remove all parentheses and their contents
        return re.sub(r'\(\d+\)', '', path)
    
    def replace_nodes_with_acronyms(self, path_str):
        """Replace node IDs in path with abbreviation names"""
        # Split path into node list
        nodes = path_str.split('→')
        # Iterate through each node: replace if in dictionary, otherwise keep original value (avoid missing key error)
        replaced_nodes = [
            str(self.stl_acro_dict.get(int(node), node))  # Handle non-numeric nodes (like 484682470)
            if node.isdigit() else node
            for node in nodes
        ]
        # Reconnect as path string
        return '→'.join(replaced_nodes)
    
    def split_path_to_columns(self, df):
        """
        Split clean_path into three columns:
        - start_node: Start node (string)
        - end_node: End node (string)
        - middle_nodes: Set of middle nodes (set type)
        """
        # Vectorized operation to extract start and end nodes (efficient)
        split_paths = df['clean_path'].str.split('→')
        df['start_node'] = split_paths.str[0]
        df['end_node'] = split_paths.str[-1]
        
        # Process middle nodes as sets
        def get_middle_set(path):
            if not isinstance(path, str):
                return set()
            nodes = path.split('→')
            return set(nodes[1:-1]) if len(nodes) > 2 else set()
        
        df['middle_nodes'] = df['clean_path'].apply(get_middle_set)
        return df
    
    def process_path_pipeline(self, combined_df, keys_set):
        """
        Integrated path processing pipeline - Combine multiple processing steps into one
        
        Args:
            combined_df: DataFrame containing compressed paths
            keys_set: Set of nodes to filter
            
        Returns:
            DataFrame: Processed DataFrame
        """
        # Step 1: Process compressed paths, filter specified nodes
        combined_df['processed_compressed_path'] = combined_df['compressed_path'].apply(
            lambda x: self.process_compressed_path(x, keys_set)
        )
        
        # Step 2: Merge consecutive identical nodes
        combined_df["merged_compressed_path"] = combined_df["processed_compressed_path"].apply(
            self.merge_consecutive_nodes
        )
        
        # Step 3: Clean path weight information
        combined_df['clean_path'] = combined_df['merged_compressed_path'].apply(
            self.remove_weights
        )
        
        # Step 4: Replace node IDs with abbreviation names
        combined_df['replace_path'] = combined_df['clean_path'].apply(
            self.replace_nodes_with_acronyms
        )
        
        # Step 5: Split paths into column format
        combined_df = self.split_path_to_columns(combined_df)
        
        return combined_df

    @staticmethod
    def build_path_with_minimal_data(df, start, end):
        """
        Build path from start to end using minimal data rows
        
        Args:
            df: DataFrame containing node pairs and frequencies
            start: Start node
            end: End node
            
        Returns:
            tuple: (Path string, number of edges used)
        """
        sorted_df = df.sort_values('Frequency', ascending=False).reset_index(drop=True)
        start_added = False
        end_added = False
        
        for n in range(1, len(sorted_df) + 1):
            G = nx.DiGraph()
            
            for i in range(n):
                source, target = sorted_df.iloc[i]['Node Pair'].split('→')
                G.add_edge(source, target)
                
                if source == start or target == start:
                    start_added = True
                if source == end or target == end:
                    end_added = True
            
            if start_added and end_added:
                try:
                    if nx.has_path(G, start, end):
                        path = nx.shortest_path(G, source=start, target=end)
                        return "→".join(path), n
                except nx.NodeNotFound:
                    continue
        
        return "", 0
    
    def build_representative_paths(self, combined_df, save_progress_path=None):
        """
        Build representative paths for all unique start-end pairs (with resume capability)
        
        Args:
            combined_df: DataFrame containing all path data
            save_progress_path: Progress save path
            
        Returns:
            DataFrame: Result containing representative paths
        """
        # Split paths into columns
        df_with_nodes = self.split_path_to_columns(combined_df)
        unique_pairs = df_with_nodes[['start_node', 'end_node']].drop_duplicates().reset_index(drop=True)
        
        # Initialize result columns
        unique_pairs['path'] = ""
        unique_pairs['path_length'] = 0
        unique_pairs['edges_used'] = 0
        
        # Resume from breakpoint: Check if there's saved progress
        processed_count = 0
        if save_progress_path and os.path.exists(save_progress_path):
            try:
                progress_df = pd.read_csv(save_progress_path)
                
                # Create progress dictionary, key is (start_node, end_node), value is entire row data
                progress_dict = {}
                for _, row in progress_df.iterrows():
                    key = (str(row['start_node']), str(row['end_node']))
                    progress_dict[key] = row
                
                # Update current results and count processed items
                for idx in range(len(unique_pairs)):
                    row = unique_pairs.loc[idx]
                    key = (str(row['start_node']), str(row['end_node']))
                    
                    if key in progress_dict:
                        progress_row = progress_dict[key]
                        # Stricter judgment: only update when there's non-empty and valid path string
                        if (pd.notna(progress_row['path']) and 
                            progress_row['path'].strip() and 
                            progress_row['path'] not in ["No paths found", "No valid edges"]):
                            unique_pairs.at[idx, 'path'] = progress_row['path']
                            unique_pairs.at[idx, 'path_length'] = progress_row.get('path_length', 0)
                            unique_pairs.at[idx, 'edges_used'] = progress_row.get('edges_used', 0)
                            processed_count += 1
                
                print(f"Resuming from breakpoint: {processed_count} path pairs already processed")
                
            except Exception as e:
                print(f"Failed to load progress file, starting from scratch: {e}")
        
        # Process each start-end pair
        total_pairs = len(unique_pairs)
        
        # Create a list to track indices of rows that need processing
        indices_to_process = []
        for idx in range(total_pairs):
            row = unique_pairs.loc[idx]
            
            # Check if this row needs processing
            if not (pd.notna(row['path']) and 
                    row['path'].strip() and 
                    row['path'] not in ["No paths found", "No valid edges"]):
                indices_to_process.append(idx)
        
        print(f"Need to process {len(indices_to_process)} path pairs")
        
        # Use tqdm to process rows that need processing, initial value set to already processed count
        for idx in tqdm(indices_to_process, desc="Building representative paths", initial=processed_count, total=total_pairs):
            row = unique_pairs.loc[idx]
            start_node = str(row['start_node'])
            end_node = str(row['end_node'])
            
            # Filter target paths
            target_paths = df_with_nodes[
                (df_with_nodes['start_node'] == start_node) & 
                (df_with_nodes['end_node'] == end_node)
            ].copy()
            
            if len(target_paths) == 0:
                unique_pairs.at[idx, 'path'] = "No paths found"
                processed_count += 1
            else:
                # Prepare middle nodes and deduplicate
                target_paths.loc[:, 'middle_nodes'] = target_paths['middle_nodes'].apply(
                    lambda x: tuple(sorted(x)) if x else ()
                )
                
                deduped_df = target_paths.drop_duplicates(
                    subset=['neuron_id', 'middle_nodes'],
                    keep='first'
                )
                
                # Extract all consecutive node pairs and count frequencies
                node_pairs = []
                for path in deduped_df['clean_path']:
                    nodes = path.split('→')
                    pairs = [f"{nodes[i]}→{nodes[i+1]}" for i in range(len(nodes) - 1)]
                    node_pairs.extend(pairs)
                
                if len(node_pairs) == 0:
                    unique_pairs.at[idx, 'path'] = "No valid edges"
                    processed_count += 1
                else:
                    pair_counter = Counter(node_pairs)
                    pair_freq_df = pd.DataFrame(pair_counter.most_common(), 
                                                columns=['Node Pair', 'Frequency'])
                    
                    # Build path
                    path_str, edges_used = self.build_path_with_minimal_data(pair_freq_df, start_node, end_node)
                    
                    # Save results
                    unique_pairs.at[idx, 'path'] = path_str
                    unique_pairs.at[idx, 'edges_used'] = edges_used
                    unique_pairs.at[idx, 'path_length'] = len(path_str.split('→')) - 1 if path_str else 0
                    processed_count += 1
            
            # Periodically save progress (every 100 pairs or last pair)
            if save_progress_path and (processed_count % 100 == 0 or processed_count == total_pairs):
                unique_pairs.to_csv(save_progress_path, index=False)
                print(f"Progress saved: {processed_count}/{total_pairs} path pairs processed")
        
        # Final save to ensure data completeness
        if save_progress_path:
            unique_pairs.to_csv(save_progress_path, index=False)
            print(f"Final results saved: Total {total_pairs} path pairs processed")
        
        return unique_pairs