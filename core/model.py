import pandas as pd
import numpy as np
import ast
import re
from collections import defaultdict, Counter
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
import networkx as nx
from tqdm import tqdm
import os


# Custom activation function (according to your actual definition)
def custom_activation(x):
    """Improved activation function allowing small range negative outputs"""
    return tf.where(x < 0, 
                  1 * tf.nn.softplus(x),  # Maintain tiny positive values for deep negatives
                  tf.nn.softplus(x))          # Normal positive response

class SequenceDataProcessor:

    def __init__(self, stl_acro_dict, gene_filled_result_path):
        """
        Initialize sequence data processor
        Args:
            stl_acro_dict: Brain region abbreviation dictionary
            gene_filled_result_path: Gene filling result file path
        """
        self.stl_acro_dict = stl_acro_dict
        self.gene_filled_result_path = gene_filled_result_path
        self.index_mapping = None
        self.gene_embeddings_df = None
        self.pca = None
        self.result = None
        
    def load_and_prepare_data(self, filtered_results_path, window_size=5 ,n_components=64):
        """
        Load and prepare data
        
        Args:
            filtered_results_path: Filtered results file path
            window_size: Sliding window size
            
        Returns:
            tuple: (X, y, max_len, pca)
        """
        
        # Create index mapping
        self.index_mapping = {v: k for k, v in self.stl_acro_dict.items()}
        
        # Load filtered results
        filtered_results_df = pd.read_csv(filtered_results_path)
        filtered_results_df['strength'] = filtered_results_df['strength'].apply(ast.literal_eval)
        
        # Generate named_seqs (sequence value lists)
        named_seqs = [
            row['path'].split('→') 
            for _, row in filtered_results_df.iterrows()
        ]
        
        # Generate strength_seqs (strength value lists)
        strength_seqs = [
            row['strength'] 
            for _, row in filtered_results_df.iterrows()
        ]
        
        # Prepare model input
        X, y, idx_to_name = self.prepare_model_input(named_seqs, strength_seqs, self.index_mapping)
        
        # Sliding window cutting
        new_X, new_y = self.sliding_window_cut(X, y, window_size=window_size)
        
        # Deduplication and filtering
        X, y = self.deduplicate_and_filter(new_X, new_y)
        
        # Log transformation
        y_log = np.log2(y + 1)
        
        max_len = X.shape[1]
        
        # Load gene data and process
        gene_filled_result = pd.read_csv(self.gene_filled_result_path, index_col=0)
        
        column_means = gene_filled_result.mean()
        std_dev = gene_filled_result.std()
        threshold = std_dev.quantile(0.5)
        selected_columns = std_dev[std_dev >= threshold].index
        top_50_percent_columns = gene_filled_result[selected_columns]
        result = top_50_percent_columns
        #####Remove columns ending with rik, starting with GM
        result = result.loc[:, ~result.columns.str.endswith('Rik') & ~result.columns.str.startswith('Gm')]
        ########First by column, then by row
        result = result.apply(lambda col: (col - col.mean()) / col.std(), axis=0)
        result = result.apply(lambda row: (row - row.mean()) / row.std(), axis=1)
        self.result = result
        
        # PCA dimensionality reduction
        self.pca = PCA(n_components=n_components)
        result_pca = self.pca.fit_transform(result)
        pca_df = pd.DataFrame(result_pca, index=result.index)
        self.gene_embeddings_df = pca_df
        
        return X, y_log, max_len, self.pca
    
    def prepare_model_input(self, named_sequences, strength_sequences, node_to_idx):
        """
        Prepare model input data
        """
        
        # Convert to index sequences
        X_idx = [[node_to_idx[node] for node in seq] for seq in named_sequences]
        
        # Calculate maximum length
        max_len = max(len(seq) for seq in named_sequences)
        
        # Pad node indices (X)
        X_padded = pad_sequences(
            X_idx,
            maxlen=max_len,
            padding='post',
            value=0,
            dtype='int32'
        )
        
        # Pad strength values (y) preserve floating point numbers
        y_padded = pad_sequences(
            strength_sequences,
            maxlen=max_len,
            padding='post',
            value=0.0,
            dtype='float32'
        )
        
        # Create index to name mapping
        idx_to_name = {idx: name for name, idx in node_to_idx.items()}
        
        return X_padded, y_padded, idx_to_name
    
    def sliding_window_cut(self, X, y, window_size=3):
        """
        Sliding window cutting
        """
        
        new_X_list = []
        new_y_list = []
        
        for i in range(X.shape[0]):
            row_x = X[i]
            row_y = y[i]
            
            start = 0
            n = len(row_x)
            
            while start + window_size <= n:
                window_x = row_x[start:start+window_size]
                window_y = row_y[start:start+window_size]
                
                new_X_list.append(window_x)
                new_y_list.append(window_y)
                
                if window_x[-1] == 0:
                    break
                    
                start += 1
        
        new_X = np.array(new_X_list, dtype=np.int32)
        new_y = np.array(new_y_list, dtype=np.float32)
        
        return new_X, new_y
    
    def deduplicate_and_filter(self, new_X, new_y):
        """
        Deduplication and filtering
        """
        
        assert new_X.ndim == 2, "new_X must be a 2D array"
        assert new_y.ndim == 2, "new_y must be a 2D array"
        assert new_X.shape[0] == new_y.shape[0], "X and y must have the same number of rows"
        assert new_X.shape[1] == new_y.shape[1], "X and y must have the same number of columns"
        
        n_rows, n_cols = new_X.shape
        
        dtype = []
        for i in range(n_cols):
            dtype.append((f'X{i}', new_X.dtype))
        for i in range(n_cols):
            dtype.append((f'y{i}', new_y.dtype))
        
        combined = np.empty(n_rows, dtype=dtype)
        
        for i in range(n_cols):
            combined[f'X{i}'] = new_X[:, i]
            combined[f'y{i}'] = new_y[:, i]
        
        last_col_name = f'X{n_cols-1}'
        mask = (combined[last_col_name] != 0)
        
        filtered = combined[mask]
        
        _, unique_indices = np.unique(filtered, return_index=True, axis=0)
        dedup_combined = filtered[unique_indices]
        
        dedup_X = np.empty((len(dedup_combined), n_cols), dtype=new_X.dtype)
        dedup_y = np.empty((len(dedup_combined), n_cols), dtype=new_y.dtype)
        
        for i in range(n_cols):
            dedup_X[:, i] = dedup_combined[f'X{i}']
            dedup_y[:, i] = dedup_combined[f'y{i}']
        
        return dedup_X, dedup_y
    
    def prepare_gene_sequences(self, node_sequences, max_len):
        """
        Prepare gene sequences
        """
        
        num_samples = len(node_sequences)
        embed_dim = self.gene_embeddings_df.shape[1]
        gene_embed_sequences = np.zeros((num_samples, max_len, embed_dim))
        
        for i, seq in enumerate(node_sequences):
            for j, node_id in enumerate(seq):
                if node_id > 0:
                    gene_embed_sequences[i, j] = self.gene_embeddings_df.loc[node_id]
        
        return gene_embed_sequences
    
    def split_data(self, X, y, test_size=0.2, random_state=200054):
        """
        Split data
        """
        
        node_train, node_test, strength_train, strength_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
        
        strength_train_processed, node_train_processed = (strength_train, node_train)
        strength_test_processed, node_test_processed = (strength_test, node_test)
        
        return (node_train_processed, node_test_processed, 
                strength_train_processed, strength_test_processed)
    
    def prepare_final_data(self, node_train_processed, node_test_processed, 
                          strength_train_processed, strength_test_processed, max_len):
        """
        Prepare final data
        """
        
        # Prepare gene sequences
        gene_train = self.prepare_gene_sequences(node_train_processed, max_len)
        gene_test = self.prepare_gene_sequences(node_test_processed, max_len)
        
        # Prepare initial strength
        init_strength_train = strength_train_processed[:, 0:1]
        init_strength_test = strength_test_processed[:, 0:1]
        
        # Prepare shifted strength
        strength_train_shift = np.array([np.roll(row, -1)[:-1] for row in strength_train_processed])
        strength_test_shift = np.array([np.roll(row, -1)[:-1] for row in strength_test_processed])
        
        return (gene_train, gene_test, init_strength_train, init_strength_test,
                strength_train_shift, strength_test_shift)
    
    def create_adaptive_weighted_loss(self, seq_length, weights=None):
        """
        Create adaptive weighted loss function - supports arbitrary time steps
        
        Args:
            seq_length (int): Number of time steps
            weights (list): Optional, custom weight list, length must equal seq_length
        
        Returns:
            function: Configured loss function
        """
        # If weights not provided, use uniform weights
        if weights is None:
            weights = [1.0 / seq_length] * seq_length
        else:
            # Validate weight length matches
            if len(weights) != seq_length:
                raise ValueError(f"Weight list length({len(weights)}) must equal time steps({seq_length})")
        
        # Convert weights to Tensor constants
        time_weights_tensor = tf.constant(weights, dtype=tf.float32)
        
        def adaptive_weighted_loss(y_true, y_pred):
            """
            Adaptive weighted loss function
            
            Parameters:
                y_true: True values, shape (batch_size, seq_length)
                y_pred: Predicted values, shape (batch_size, seq_length)
            
            Returns:
                Weighted loss value
            """
            batch_size = tf.shape(y_true)[0]
            
            # Validate input time steps
            input_seq_length = tf.shape(y_true)[1]
            tf.debugging.assert_equal(
                input_seq_length, 
                seq_length,
                message=f"This loss function only supports sequences with {seq_length} time steps"
            )
            
            # Calculate base MAE loss
            mae_term = tf.abs(y_true - y_pred)
            
            # Expand weight dimensions for broadcasting to entire batch
            time_weights = tf.expand_dims(time_weights_tensor, axis=0)  # Shape becomes (1, seq_length)
            time_weights = tf.tile(time_weights, [batch_size, 1])  # Shape becomes (batch_size, seq_length)
            
            # Apply time step weights
            weighted_mae = mae_term * time_weights
            
            # Calculate weighted loss
            total_weight = tf.reduce_sum(time_weights)
            loss = tf.reduce_sum(weighted_mae) / total_weight
            
            return loss
        
        return adaptive_weighted_loss

    def get_pca(self):
        """
        Get PCA object for feature analysis
        
        Returns:
            PCA: Trained PCA object
        """
        return self.pca

    def build_true_autoregressive_model_with_k(self, max_len, gene_embed_dim=64):
        """Build autoregressive model with learnable error coefficient"""
        # Use max_len-1 to create loss function
        loss_function = self.create_adaptive_weighted_loss(max_len - 1)
        
        # Input layers
        gene_embed_input = tf.keras.Input(shape=(max_len, gene_embed_dim), name='gene_embed_input')
        init_strength_input = tf.keras.Input(shape=(1,), name='init_strength_input')
        
        # ====== Enhanced gene embedding processing path (using PReLU) ======
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(256, use_bias=True)
        )(gene_embed_input)
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.PReLU()
        )(x)
        
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(128, use_bias=True)
        )(x)
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.PReLU()
        )(x)
        
        processed_gene_embed = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(64, use_bias=True)
        )(x)
        processed_gene_embed = tf.keras.layers.TimeDistributed(
            tf.keras.layers.PReLU()
        )(processed_gene_embed)
        # ===========================================
        
        # Extract first time step's processed gene embedding information
        first_gene_embed = tf.keras.layers.Lambda(lambda x: x[:, 0, :])(processed_gene_embed)
        second_and_third = tf.keras.layers.Lambda(lambda x: x[:, 1:, :])(processed_gene_embed)
        
        # Concatenate initial strength and first gene embedding
        combined_init_input = tf.keras.layers.Concatenate(axis=-1)([init_strength_input, first_gene_embed])
        
        # Create autoregressive RNN layer
        autoregressive_cell = self.AutoregressiveCell(32)
        rnn_layer = tf.keras.layers.RNN(
            autoregressive_cell,
            return_sequences=True,
            return_state=False,
            unroll=False
        )
        
        # Initialize state - use concatenated information
        h0 = tf.keras.layers.Dense(32)(combined_init_input)
        # h0 = tf.keras.layers.PReLU()(h0)
        c0 = tf.keras.layers.Dense(32)(combined_init_input)
        # c0 = tf.keras.layers.PReLU()(c0)
        initial_state = [h0, c0]
        
        # Run autoregressive RNN (using processed gene embeddings)
        output = rnn_layer(
            second_and_third,
            initial_state=initial_state
        )
        
        # Output processing - use Lambda layer to wrap TensorFlow operations
        output = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=-1))(output)  # (batch_size, max_len)
        
        # ====== Add learnable error coefficient k ======
        # Create independent learnable coefficient k (initial value 1.0)
        ones_vector = tf.keras.layers.Lambda(lambda x: tf.ones_like(x[:, :1]))(output)
        k = tf.keras.layers.Dense(
            1, 
            activation=None, 
            use_bias=False,
            kernel_initializer='ones',  # Initialize to 1.0
            name='error_coefficient'
        )(ones_vector)  # Use unit vector with same batch size as output
        
        # Ensure k is scalar (but keep matching batch)
        k = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=-1))(k)  # Now shape is (batch_size,)
        
        # Expand k to same shape as output
        k_expanded = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(k)  # (batch_size, 1)
        
        # Get output shape information for tile operation
        output_shape = tf.keras.backend.int_shape(output)
        k_expanded = tf.keras.layers.Lambda(
            lambda x: tf.tile(x[0], [1, output_shape[1]])
        )([k_expanded])  # (batch_size, max_len)
        
        # Apply error coefficient: final prediction = model output * k
        final_output = tf.keras.layers.Multiply()([output, k_expanded])
        
        # Build model
        model = tf.keras.Model(
            inputs=[gene_embed_input, init_strength_input],
            outputs=final_output
        )
        
        # Compile model - use dynamically created loss function
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=loss_function,  # Use dynamically created loss function
            metrics=['mae', tf.keras.metrics.MeanAbsolutePercentageError()]
        )
        
        return model

    class MultiInputR2ScoreCallback(Callback):
        """R² callback function supporting multi-input models"""
        def __init__(self, validation_data):
            super().__init__()
            self.X_val_list, self.y_val = validation_data
            self.X_val_list = [np.array(x) for x in self.X_val_list]  # Ensure all inputs are NumPy arrays
        
        def on_epoch_end(self, epoch, logs=None):
            # Get model expected input shapes - fix shape acquisition method
            expected_shapes = []
            for input_tensor in self.model.inputs:
                # Handle different types of input shape representations
                if hasattr(input_tensor, 'shape') and hasattr(input_tensor.shape, 'as_list'):
                    expected_shapes.append(input_tensor.shape.as_list())
                elif hasattr(input_tensor, 'shape'):
                    # If shape is tuple or other type, use directly
                    expected_shapes.append(input_tensor.shape)
                else:
                    # If cannot get shape, use None as placeholder
                    expected_shapes.append(None)
            
            # Adjust each input shape to match model expectations
            adjusted_X_val = []
            for i, (input_data, expected_shape) in enumerate(zip(self.X_val_list, expected_shapes)):
                if expected_shape is None:
                    # If cannot get expected shape, use original data directly
                    adjusted_data = input_data
                elif len(expected_shape) == 3 and len(input_data.shape) == 3:
                    expected_timesteps = expected_shape[1]
                    # Take first N time steps
                    if input_data.shape[1] > expected_timesteps:
                        adjusted_data = input_data[:, :expected_timesteps, :]
                        print(f"Adjusting input {i} shape: {input_data.shape} -> {adjusted_data.shape}")
                    else:
                        adjusted_data = input_data
                else:
                    adjusted_data = input_data
                adjusted_X_val.append(adjusted_data)
            
            # Use adjusted inputs for prediction
            y_pred = self.model.predict(adjusted_X_val, verbose=0)
            
            # Calculate R² scores
            r2_scores = r2_score(self.y_val, y_pred, multioutput='raw_values')
            avg_r2 = np.mean(r2_scores)
            
            # Print results
            print(f"\nEpoch {epoch+1} Validation R² Scores:")
            for i, score in enumerate(r2_scores):
                print(f"  Output {i+1}: {score:.4f}")
            print(f"  Average R²: {avg_r2:.4f}")
            
            # Record to logs
            logs = logs or {}
            logs['val_r2'] = avg_r2
            for i, score in enumerate(r2_scores):
                logs[f'val_r2_output_{i+1}'] = score

    class AutoregressiveCell(tf.keras.layers.Layer):
        """Custom autoregressive cell"""
        def __init__(self, units, **kwargs):
            super().__init__(**kwargs)
            self.units = units
            self.state_size = [units, units]  # [h_state, c_state]
            self.output_size = 1  # Predict strength value
            
        def build(self, input_shape):
            # Input shape: (batch_size, gene_embed_dim)
            self.lstm_cell = tf.keras.layers.LSTMCell(self.units)
            self.output_dense = tf.keras.layers.Dense(1, activation=custom_activation)
            self.built = True
            
        @tf.autograph.experimental.do_not_convert
        def call(self, inputs, states, training=None, **kwargs):
            # Unpack states
            h_state, c_state = states
            
            # Directly use gene embedding as input
            lstm_input = inputs  # Shape: (batch_size, gene_embed_dim)
            
            # LSTM processing
            lstm_output, [new_h, new_c] = self.lstm_cell(
                lstm_input, 
                [h_state, c_state],
                training=training
            )
            
            # Predict current strength
            strength_pred = self.output_dense(lstm_output)
            
            return strength_pred, [new_h, new_c]

    def compute_gene_importance(self, model, dataset, target_timestep=-1, n_samples=100):
        """
        Calculate gene embedding importance scores
        
        Parameters:
            model: Trained autoregressive model
            dataset: Input dataset (gene_embed, init_strength)
            target_timestep: Target time step (default last)
            n_samples: Number of samples for calculation
            
        Returns:
            position_importance: Average importance per position [max_len]
            dimension_importance: Average importance per embedding dimension [embed_dim]
        """
        # Initialize importance matrices
        position_importance = np.zeros(model.input_shape[0][1])  # max_len
        dimension_importance = np.zeros(model.input_shape[0][2])  # embed_dim
        
        # Get samples
        gene_embeds, init_strengths = dataset
        sample_indices = np.random.choice(len(gene_embeds), n_samples, replace=False)
        
        # Add progress bar
        for idx in tqdm(sample_indices, desc="Calculating gene importance", unit="sample"):
            gene_embed = tf.convert_to_tensor(gene_embeds[idx][np.newaxis])
            init_strength = tf.convert_to_tensor(init_strengths[idx][np.newaxis])
            
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(gene_embed)
                predictions = model([gene_embed, init_strength])
                
                # Select target output (specific time step)
                target = predictions[:, target_timestep] if target_timestep >= 0 else predictions
            
            # Calculate gradients
            gradients = tape.gradient(target, gene_embed)
            
            # Calculate position importance (L2 norm)
            per_position = tf.norm(gradients, axis=-1).numpy().squeeze()
            position_importance += per_position
            
            # Calculate dimension importance (absolute value average)
            per_dimension = tf.reduce_mean(tf.abs(gradients), axis=[0,1]).numpy()
            dimension_importance += per_dimension
        
        # Average importance
        position_importance /= n_samples
        dimension_importance /= n_samples
        
        return position_importance, dimension_importance

    def get_gene_importance_from_pca(self, dimension_importance, gene_names=None):
        """
        Calculate original gene importance from PCA dimension importance
        
        Parameters:
            dimension_importance: PCA dimension importance scores [n_components]
            gene_names: Original gene name list [n_genes], if None use self.result column names
            
        Returns:
            gene_importance: Original gene importance scores [n_genes]
            gene_importance_df: DataFrame containing gene names and importance scores
        """
        if gene_names is None:
            # Use self.result column names
            gene_names = self.result.columns.tolist()
        
        # Get PCA components matrix (n_components × n_genes)
        pca_components = self.pca.components_  # Shape: (n_components, n_genes)
        
        # Calculate each gene's importance
        # Method 1: Weighted sum - each gene importance = sum(PCA dimension importance × gene weight in PCA dimension)
        gene_importance = np.dot(dimension_importance, np.abs(pca_components))
        
        # Method 2: Or use squared weighting (emphasize high weight relationships more)
        # gene_importance = np.dot(dimension_importance, pca_components ** 2)
        
        # Create result DataFrame
        gene_importance_df = pd.DataFrame({
            'gene': gene_names,
            'importance': gene_importance
        })
        
        # Sort by importance
        gene_importance_df = gene_importance_df.sort_values('importance', ascending=False)
        
        return gene_importance, gene_importance_df