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


# 自定义激活函数（根据您的实际定义）
def custom_activation(x):
    """允许小范围负值输出的改进激活函数"""
    return tf.where(x < 0, 
                  1 * tf.nn.softplus(x),  # 对深度负值保持微小正值
                  tf.nn.softplus(x))          # 正常正值响应

class SequenceDataProcessor:

    def __init__(self, stl_acro_dict, gene_filled_result_path):
        """
        初始化序列数据处理器
        Args:
            stl_acro_dict: 脑区缩写字典
            gene_filled_result_path: 基因填充结果文件路径
        """
        self.stl_acro_dict = stl_acro_dict
        self.gene_filled_result_path = gene_filled_result_path
        self.index_mapping = None
        self.gene_embeddings_df = None
        self.pca = None
        self.result = None
        
    def load_and_prepare_data(self, filtered_results_path, window_size=5 ,n_components=64):
        """
        加载和准备数据
        
        Args:
            filtered_results_path: 过滤结果文件路径
            window_size: 滑动窗口大小
            
        Returns:
            tuple: (X, y, max_len, pca)
        """
        
        # 创建索引映射
        self.index_mapping = {v: k for k, v in self.stl_acro_dict.items()}
        
        # 加载过滤结果
        filtered_results_df = pd.read_csv(filtered_results_path)
        filtered_results_df['strength'] = filtered_results_df['strength'].apply(ast.literal_eval)
        
        # 生成 named_seqs（序列值列表）
        named_seqs = [
            row['path'].split('→') 
            for _, row in filtered_results_df.iterrows()
        ]
        
        # 生成 strength_seqs（强度值列表）
        strength_seqs = [
            row['strength'] 
            for _, row in filtered_results_df.iterrows()
        ]
        
        # 准备模型输入
        X, y, idx_to_name = self.prepare_model_input(named_seqs, strength_seqs, self.index_mapping)
        
        # 滑动窗口切割
        new_X, new_y = self.sliding_window_cut(X, y, window_size=window_size)
        
        # 去重和过滤
        X, y = self.deduplicate_and_filter(new_X, new_y)
        
        # 对数变换
        y_log = np.log2(y + 1)
        
        max_len = X.shape[1]
        
        # 加载基因数据并处理
        gene_filled_result = pd.read_csv(self.gene_filled_result_path, index_col=0)
        
        column_means = gene_filled_result.mean()
        std_dev = gene_filled_result.std()
        threshold = std_dev.quantile(0.5)
        selected_columns = std_dev[std_dev >= threshold].index
        top_50_percent_columns = gene_filled_result[selected_columns]
        result = top_50_percent_columns
        #####去掉以rik结尾，GM开头的列
        result = result.loc[:, ~result.columns.str.endswith('Rik') & ~result.columns.str.startswith('Gm')]
        ########先对列，在对行
        result = result.apply(lambda col: (col - col.mean()) / col.std(), axis=0)
        result = result.apply(lambda row: (row - row.mean()) / row.std(), axis=1)
        self.result = result
        
        # PCA降维
        self.pca = PCA(n_components=n_components)
        result_pca = self.pca.fit_transform(result)
        pca_df = pd.DataFrame(result_pca, index=result.index)
        self.gene_embeddings_df = pca_df
        
        return X, y_log, max_len, self.pca
    
    def prepare_model_input(self, named_sequences, strength_sequences, node_to_idx):
        """
        准备模型输入数据
        """
        
        # 转换为索引序列
        X_idx = [[node_to_idx[node] for node in seq] for seq in named_sequences]
        
        # 计算最大长度
        max_len = max(len(seq) for seq in named_sequences)
        
        # 填充节点索引（X）
        X_padded = pad_sequences(
            X_idx,
            maxlen=max_len,
            padding='post',
            value=0,
            dtype='int32'
        )
        
        # 填充强度值（y）保留浮点数
        y_padded = pad_sequences(
            strength_sequences,
            maxlen=max_len,
            padding='post',
            value=0.0,
            dtype='float32'
        )
        
        # 创建索引到名称的映射
        idx_to_name = {idx: name for name, idx in node_to_idx.items()}
        
        return X_padded, y_padded, idx_to_name
    
    def sliding_window_cut(self, X, y, window_size=3):
        """
        滑动窗口切割
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
        去重和过滤
        """
        
        assert new_X.ndim == 2, "new_X 必须是二维数组"
        assert new_y.ndim == 2, "new_y 必须是二维数组"
        assert new_X.shape[0] == new_y.shape[0], "X和y的行数必须相同"
        assert new_X.shape[1] == new_y.shape[1], "X和y的列数必须相同"
        
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
        准备基因序列
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
        分割数据
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
        准备最终数据
        """
        
        # 准备基因序列
        gene_train = self.prepare_gene_sequences(node_train_processed, max_len)
        gene_test = self.prepare_gene_sequences(node_test_processed, max_len)
        
        # 准备初始强度
        init_strength_train = strength_train_processed[:, 0:1]
        init_strength_test = strength_test_processed[:, 0:1]
        
        # 准备移位强度
        strength_train_shift = np.array([np.roll(row, -1)[:-1] for row in strength_train_processed])
        strength_test_shift = np.array([np.roll(row, -1)[:-1] for row in strength_test_processed])
        
        return (gene_train, gene_test, init_strength_train, init_strength_test,
                strength_train_shift, strength_test_shift)
    
    def create_adaptive_weighted_loss(self, seq_length, weights=None):
        """
        创建自适应加权损失函数 - 支持任意时间步长
        
        Args:
            seq_length (int): 时间步长数量
            weights (list): 可选，自定义权重列表，长度必须等于seq_length
        
        Returns:
            function: 配置好的损失函数
        """
        # 如果未提供权重，使用均匀权重
        if weights is None:
            weights = [1.0 / seq_length] * seq_length
        else:
            # 验证权重长度匹配
            if len(weights) != seq_length:
                raise ValueError(f"权重列表长度({len(weights)})必须等于时间步长({seq_length})")
        
        # 将权重转换为Tensor常量
        time_weights_tensor = tf.constant(weights, dtype=tf.float32)
        
        def adaptive_weighted_loss(y_true, y_pred):
            """
            自适应加权损失函数
            
            参数:
                y_true: 真实值，形状为(batch_size, seq_length)
                y_pred: 预测值，形状为(batch_size, seq_length)
            
            返回:
                加权损失值
            """
            batch_size = tf.shape(y_true)[0]
            
            # 验证输入时间步长
            input_seq_length = tf.shape(y_true)[1]
            tf.debugging.assert_equal(
                input_seq_length, 
                seq_length,
                message=f"此损失函数仅支持{seq_length}个时间步的序列"
            )
            
            # 计算基础MAE损失
            mae_term = tf.abs(y_true - y_pred)
            
            # 扩展权重维度以便广播到整个batch
            time_weights = tf.expand_dims(time_weights_tensor, axis=0)  # 形状变为 (1, seq_length)
            time_weights = tf.tile(time_weights, [batch_size, 1])  # 形状变为 (batch_size, seq_length)
            
            # 应用时间步权重
            weighted_mae = mae_term * time_weights
            
            # 计算加权损失
            total_weight = tf.reduce_sum(time_weights)
            loss = tf.reduce_sum(weighted_mae) / total_weight
            
            return loss
        
        return adaptive_weighted_loss

    def get_pca(self):
        """
        获取PCA对象用于特征分析
        
        Returns:
            PCA: 训练好的PCA对象
        """
        return self.pca

    def build_true_autoregressive_model_with_k(self, max_len, gene_embed_dim=64):
        """构建带可学习误差系数的自回归模型"""
        # 使用max_len-1创建损失函数
        loss_function = self.create_adaptive_weighted_loss(max_len - 1)
        
        # 输入层
        gene_embed_input = tf.keras.Input(shape=(max_len, gene_embed_dim), name='gene_embed_input')
        init_strength_input = tf.keras.Input(shape=(1,), name='init_strength_input')
        
        # ====== 增强的基因嵌入处理路径（使用PReLU）======
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
        
        # 提取第一个时间步的处理后基因嵌入信息
        first_gene_embed = tf.keras.layers.Lambda(lambda x: x[:, 0, :])(processed_gene_embed)
        second_and_third = tf.keras.layers.Lambda(lambda x: x[:, 1:, :])(processed_gene_embed)
        
        # 拼接初始强度和第一个基因嵌入
        combined_init_input = tf.keras.layers.Concatenate(axis=-1)([init_strength_input, first_gene_embed])
        
        # 创建自回归RNN层
        autoregressive_cell = self.AutoregressiveCell(32)
        rnn_layer = tf.keras.layers.RNN(
            autoregressive_cell,
            return_sequences=True,
            return_state=False,
            unroll=False
        )
        
        # 初始化状态 - 使用拼接后的信息
        h0 = tf.keras.layers.Dense(32)(combined_init_input)
        # h0 = tf.keras.layers.PReLU()(h0)
        c0 = tf.keras.layers.Dense(32)(combined_init_input)
        # c0 = tf.keras.layers.PReLU()(c0)
        initial_state = [h0, c0]
        
        # 运行自回归RNN（使用处理后的基因嵌入）
        output = rnn_layer(
            second_and_third,
            initial_state=initial_state
        )
        
        # 输出处理 - 使用Lambda层包装TensorFlow操作
        output = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=-1))(output)  # (batch_size, max_len)
        
        # ====== 添加可学习的误差系数k ======
        # 创建独立的可学习系数k（初始值为1.0）
        ones_vector = tf.keras.layers.Lambda(lambda x: tf.ones_like(x[:, :1]))(output)
        k = tf.keras.layers.Dense(
            1, 
            activation=None, 
            use_bias=False,
            kernel_initializer='ones',  # 初始化为1.0
            name='error_coefficient'
        )(ones_vector)  # 使用与output相同batch大小的单位向量
        
        # 确保k是标量（但保持与batch匹配）
        k = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=-1))(k)  # 现在形状为 (batch_size,)
        
        # 将k扩展为与output相同的形状
        k_expanded = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(k)  # (batch_size, 1)
        
        # 获取output的形状信息用于tile操作
        output_shape = tf.keras.backend.int_shape(output)
        k_expanded = tf.keras.layers.Lambda(
            lambda x: tf.tile(x[0], [1, output_shape[1]])
        )([k_expanded])  # (batch_size, max_len)
        
        # 应用误差系数：最终预测 = 模型输出 * k
        final_output = tf.keras.layers.Multiply()([output, k_expanded])
        
        # 构建模型
        model = tf.keras.Model(
            inputs=[gene_embed_input, init_strength_input],
            outputs=final_output
        )
        
        # 编译模型 - 使用动态创建的损失函数
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=loss_function,  # 使用动态创建的损失函数
            metrics=['mae', tf.keras.metrics.MeanAbsolutePercentageError()]
        )
        
        return model

    class MultiInputR2ScoreCallback(Callback):
        """支持多输入模型的R²回调函数"""
        def __init__(self, validation_data):
            super().__init__()
            self.X_val_list, self.y_val = validation_data
            self.X_val_list = [np.array(x) for x in self.X_val_list]  # 确保所有输入都是NumPy数组
        
        def on_epoch_end(self, epoch, logs=None):
            # 获取模型期望的输入形状 - 修复形状获取方式
            expected_shapes = []
            for input_tensor in self.model.inputs:
                # 处理不同类型的输入形状表示
                if hasattr(input_tensor, 'shape') and hasattr(input_tensor.shape, 'as_list'):
                    expected_shapes.append(input_tensor.shape.as_list())
                elif hasattr(input_tensor, 'shape'):
                    # 如果shape是元组或其他类型，直接使用
                    expected_shapes.append(input_tensor.shape)
                else:
                    # 如果无法获取形状，使用None作为占位符
                    expected_shapes.append(None)
            
            # 调整每个输入的形状以匹配模型期望
            adjusted_X_val = []
            for i, (input_data, expected_shape) in enumerate(zip(self.X_val_list, expected_shapes)):
                if expected_shape is None:
                    # 如果无法获取期望形状，直接使用原始数据
                    adjusted_data = input_data
                elif len(expected_shape) == 3 and len(input_data.shape) == 3:
                    expected_timesteps = expected_shape[1]
                    # 截取前N个时间步
                    if input_data.shape[1] > expected_timesteps:
                        adjusted_data = input_data[:, :expected_timesteps, :]
                        print(f"调整输入 {i} 的形状: {input_data.shape} -> {adjusted_data.shape}")
                    else:
                        adjusted_data = input_data
                else:
                    adjusted_data = input_data
                adjusted_X_val.append(adjusted_data)
            
            # 使用调整后的输入进行预测
            y_pred = self.model.predict(adjusted_X_val, verbose=0)
            
            # 计算R²分数
            r2_scores = r2_score(self.y_val, y_pred, multioutput='raw_values')
            avg_r2 = np.mean(r2_scores)
            
            # 打印结果
            print(f"\nEpoch {epoch+1} Validation R² Scores:")
            for i, score in enumerate(r2_scores):
                print(f"  Output {i+1}: {score:.4f}")
            print(f"  Average R²: {avg_r2:.4f}")
            
            # 记录到日志
            logs = logs or {}
            logs['val_r2'] = avg_r2
            for i, score in enumerate(r2_scores):
                logs[f'val_r2_output_{i+1}'] = score

    class AutoregressiveCell(tf.keras.layers.Layer):
        """自定义自回归单元"""
        def __init__(self, units, **kwargs):
            super().__init__(**kwargs)
            self.units = units
            self.state_size = [units, units]  # [h_state, c_state]
            self.output_size = 1  # 预测强度值
            
        def build(self, input_shape):
            # 输入形状: (batch_size, gene_embed_dim)
            self.lstm_cell = tf.keras.layers.LSTMCell(self.units)
            self.output_dense = tf.keras.layers.Dense(1, activation=custom_activation)
            self.built = True
            
        @tf.autograph.experimental.do_not_convert
        def call(self, inputs, states, training=None, **kwargs):
            # 解包状态
            h_state, c_state = states
            
            # 直接使用基因嵌入作为输入
            lstm_input = inputs  # 形状: (batch_size, gene_embed_dim)
            
            # LSTM处理
            lstm_output, [new_h, new_c] = self.lstm_cell(
                lstm_input, 
                [h_state, c_state],
                training=training
            )
            
            # 预测当前强度
            strength_pred = self.output_dense(lstm_output)
            
            return strength_pred, [new_h, new_c]

    def compute_gene_importance(self, model, dataset, target_timestep=-1, n_samples=100):
        """
        计算基因嵌入的重要性分数
        
        参数:
            model: 训练好的自回归模型
            dataset: 输入数据集 (gene_embed, init_strength)
            target_timestep: 目标时间步（默认最后一个）
            n_samples: 用于计算的样本数量
            
        返回:
            position_importance: 每个位置的平均重要性 [max_len]
            dimension_importance: 每个嵌入维度的平均重要性 [embed_dim]
        """
        # 初始化重要性矩阵
        position_importance = np.zeros(model.input_shape[0][1])  # max_len
        dimension_importance = np.zeros(model.input_shape[0][2])  # embed_dim
        
        # 获取样本
        gene_embeds, init_strengths = dataset
        sample_indices = np.random.choice(len(gene_embeds), n_samples, replace=False)
        
        # 添加进度条
        for idx in tqdm(sample_indices, desc="计算基因重要性", unit="样本"):
            gene_embed = tf.convert_to_tensor(gene_embeds[idx][np.newaxis])
            init_strength = tf.convert_to_tensor(init_strengths[idx][np.newaxis])
            
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(gene_embed)
                predictions = model([gene_embed, init_strength])
                
                # 选择目标输出（特定时间步）
                target = predictions[:, target_timestep] if target_timestep >= 0 else predictions
            
            # 计算梯度
            gradients = tape.gradient(target, gene_embed)
            
            # 计算位置重要性（L2范数）
            per_position = tf.norm(gradients, axis=-1).numpy().squeeze()
            position_importance += per_position
            
            # 计算维度重要性（绝对值平均）
            per_dimension = tf.reduce_mean(tf.abs(gradients), axis=[0,1]).numpy()
            dimension_importance += per_dimension
        
        # 平均重要性
        position_importance /= n_samples
        dimension_importance /= n_samples
        
        return position_importance, dimension_importance

    def get_gene_importance_from_pca(self, dimension_importance, gene_names=None):
        """
        从PCA维度重要性计算原始基因的重要性
        
        参数:
            dimension_importance: PCA维度重要性分数 [n_components]
            gene_names: 原始基因名称列表 [n_genes]，如果为None则使用self.result的列名
            
        返回:
            gene_importance: 原始基因重要性分数 [n_genes]
            gene_importance_df: 包含基因名称和重要性分数的DataFrame
        """
        if gene_names is None:
            # 使用self.result的列名
            gene_names = self.result.columns.tolist()
        
        # 获取PCA的components_矩阵 (n_components × n_genes)
        pca_components = self.pca.components_  # 形状: (n_components, n_genes)
        
        # 计算每个基因的重要性
        # 方法1: 加权求和 - 每个基因的重要性 = sum(PCA维度重要性 × 该基因在PCA维度中的权重)
        gene_importance = np.dot(dimension_importance, np.abs(pca_components))
        
        # 方法2: 或者使用平方加权（更强调高权重的关系）
        # gene_importance = np.dot(dimension_importance, pca_components ** 2)
        
        # 创建结果DataFrame
        gene_importance_df = pd.DataFrame({
            'gene': gene_names,
            'importance': gene_importance
        })
        
        # 按重要性排序
        gene_importance_df = gene_importance_df.sort_values('importance', ascending=False)
        
        return gene_importance, gene_importance_df
