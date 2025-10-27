from core.swc_processor import BatchSWCProcessor
import pyswcloader


# 加载注解数据
anno = pyswcloader.brain.read_nrrd('data/annotation_25.nrrd')
resolution = 25  # 根据实际情况调整

# 创建批量处理器
batch_processor = BatchSWCProcessor(anno, resolution)

# 处理所有数据
root_path = "data/orig_swc_data/test/unzip/"
results = batch_processor.process_batch_folders(root_path)

print(f"处理完成，共得到 {len(results)} 条路径记录")



