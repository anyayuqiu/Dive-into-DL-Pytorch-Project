import os


for key, value in os.environ.items():
    print(f"{key}: {value}")

# torch_model_zoo = os.environ.get('TORCH_MODEL_ZOO')
# print(f"TORCH_MODEL_ZOO: {torch_model_zoo}")
#
# # 方法2：检查是否存在
# if 'TORCH_MODEL_ZOO' in os.environ:
#     print(f"TORCH_MODEL_ZOO路径: {os.environ['TORCH_MODEL_ZOO']}")
# else:
#     print("TORCH_MODEL_ZOO环境变量未设置")