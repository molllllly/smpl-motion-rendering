# import pandas as pd
#
# # 1. 读取 pkl 文件
# #df = pd.read_pickle("/Users/liya/Downloads/courtyard_dancing_00.pkl")
#
#
# import pickle
#
# with open("/Users/liya/Downloads/courtyard_dancing_00.pkl", "rb") as f:
#     data = pickle.load(f, encoding="latin1")
#
# print(type(data))
# print(data.keys())
#
#
# # 2. 存成 xlsx
# #df.to_excel("/Users/liya/Downloads/courtyard_dancing_00.xlsx", index=False)


import zlib
from pathlib import Path

src = Path("/Users/liya/Downloads/courtyard_dancing_00.pkl")
dst = Path("/Users/liya/Downloads/courtyard_dancing_00.raw")  # 解压后的原始内容

raw = src.read_bytes()
print("raw size:", len(raw), "bytes")

decompressed = zlib.decompress(raw)
print("decompressed size:", len(decompressed), "bytes")

dst.write_bytes(decompressed)
print("written to:", dst)
