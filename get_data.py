import sys
import pyarrow.parquet as pq

path = sys.argv[1]

tb = pq.read_table(path)
df = tb.to_pandas()
v = df.iloc[0]

q = v["prompt"][0]
a = v["solution_text_format"]
print(q)
print(a)