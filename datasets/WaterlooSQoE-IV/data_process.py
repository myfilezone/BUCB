import pandas as pd
import os

df = pd.read_csv('data.csv')

total_rebuffering = []
avg_bitrate = []
total_chunk_size = []
avg_vmaf = []

for filename in df['streaming_log']:
    log_path = os.path.join('streaming_logs', filename)
    log_df = pd.read_csv(log_path)

    total_rebuffering.append(log_df['rebuffering_duration'].sum())
    avg_bitrate.append(log_df['video_bitrate'].mean())
    total_chunk_size.append(log_df['chunk_size'].sum())
    avg_vmaf.append(log_df['vmaf'].mean())

df['total_rebuffering_duration'] = total_rebuffering
df['avg_video_bitrate'] = avg_bitrate
df['total_chunk_size'] = total_chunk_size
df['avg_vmaf'] = avg_vmaf

df.to_csv('data_sum.csv', index=False)
