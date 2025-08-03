import csv
import json

def create_dummy_json(csv_path, json_path):
    videos = []
    sentences = []
    video_ids = set()
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            video_id = row['video_id']
            if video_id not in video_ids:
                videos.append({'video_id': video_id})
                video_ids.add(video_id)
            sentences.append({
                'caption': row['sentence'],
                'id': i,
                'video_id': video_id
            })
    with open(json_path, 'w') as f:
        json.dump({'videos': videos, 'sentences': sentences}, f, indent=2)

create_dummy_json('datasets/MSRVTT-v2/MSRVTT_train.csv', 'datasets/MSRVTT-v2/MSRVTT_train.json')
create_dummy_json('datasets/MSRVTT-v2/MSRVTT_test.csv', 'datasets/MSRVTT-v2/MSRVTT_val.json')
