import ijson
import json

input_path = '/datasets/MSRVTT-v2/current.json'
output_path = '/datasets/MSRVTT-v2/MSRVTT_data.json'

def transform_large_json(input_path, output_path):
    videos = []
    sentences = []

    with open(input_path, 'r', encoding='utf-8') as f:
        # ijson parses one item at a time from a specific prefix
        images = ijson.items(f, 'images.item')
        for image in images:
            videos.append({'video_id': image['id']})

    with open(input_path, 'r', encoding='utf-8') as f:
        annotations = ijson.items(f, 'annotations.item')
        for annotation in annotations:
            sentences.append({
                'caption': annotation['caption'],
                'id': annotation['id'],
                'video_id': annotation['image_id']
            })

    with open(output_path, 'w', encoding='utf-8') as out_f:
        json.dump({
            'videos': videos,
            'sentences': sentences
        }, out_f, indent=2)

# Run the function
transform_large_json(input_path, output_path)