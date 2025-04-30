import json

# Input and output file paths
input_file = 'annotations.txt'
output_file = 'msvd-retrieval_train-val-test.json'

# List to hold the JSON entries

train = []
val = []
test = []

data = {}

# Read and process each line of the input file
count = 1
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue  # Skip empty lines
        # Split only on the first space
        parts = line.split(' ', 1)
        if len(parts) != 2:
            print("Does not contain 2 parts")
            continue  # Skip malformed lines
        video_id, caption = parts

        captions = data.get(video_id, [])
        captions.append(caption)
        data[video_id] = captions

    print("Total :", len(data.keys()))

    for video_id, captions in data.items():
        if count > 10:
            break
        for caption in captions:
            if count <= 5:
                train.append({
                    "video": video_id,
                    "caption": caption
                })
            elif count <= 7:
                val.append({
                    "video": video_id,
                    "caption": caption
                })
            else:
                test.append({
                    "video": video_id,
                    "caption": caption
                })

        count = count + 1

# Create the final JSON structure
output_data = {
    "train": train,
    "val": val,
    "test": test
}

# Write to the JSON file
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"Number of train videos: {len(train)}")
print(f"Number of val videos: {len(val)}")
print(f"Number of test videos: {len(train)}")
