# authored by: Mourad Boustani
from PIL import Image
import io
import pyarrow.parquet as pq
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
import openai

openai_key = "redacted"
openai.api_key = openai_key

table = pq.read_table("1.parquet")
df = table.to_pandas()

# parse data.json file for labels
# with open("data.json", "r") as f:
#     data = json.load(f)

# labels = data["names"]
# high_level_labels = []
# for label in labels:
#     completion = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "system", "content": "We need to re-label classes from ImageNet. These are the higher-level we want to use for re-labeling: airplane, bear, bicycle, bird, boat, bottle, car, cat, chair, clock, dog, elephant, keyboard, knife, oven, truck. If it doesn't belong to any, return None. Only return one of the labels or None, absolutely nothing else"},
#             {"role": "user", "content": f"which label does '{label}' belong to? Only respond with one of the labels or None, absolutely nothing else"},
#         ],
#     )

#     print(label, completion.choices[0].message.content)
#     high_level_labels.append(completion.choices[0].message.content)

#     # input_text = f"We need to re-label classes from ImageNet. These are the higher-level we want to use for re-labeling: airplane, bear, bicycle, bird, boat, bottle, car, cat, chair, clock, dog, elephant, keyboard, knife, oven, truck. Where does this label belong: {label}. If it doesn't belong to any, return None"
#     # input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
#     # outputs = model.generate(input_ids)
#     # output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     # print(f'{label}: ', output_text)
#     # high_level_labels.append(output_text)

# print(high_level_labels)

with open("data.json", "r") as f:
    data = json.load(f)

high_level_labels = data["labels"]

high_level_labels = [None if label == "None" else label for label in high_level_labels]

# print image column of first row

# image_bytes = df.iloc[0]['image']['bytes']
# label_index = df.iloc[0]['label']
# label = labels[label_index]
# print(label)


# convert bytes to image
# image = Image.open(io.BytesIO(image_bytes))
# image.show()
