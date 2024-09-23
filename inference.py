from transformers import AutoImageProcessor, ResNetForImageClassification
from datasets import load_dataset
import torch
from torchvision import transforms
from tqdm import tqdm

image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50", num_labels=10, ignore_mismatched_sizes=True)

dataset = load_dataset("mnist", split="test")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(3),
    transforms.ToTensor(),
])

number = 0
for i in tqdm(range(len(dataset))):
    image = transform(dataset["image"][i])
    label = dataset["label"][i]
    input = image_processor(image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**input).logits
    predicted_label = logits.argmax(-1).item()
    number += 1 if (predicted_label == label) else 0
accuracy = number / len(dataset)
print(f"accuracy: {accuracy}")