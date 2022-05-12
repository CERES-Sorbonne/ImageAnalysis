from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel

from sklearn.metrics.pairwise import cosine_similarity

model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def vectorize(img):
    input1 = processor(images=img, return_tensors="pt")
    output1 = model(**input1)
    pooled_output1 = output1[1].detach().numpy()  # pooled_output
    return pooled_output1.flatten()


img1 = Image.open("../images/cat1.jpg").convert("RGB")
v1 = vectorize(img1)

img2 = Image.open("../images/cat2.jpg").convert("RGB")
v2 = vectorize(img2)

img3 = Image.open("../images/cat3.jpg").convert("RGB")
v3 = vectorize(img3)

img4 = Image.open("../images/cat4.jpg").convert("RGB")
v4 = vectorize(img4)

img5 = Image.open("../images/pain1.jpg").convert("RGB")
v5 = vectorize(img5)

images = []
images.append(v1)
images.append(v2)
images.append(v3)
images.append(v4)
images.append(v5)


print(cosine_similarity(images))
