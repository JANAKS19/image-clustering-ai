import os
import shutil
import torch
import open_clip
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler



# -------- CONFIG --------
INPUT_FOLDER = "input_images"
OUTPUT_FOLDER = "output_clusters"
NUM_CLUSTERS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ------------------------

# Load OpenCLIP model
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k'
)
model = model.to(DEVICE)
model.eval()

def get_image_paths(folder):
    image_extensions = (".jpg", ".jpeg", ".png")
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(image_extensions)
    ]

def extract_embeddings(image_paths):
    embeddings = []

    for path in image_paths:
        try:
            pil_image = Image.open(path).convert("RGB")
            image_np = np.array(pil_image)

            # ---- CLIP embedding ----
            image_clip = preprocess(pil_image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                clip_embedding = model.encode_image(image_clip)
                clip_embedding = clip_embedding / clip_embedding.norm(dim=-1, keepdim=True)

            clip_embedding = clip_embedding.cpu().numpy().flatten()

            # ---- Background brightness feature ----
            h, w, _ = image_np.shape
            border = 50  # pixels

            background_region = np.concatenate([
                image_np[:border,:,:].reshape(-1,3),
                image_np[-border:,:,:].reshape(-1,3),
                image_np[:, :border,:].reshape(-1,3),
                image_np[:, -border:,:].reshape(-1,3),
            ])

            mean_color = np.mean(background_region, axis=0)
            brightness = np.mean(mean_color)
            color_variance = np.var(background_region)

            extra_features = np.array([brightness, color_variance])

            # ---- Combine features ----
            final_embedding = np.concatenate([clip_embedding, extra_features])

            embeddings.append(final_embedding)

        except Exception as e:
            print(f"Error processing {path}: {e}")

    return np.array(embeddings)

def cluster_images(embeddings, image_paths):

    dbscan = DBSCAN(eps=0.35, min_samples=5, metric='cosine')
    labels = dbscan.fit_predict(embeddings)

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    for idx, label in enumerate(labels):

        if label == -1:
            cluster_folder = os.path.join(OUTPUT_FOLDER, "noise")
        else:
            cluster_folder = os.path.join(OUTPUT_FOLDER, f"cluster_{label}")

        os.makedirs(cluster_folder, exist_ok=True)
        shutil.copy(image_paths[idx], cluster_folder)

    print("Clustering complete!")


def main():
    image_paths = get_image_paths(INPUT_FOLDER)
    print(f"Found {len(image_paths)} images")

    embeddings = extract_embeddings(image_paths)
    cluster_images(embeddings, image_paths)

if __name__ == "__main__":
    main()
