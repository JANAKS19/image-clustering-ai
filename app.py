import os
import shutil
import numpy as np
from PIL import Image
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

app = FastAPI()
templates = Jinja2Templates(directory="templates")


# --------------------------------
# Helper Functions
# --------------------------------

def get_image_paths(folder):
    image_extensions = (".jpg", ".jpeg", ".png")
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(image_extensions)
    ]


def is_white_background(image_np):

    h, w, _ = image_np.shape
    border = int(min(h, w) * 0.07)   # smaller border (7%)

    # Extract border
    bg = np.concatenate([
        image_np[:border,:,:].reshape(-1,3),
        image_np[-border:,:,:].reshape(-1,3),
        image_np[:, :border,:].reshape(-1,3),
        image_np[:, -border:,:].reshape(-1,3),
    ]).astype(np.float32)

    # Convert to grayscale brightness
    gray = 0.299 * bg[:,0] + 0.587 * bg[:,1] + 0.114 * bg[:,2]

    # Take only bright pixels (ignore object leak)
    bright_pixels = gray[gray > 150]

    if len(bright_pixels) < len(gray) * 0.4:
        return False

    mean_brightness = np.mean(bright_pixels)
    std_brightness = np.std(bright_pixels)

    # Color neutrality check
    color_diff = np.mean(
        np.abs(bg[:,0] - bg[:,1]) +
        np.abs(bg[:,1] - bg[:,2]) +
        np.abs(bg[:,0] - bg[:,2])
    )

    if (
        mean_brightness > 190 and
        std_brightness < 40 and
        color_diff < 40
    ):
        return True

    return False






# --------------------------------
# Main Logic
# --------------------------------

def cluster_images(input_folder):

    if not os.path.exists(input_folder):
        return "❌ Folder does not exist."

    image_paths = get_image_paths(input_folder)

    if len(image_paths) == 0:
        return "❌ No images found."

    output_base = os.path.join(input_folder, "output_clusters")

    if os.path.exists(output_base):
        shutil.rmtree(output_base)

    white_folder = os.path.join(output_base, "white_background")
    creative_folder = os.path.join(output_base, "creative_background")

    os.makedirs(white_folder, exist_ok=True)
    os.makedirs(creative_folder, exist_ok=True)

    for path in image_paths:
        try:
            pil_image = Image.open(path).convert("RGB")
            image_np = np.array(pil_image)

            if is_white_background(image_np):
                shutil.copy(path, white_folder)
            else:
                shutil.copy(path, creative_folder)

        except Exception as e:
            print(f"Error processing {path}: {e}")

    return f"✅ Done.\nOutput saved to: {output_base}"


# --------------------------------
# Routes
# --------------------------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/process", response_class=HTMLResponse)
def process(request: Request, folder_path: str = Form(...)):

    # Clean user input
    folder_path = folder_path.strip().strip('"').strip("'")

    # Normalize path (handles \ and / automatically)
    folder_path = os.path.normpath(folder_path)

    # Convert to absolute path
    folder_path = os.path.abspath(folder_path)

    result = cluster_images(folder_path)

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": result}
    )
