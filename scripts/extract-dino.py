from transformers import ViTImageProcessor, ViTModel
from PIL import Image
from sklearn.decomposition import PCA
import os
import numpy as np
import torch
import torchvision.transforms as T
from math import isqrt

def load_img(path, size=224):
    image = Image.open(path).convert("RGB")
    x, y = image.size
    print(f"loaded input image of size ({x}, {y}) from {path}")
    h = w = size
    image = T.CenterCrop(min(x,y))(image)
    image = image.resize((w, h))
    return image

def rgb_pca(hidden_states):
    dino_full_features = hidden_states.cpu().numpy()
    pca = PCA(n_components=3)
    pca.fit_transform(dino_full_features)
    dino_pca_features = pca.transform(dino_full_features)
    dino_pca_features = dino_pca_features.reshape(-1, 3) 
    pca_img = dino_pca_features
    h = w = isqrt(pca_img.shape[0])
    pca_img = pca_img.reshape(h, w, 3)
    pca_img_min = pca_img.min(axis=(0, 1))
    pca_img_max = pca_img.max(axis=(0, 1))
    pca_img = (pca_img - pca_img_min) / (pca_img_max - pca_img_min)
    return pca_img
    # pca_img = Image.fromarray((pca_img * 255).astype(np.uint8))
    # pca_img = T.Resize(512, interpolation=T.InterpolationMode.NEAREST)(pca_img)
    # pca_img.save(os.path.join(save_dir, f"{experiment}_time_{t}.png"))

def feature_pca(hidden_states, n_components=64):
    dino_full_features = hidden_states.cpu().numpy()
    pca = PCA(n_components=n_components)
    dino_pca_features = pca.fit_transform(dino_full_features)
    return dino_pca_features

BERKELEY_TRAIN_PATH = "../data/BSDS300/images/train"
BERKELEY_TEST_PATH = "../data/BSDS300/images/test"
BSBD_PATH = "../data/BSDS300"
BERKELEY_TRAIN_IMG_PATHS = [os.path.join(BERKELEY_TRAIN_PATH, f) for f in os.listdir(BERKELEY_TRAIN_PATH) if os.path.isfile(os.path.join(BERKELEY_TRAIN_PATH, f))]
BERKELEY_TRAIN_IMG_NAMES = [os.path.splitext(os.path.basename(path))[0] for path in BERKELEY_TRAIN_IMG_PATHS]
BERKELEY_TEST_IMG_PATHS = [os.path.join(BERKELEY_TEST_PATH, f) for f in os.listdir(BERKELEY_TEST_PATH) if os.path.isfile(os.path.join(BERKELEY_TEST_PATH, f))]
BERKELEY_TEST_IMG_NAMES = [os.path.splitext(os.path.basename(path))[0] for path in BERKELEY_TEST_IMG_PATHS]
print(BERKELEY_TRAIN_IMG_NAMES)

processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb8')
model = ViTModel.from_pretrained('facebook/dino-vitb8')

# PCA_SIZES = [16, 32, 64, 128, 256]
PCA_SIZES = [32, 64, 128]
N_TILES = 2

# os.makedirs(os.path.join(BSBD_PATH, "dino_train"), exist_ok=True) 
os.makedirs(os.path.join(BSBD_PATH, f"dino_train_tile_{N_TILES}", "rgb_ref"), exist_ok=True) 
os.makedirs(os.path.join(BSBD_PATH, f"dino_train_tile_{N_TILES}", "pca_vis"), exist_ok=True) 
os.makedirs(os.path.join(BSBD_PATH, f"dino_train_tile_{N_TILES}", "raw"), exist_ok=True) 
for pca_size in PCA_SIZES:
    os.makedirs(os.path.join(BSBD_PATH, f"dino_train_tile_{N_TILES}", f"pca_{pca_size}"), exist_ok=True) 

# Read all images (TRAIN)
for i_img, img_path in enumerate(BERKELEY_TRAIN_IMG_PATHS):
    image = load_img(img_path, 224*N_TILES)
    image.save(os.path.join(BSBD_PATH, f"dino_train_tile_{N_TILES}", "rgb_ref", BERKELEY_TRAIN_IMG_NAMES[i_img] + ".jpg"))
    image = T.functional.pil_to_tensor(image)

    with torch.no_grad():

        dino_features = torch.zeros((N_TILES*28, N_TILES*28, 768))
        for i_tile in range(N_TILES):
            for j_tile in range(N_TILES):
                inputs = processor(images=image[:, i_tile*224:(i_tile+1)*224, j_tile*224:(j_tile+1)*224], return_tensors="pt")
                outputs = model(**inputs)
                last_hidden_states = outputs.last_hidden_state
                last_hidden_states = last_hidden_states.squeeze(0)[:-1, :]
                last_hidden_states = last_hidden_states.reshape((28, 28, 768))
                dino_features[i_tile*28:(i_tile+1)*28, j_tile*28:(j_tile+1)*28, :] = last_hidden_states
        # print(last_hidden_states.shape)
        dino_features = dino_features.reshape((-1, 768))

        # Save pca vis
        pca_rgb_img = rgb_pca(dino_features)
        pca_img = Image.fromarray((pca_rgb_img * 255).astype(np.uint8))
        pca_img = T.Resize(224*N_TILES)(pca_img)
        pca_img.save(os.path.join(BSBD_PATH, f"dino_train_tile_{N_TILES}", "pca_vis", BERKELEY_TRAIN_IMG_NAMES[i_img] + ".jpg"))

        # Save raw features
        # np.save(os.path.join(BSBD_PATH, f"dino_train_tile_{N_TILES}", "raw", BERKELEY_TRAIN_IMG_NAMES[i_img]), last_hidden_states.cpu().numpy())

        # Save pca features
        for pca_size in PCA_SIZES:
            pca_features = feature_pca(dino_features, n_components=pca_size)
            np.save(os.path.join(BSBD_PATH, f"dino_train_tile_{N_TILES}", f"pca_{pca_size}", BERKELEY_TRAIN_IMG_NAMES[i_img]), pca_features)

os.makedirs(os.path.join(BSBD_PATH, f"dino_test_tile_{N_TILES}", "rgb_ref"), exist_ok=True) 
os.makedirs(os.path.join(BSBD_PATH, f"dino_test_tile_{N_TILES}", "pca_vis"), exist_ok=True) 
os.makedirs(os.path.join(BSBD_PATH, f"dino_test_tile_{N_TILES}", "raw"), exist_ok=True) 
for pca_size in PCA_SIZES:
    os.makedirs(os.path.join(BSBD_PATH, f"dino_test_tile_{N_TILES}", f"pca_{pca_size}"), exist_ok=True) 

# Read all images (TEST)
for i_img, img_path in enumerate(BERKELEY_TEST_IMG_PATHS):
    image = load_img(img_path, 224*N_TILES)
    image.save(os.path.join(BSBD_PATH, f"dino_test_tile_{N_TILES}", "rgb_ref", BERKELEY_TEST_IMG_NAMES[i_img] + ".jpg"))
    image = T.functional.pil_to_tensor(image)

    with torch.no_grad():
        dino_features = torch.zeros((N_TILES*28, N_TILES*28, 768))
        for i_tile in range(N_TILES):
            for j_tile in range(N_TILES):
                inputs = processor(images=image[:, i_tile*224:(i_tile+1)*224, j_tile*224:(j_tile+1)*224], return_tensors="pt")
                outputs = model(**inputs)
                last_hidden_states = outputs.last_hidden_state
                last_hidden_states = last_hidden_states.squeeze(0)[:-1, :]
                last_hidden_states = last_hidden_states.reshape((28, 28, 768))
                dino_features[i_tile*28:(i_tile+1)*28, j_tile*28:(j_tile+1)*28, :] = last_hidden_states
        # print(last_hidden_states.shape)
        dino_features = dino_features.reshape((-1, 768))

        # Save pca vis
        pca_rgb_img = rgb_pca(dino_features)
        pca_img = Image.fromarray((pca_rgb_img * 255).astype(np.uint8))
        pca_img = T.Resize(224*N_TILES)(pca_img)
        pca_img.save(os.path.join(BSBD_PATH, f"dino_test_tile_{N_TILES}", "pca_vis", BERKELEY_TEST_IMG_NAMES[i_img] + ".jpg"))

        # Save raw features
        # np.save(os.path.join(BSBD_PATH, f"dino_test_tile_{N_TILES}", "raw", BERKELEY_TEST_IMG_NAMES[i_img]), last_hidden_states.cpu().numpy())

        # Save pca features
        for pca_size in PCA_SIZES:
            pca_features = feature_pca(dino_features, n_components=pca_size)
            np.save(os.path.join(BSBD_PATH, f"dino_test_tile_{N_TILES}", f"pca_{pca_size}", BERKELEY_TEST_IMG_NAMES[i_img]), pca_features)

