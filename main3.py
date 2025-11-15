import numpy as np
from PIL import Image
from scipy import ndimage as ndi

def load_and_resize_grayscale(path, max_long_side=1024):
    im = Image.open(path).convert("L")
    im = im.copy()
    im.thumbnail((max_long_side, max_long_side), Image.Resampling.LANCZOS)
    return im

def conv2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode='reflect')
    k = np.flipud(np.fliplr(kernel)).astype(np.float32)
    out = np.zeros_like(image, dtype=np.float32)
    H, W = image.shape
    for i in range(H):
        for j in range(W):
            region = padded[i:i+kh, j:j+kw]
            out[i, j] = np.sum(region * k)
    return out



def crop_single_object_on_plain_bg(in_path, out_path, edge_quantile=0.75, dilate_iter=3, close_iter=2, margin_ratio=0.02):
    # 1) Charger en niveaux de gris
    img = Image.open(in_path).convert("L")
    I = np.array(img, dtype=np.float32)

    # 2) Sobel: magnitude des gradients
    Gx = ndi.sobel(I, axis=1, mode="reflect")
    Gy = ndi.sobel(I, axis=0, mode="reflect")
    mag = np.hypot(Gx, Gy)
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)

    # 3) Seuil simple sur les bords (quantile)
    t = np.quantile(mag, edge_quantile)
    edges = mag > t

    # 4) Morphologie: épaissir/fermer/combler
    thick = ndi.binary_dilation(edges, iterations=dilate_iter)
    solid = ndi.binary_closing(thick, iterations=close_iter)
    filled = ndi.binary_fill_holes(solid)

    # 5) Composantes connectées et plus grande région
    labels, n = ndi.label(filled)
    if n == 0:
        raise RuntimeError("Aucun objet détecté")

    # Trouver l'étiquette la plus grande par aire
    slices = ndi.find_objects(labels)
    areas = []
    for idx, slc in enumerate(slices, start=1):
        if slc is None:
            areas.append(0)
            continue
        h = slc[0].stop - slc[0].start
        w = slc[1].stop - slc[1].start
        areas.append(h * w)
    best_label = 1 + int(np.argmax(areas))
    slc = slices[best_label - 1]

    # 6) Rogner avec marge
    H, W = I.shape
    margin = int(margin_ratio * max(H, W))
    y0 = max(slc[0].start - margin, 0)
    y1 = min(slc[0].stop + margin, H)
    x0 = max(slc[1].start - margin, 0)
    x1 = min(slc[1].stop + margin, W)

    cropped = img.crop((x0, y0, x1, y1))
    cropped.save(out_path)



# Noyaux de Sobel (3x3)
Kx = np.array([[1, 0, -1],
               [2, 0, -2],
               [1, 0, -1]], dtype=np.float32)
Ky = np.array([[1,  2,  1],
               [0,  0,  0],
               [-1, -2, -1]], dtype=np.float32)

# 1) Charger et redimensionner
i = "IMG_20251019_180316.jpg"
im = load_and_resize_grayscale(i)
img = load_and_resize_grayscale(i, max_long_side=1024)

# 2) Convolutions Sobel
I = np.array(img, dtype=np.float32)
Gx = conv2d(I, Kx)
Gy = conv2d(I, Ky)

# 3) Magnitude du gradient et sauvegarde
mag = np.hypot(Gx, Gy)  # sqrt(Gx**2 + Gy**2)
mag = (mag / (mag.max() + 1e-8)) * 255.0
Image.fromarray(mag.astype(np.uint8)).save(f"output/{i}-edges_sobel.png")

# Exemple d'utilisation
crop_single_object_on_plain_bg(i, "cropped.png")