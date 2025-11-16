import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from sklearn.cluster import KMeans


def auto_rotate(img_np: np.ndarray) -> np.ndarray:
    # Binarisation adaptative simple
    thresh = np.percentile(img_np, 50)
    mask = img_np < thresh

    pts = np.column_stack(np.nonzero(mask))
    if pts.shape[0] < 10:
        return img_np

    # PCA pour l'angle principal
    pts_mean = pts.mean(axis=0)
    cov = np.cov((pts - pts_mean).T)
    eigvals, eigvecs = np.linalg.eig(cov)
    main_axis = eigvecs[:, np.argmax(eigvals)]

    angle = np.degrees(np.arctan2(main_axis[0], main_axis[1]))

    # Rotation pour mettre horizontal
    return ndi.rotate(img_np, angle, reshape=True)


def remove_metal_legs(img_np: np.ndarray):
    # Projection verticale
    projection = img_np.mean(axis=1)

    # Le corps beige est une longue zone "flat"
    diff = np.diff(projection)

    # On estime les zones stables (faible variation)
    stable = np.abs(diff) < (0.1 * np.std(diff))
    stable = ndi.binary_closing(stable, iterations=10)

    idx = np.where(stable)[0]
    y0, y1 = idx[0], idx[-1]

    return img_np[y0:y1, :]


def detect_color_bands(img_np: np.ndarray, n_clusters=6):
    h, w = img_np.shape
    X = img_np.reshape(-1, 1)  # Kmeans sur valeurs de gris

    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    kmeans.fit(X)
    labels = kmeans.labels_.reshape(h, w)
    centers = kmeans.cluster_centers_.flatten()

    # Trier les clusters du plus foncé au plus clair
    order = np.argsort(centers)
    labels_sorted = np.zeros_like(labels)
    for i, o in enumerate(order):
        labels_sorted[labels == o] = i

    # Profil horizontal moyen
    profile = labels_sorted.mean(axis=0)

    # Détection des transitions = changements de couleur => bandes
    transitions = np.diff(profile)
    bands_positions = np.where(np.abs(transitions) > 0.1)[0]

    return bands_positions


def preprocess_resistor(path):
    img = Image.open(path).convert("L")
    img_np = np.array(img, float)

    # 1) Rotation automatique
    rotated = auto_rotate(img_np)

    # 2) Crop horizontal (suppression pattes)
    body = remove_metal_legs(rotated)

    return body
