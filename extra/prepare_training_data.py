"""
Script to prepare training data for the ML models.
Includes data downloading, preprocessing, and organization.
"""
import os
import urllib.request
import zipfile
import tarfile
from pathlib import Path
import shutil

def download_file(url, dest_path):
    """Download file from URL to destination path."""
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, dest_path)
    print(f"Downloaded to {dest_path}")

def extract_archive(archive_path, extract_to):
    """Extract zip or tar archive."""
    print(f"Extracting {archive_path}...")
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")

def prepare_wider_face_dataset(data_dir):
    """Prepare WIDER FACE dataset for MTCNN training."""
    wider_dir = Path(data_dir) / "WIDER_FACE"
    wider_dir.mkdir(parents=True, exist_ok=True)

    # WIDER FACE dataset URLs (you need to download manually or from official source)
    # Note: WIDER FACE requires registration and manual download
    print("WIDER FACE dataset preparation:")
    print("1. Visit: http://shuoyang1213.me/WIDERFACE/")
    print("2. Download WIDER_train.zip, WIDER_val.zip, WIDER_test.zip")
    print("3. Download wider_face_split.zip")
    print("4. Place them in:", wider_dir)
    print("5. Run extraction manually or modify this script")

def prepare_vggface2_dataset(data_dir):
    """Prepare VGGFace2 dataset for FaceNet training."""
    vgg_dir = Path(data_dir) / "VGGFace2"
    vgg_dir.mkdir(parents=True, exist_ok=True)

    # VGGFace2 requires registration and manual download
    print("VGGFace2 dataset preparation:")
    print("1. Visit: https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/")
    print("2. Register and download the dataset")
    print("3. Extract to:", vgg_dir)
    print("Note: VGGFace2 is very large (~8GB)")

def prepare_lfw_dataset(data_dir):
    """Download and prepare LFW dataset for testing/verification."""
    lfw_dir = Path(data_dir) / "lfw"
    lfw_dir.mkdir(parents=True, exist_ok=True)

    # LFW dataset
    lfw_url = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
    archive_path = lfw_dir / "lfw.tgz"

    if not archive_path.exists():
        download_file(lfw_url, str(archive_path))

    extract_dir = lfw_dir / "extracted"
    if not extract_dir.exists():
        extract_archive(str(archive_path), str(lfw_dir))

    print(f"LFW dataset ready at: {lfw_dir}")

def prepare_celeba_dataset(data_dir):
    """Download and prepare CelebA dataset."""
    celeba_dir = Path(data_dir) / "CelebA"
    celeba_dir.mkdir(parents=True, exist_ok=True)

    # CelebA dataset (requires manual download)
    print("CelebA dataset preparation:")
    print("1. Visit: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
    print("2. Download Img/img_align_celeba.zip")
    print("3. Download Anno/list_attr_celeba.txt, etc.")
    print("4. Extract to:", celeba_dir)

def create_synthetic_training_data(data_dir, num_classes=100, images_per_class=10):
    """Create synthetic training data for testing purposes."""
    import cv2
    import numpy as np

    synthetic_dir = Path(data_dir) / "synthetic_faces"
    synthetic_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating synthetic dataset with {num_classes} classes, {images_per_class} images each...")

    for class_id in range(num_classes):
        class_dir = synthetic_dir / f"class_{class_id:03d}"
        class_dir.mkdir(exist_ok=True)

        for img_id in range(images_per_class):
            # Create base face
            img = np.ones((160, 160, 3), dtype=np.uint8) * 200

            # Add some variation
            noise = np.random.randint(-20, 20, (160, 160, 3), dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Add simple face features with variation
            center_x = 80 + np.random.randint(-10, 10)
            center_y = 80 + np.random.randint(-10, 10)

            # Eyes
            cv2.circle(img, (center_x-15, center_y-10), 5, (0, 0, 0), -1)
            cv2.circle(img, (center_x+15, center_y-10), 5, (0, 0, 0), -1)

            # Nose
            cv2.ellipse(img, (center_x, center_y+5), (3, 8), 0, 0, 360, (100, 50, 0), -1)

            # Mouth
            cv2.ellipse(img, (center_x, center_y+20), (8, 4), 0, 0, 360, (0, 0, 0), 2)

            img_path = class_dir / f"face_{img_id:02d}.jpg"
            cv2.imwrite(str(img_path), img)

    print(f"Synthetic dataset created at: {synthetic_dir}")

def main():
    """Main data preparation function."""
    data_dir = Path("training_data")
    data_dir.mkdir(exist_ok=True)

    print("Preparing training data for facial recognition models...")
    print("=" * 60)

    # Prepare datasets
    prepare_lfw_dataset(data_dir)  # Small dataset for testing
    prepare_wider_face_dataset(data_dir)  # For MTCNN
    prepare_vggface2_dataset(data_dir)  # For FaceNet
    prepare_celeba_dataset(data_dir)  # Alternative

    # Create synthetic data for initial testing
    create_synthetic_training_data(data_dir)

    print("\nData preparation completed!")
    print("Next steps:")
    print("1. Download required datasets manually (WIDER FACE, VGGFace2, CelebA)")
    print("2. Run training scripts: train_mtcnn.py, train_facenet.py, train_matching.py")
    print("3. Monitor training progress and adjust hyperparameters as needed")

if __name__ == "__main__":
    main()
