# Data Directory

This directory contains sample images for the facial recognition system.

## Directory Structure

- `enrollment/` - Images for enrolling new users
- `query/` - Images for authentication and identification
- `test/` - Images for testing and comparison

## Image Requirements

- Supported formats: JPG, JPEG, PNG
- Recommended size: At least 160x160 pixels
- Face should be clearly visible and well-lit
- Avoid extreme angles or occlusions

## Sample Usage

1. Place enrollment images in `enrollment/` directory
2. Use images from `query/` for authentication
3. Test with images in `test/` directory

## Example Commands

```bash
# Enroll a user
python main.py --mode enroll --user-id john_doe --image data/enrollment/john_face.jpg

# Authenticate a user
python main.py --mode authenticate --image data/query/query_face.jpg

# Compare two faces
python main.py --mode compare --image data/test/face1.jpg --image2 data/test/face2.jpg
```

## Privacy Note

This system is designed for privacy protection. All face embeddings are encrypted with AES-256 before storage, and no raw images are permanently stored in the database.
