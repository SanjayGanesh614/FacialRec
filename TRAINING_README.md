# Training the ML Models for Facial Recognition

This guide explains how to train the three ML models used in the facial recognition system: MTCNN for face detection, FaceNet for face embedding, and a matching model for face verification.

## Prerequisites

1. **Hardware Requirements**:
   - GPU recommended (NVIDIA with CUDA support)
   - At least 16GB RAM
   - 100GB+ storage for datasets

2. **Software Requirements**:
   - Python 3.8-3.11 (avoid 3.12+ due to dependency issues)
   - PyTorch with CUDA support
   - All dependencies from `requirements.txt`

3. **Datasets**:
   - **MTCNN**: WIDER FACE dataset (~50GB)
   - **FaceNet**: VGGFace2 dataset (~8GB) or MS-Celeb-1M
   - **Matching**: LFW dataset for validation (~200MB)

## Data Preparation

1. Run the data preparation script:
```bash
python prepare_training_data.py
```

2. Manually download large datasets:
   - WIDER FACE: http://shuoyang1213.me/WIDERFACE/
   - VGGFace2: https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/
   - CelebA: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

3. Organize data in `training_data/` directory:
```
training_data/
├── WIDER_FACE/
├── VGGFace2/
├── lfw/
├── CelebA/
└── synthetic_faces/  # For testing
```

## Training MTCNN (Face Detection)

MTCNN uses a three-stage cascaded network (P-Net, R-Net, O-Net).

### Data Requirements
- Images with face bounding box annotations
- Format: WIDER FACE style (image path + bounding boxes)

### Training Process
```bash
python train_mtcnn.py
```

**Note**: The current implementation is a placeholder. For full training, you need:
- Multi-task loss implementation (classification + bbox regression + landmarks)
- Three-stage training pipeline
- Hard negative mining
- Refer to original MTCNN paper and implementations

### Expected Results
- P-Net: ~90% recall, fast processing
- R-Net: ~95% precision, medium speed
- O-Net: ~98% precision, slower but accurate

## Training FaceNet (Face Embedding)

FaceNet learns face embeddings using triplet loss.

### Data Requirements
- Large dataset with multiple images per identity
- At least 100 identities with 5+ images each
- Aligned face images (160x160)

### Training Process
```bash
python train_facenet.py
```

**Key Components**:
- **Triplet Loss**: Ensures same person embeddings are close, different persons are far
- **Hard Triplet Mining**: Selects difficult triplets for better training
- **L2 Normalization**: Embeddings normalized to unit sphere

### Hyperparameters
- Learning rate: 0.001
- Batch size: 32-64 triplets
- Margin: 0.2-0.5
- Epochs: 10-50

### Expected Results
- Training accuracy: >95%
- Validation accuracy on LFW: >98%

## Training Matching Model (Face Verification)

Two approaches: binary classifier on embeddings or threshold tuning.

### Approach 1: Train Verification Classifier
```bash
python train_matching.py
```

### Approach 2: Fine-tune Similarity Threshold
```bash
python train_matching.py  # Uses fine_tune_threshold function
```

### Data Requirements
- Pairs of face images (same person + different persons)
- Balanced dataset (equal positive/negative pairs)

### Expected Results
- Verification accuracy: >95%
- Optimal threshold: ~0.6-0.8

## Training Tips

### General
1. **Start Small**: Use synthetic data first to test pipeline
2. **Monitor Loss**: Watch for overfitting (validation loss increases)
3. **GPU Memory**: Reduce batch size if OOM errors
4. **Data Augmentation**: Use random crops, flips, brightness changes

### MTCNN Specific
- Train P-Net first, then use its output to train R-Net, then O-Net
- Use hard negative mining to improve performance
- Balance positive/negative/partial face samples

### FaceNet Specific
- Use large margin (0.5+) for better separation
- Implement online triplet mining for efficiency
- Pre-train on large datasets before fine-tuning

### Matching Specific
- Use cosine similarity for embeddings
- Validate on LFW or similar benchmark
- Consider ROC curves for threshold selection

## Evaluation

### Metrics
- **Face Detection**: Precision, Recall, IoU
- **Face Embedding**: Accuracy on verification task
- **Face Matching**: Accuracy, Precision, Recall, F1-score

### Benchmark Datasets
- **Detection**: FDDB, WIDER FACE
- **Verification**: LFW, YTF, IJB-A
- **Identification**: MegaFace

## Troubleshooting

### Common Issues
1. **CUDA OOM**: Reduce batch size, use gradient accumulation
2. **Slow Training**: Check GPU utilization, optimize data loading
3. **Poor Accuracy**: Check data quality, increase model capacity
4. **Overfitting**: Add regularization, use data augmentation

### Performance Optimization
- Use mixed precision training (FP16)
- Implement distributed training for multiple GPUs
- Optimize data loading with multiple workers
- Use model quantization for inference

## Integration

After training, update the main system:
1. Replace pre-trained models with your trained models
2. Update config.py with new model paths
3. Retrain storage with new embeddings if needed
4. Test end-to-end system with `python test_system.py`

## Resources

- **Papers**:
  - MTCNN: "Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks"
  - FaceNet: "FaceNet: A Unified Embedding for Face Recognition and Clustering"

- **Implementations**:
  - Official FaceNet: https://github.com/davidsandberg/facenet
  - MTCNN: https://github.com/ipazc/mtcnn

- **Datasets**:
  - WIDER FACE: http://shuoyang1213.me/WIDERFACE/
  - VGGFace2: https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/
  - LFW: http://vis-www.cs.umass.edu/lfw/
