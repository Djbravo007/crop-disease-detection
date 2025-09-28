# 🌱 Crop Disease Detection (Buggy Version)

## 📌 Goal
Detect whether a crop leaf has disease or not (and which one).

## 🐛 Current Issues (to be fixed)
1. Dataset contains duplicate/blurred images → need cleaning.
2. Wrong labeling possible if dataset folder names are inconsistent.
3. No normalization of images → model accuracy unstable.
4. No data augmentation → model overfits easily.
5. CNN model is too shallow → poor accuracy.
6. No dropout/regularization → risk of overfitting.
7. Evaluation only uses accuracy → need precision, recall, F1.
8. No explainability (Grad-CAM not implemented).
9. Hardcoded dataset path → should allow CLI/argparse config.

## 🚀 Contribution Guidelines
- Fork the repo.
- Raise issues based on the above problems (or new ones you find).
- Create pull requests with fixes (dataset cleaning, augmentation, better model, Grad-CAM).
-
