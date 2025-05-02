## KMeans Image Clustering with Color Segmentation

This project implements the **K-Means clustering algorithm from scratch** to segment an image into **3 color-coded clusters** using:
- **White** `[255, 255, 255]`
- **Red** `[255, 0, 0]`
- **Green** `[0, 255, 0]`

---

### Features

- KMeans clustering implementation using only **NumPy**
- Works with any input image (JPG, PNG, etc.)
- Visually displays clustered image using `matplotlib`
- Assigns a unique color to each cluster for clear differentiation

---

### How It Works

1. The input image is loaded and reshaped into a 2D array of RGB pixels.
2. KMeans groups these pixels into `K=3` clusters based on RGB similarity.
3. Each cluster is assigned a unique color: white, red, or green.
4. The clustered image is reshaped and displayed.

---

### Requirements

Install the required libraries (all are standard and widely used):

```bash
pip install numpy opencv-python matplotlib
