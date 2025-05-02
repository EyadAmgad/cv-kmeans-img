KMeans Image Clustering with Color Segmentation
This project implements the K-Means clustering algorithm from scratch to segment an image into 3 color-coded clusters using:

White [255, 255, 255]

Red [255, 0, 0]

Green [0, 255, 0]

Features
KMeans clustering implementation using only NumPy

Works with any input image (JPG, PNG, etc.)

Visually displays clustered image using matplotlib

Assigns a unique color to each cluster for clear differentiation

How It Works
The input image is loaded and reshaped into a 2D array of RGB pixels.

KMeans groups these pixels into K=3 clusters based on RGB similarity.

Each cluster is assigned a unique color: white, red, or green.

The clustered image is reshaped and displayed.

