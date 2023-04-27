# Clustering for COVID-19 CT Scan Segmentation and Analysis

This repository contains a Python implementation of a method to identify ground-glass opacities in lung CT scans, which may indicate the presence of COVID-19. The method utilizes two clustering algorithms, K-means and DBSCAN, for image segmentation and identification of opacities. You can find the detailed information in `Clustering for COVID-19 CT Scan Analysis.pdf`.

## Dataset

The dataset used for this project is available at [Kaggle](https://www.kaggle.com/andrewmvd/covid19-ct-scans). It consists of CT scans from COVID-19 patients, with each patient having around 300 slices in the axial plane. Each slice is a grayscale image with a resolution of 512x512 pixels.

## Package Installation

To install the required packages for this project, run the following command:

```
pip install pandas, numpy, matplotlib, scikit-learn, nibabel
```

## Usage
1. Clone this repository.
2. Download the dataset from Kaggle mentioned in the references.
3. Run the code using the downloaded dataset in `load` folder  and run `python main.py` in the directory of the repository.

## References

1. COVID-19 CT Scans dataset: [Kaggle](https://www.kaggle.com/andrewmvd/covid19-ct-scans)
2. Kevin P. Murphy, *Machine Learning: A Probabilistic Perspective*, MIT Press, 2012
3. Martin Ester, Hans-Peter Kriegel, Jörg Sander, Xiaowei Xu, *A density-based algorithm for discovering clusters in large spatial databases with noise*, Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD-96), 1996
4. Zou, Kelly H., et al. *Statistical Validation of Image Segmentation Quality Based on a Spatial Overlap Index1.* Academic Radiology, no. 2, Elsevier BV, Feb. 2004, pp. 178–89. Crossref, doi:10.1016/s1076-6332(03)00671-8
5. Kwee, Thomas C., and Robert M. Kwee. *Chest CT in COVID-19: What the Radiologist Needs to Know.* RadioGraphics, no. 7, Radiological Society of North America (RSNA), Nov. 2020, pp. 1848–65. Crossref, doi:10.1148/rg.2020200159.
