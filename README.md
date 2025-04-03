# k-Nearest Neighbors (k-NN) Classification on Wine Dataset

This project implements a custom k-Nearest Neighbors (k-NN) classifier on the UCI Wine dataset using both **Euclidean** and **Manhattan** distance metrics. It includes visualization of the dataset, performance evaluation, and result plotting.

---

## Project Structure

- **Data Source**: The dataset is read from a local path (`wine.data`) and includes features like alcohol, magnesium, flavanoids, etc.
- **Visualization**: Various scatter plots for understanding feature distributions between different classes.
- **Data Preprocessing**:
  - Missing value check
  - Standardization using `StandardScaler`
- **Classifier Implementation**:
  - Custom functions for Euclidean and Manhattan distances
  - Manual k-NN implementation with majority voting and weighted tie-breaking
- **Model Evaluation**:
  - Accuracy vs. k-value plot for both distance metrics
  - Confusion matrix heatmap
  - Classification report heatmap

---

## Dependencies

Install the required Python libraries using the command below:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
