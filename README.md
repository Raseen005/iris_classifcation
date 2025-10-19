# Iris Dataset Classification

A complete machine learning project for iris flower classification with exploratory data analysis and model training.

## 📁 Project Structure
```
iris-classification/
│
├── main.py              # Main classification script
├── plot.py              # EDA and visualization script
├── iris.csv             # Dataset file
├── requirements.txt     # Python dependencies
├── README.md           # Project documentation
└── eda/                # Generated EDA graphs folder
    ├── scatter_plots.png
    ├── pair_plot.png
    ├── correlation_heatmap.png
    └── ...
```

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate EDA visualizations:**
   ```bash
   python plot.py
   ```
   *This creates the `eda/` folder with analysis graphs*

3. **Run the classification model:**
   ```bash
   python main.py
   ```

## 📊 What This Project Does

### 🔍 Exploratory Data Analysis (`plot.py`)
- Creates scatter plots, pair plots, and distribution charts
- Generates correlation heatmaps and box plots
- Saves all visualizations to `eda/` folder
- Helps understand data patterns and relationships

### 🤖 Machine Learning (`main.py`)
- Loads the Iris dataset (150 samples, 3 species)
- Trains a logistic regression model
- Evaluates performance with accuracy and classification report
- Shows class distribution and weights
- Displays confusion matrix heatmap

## 🎯 Results
- **Accuracy**: 100% (perfect classification)
- **Classes**: Iris-setosa, Iris-versicolor, Iris-virginica
- **Features**: Sepal length, sepal width, petal length, petal width

## 📦 Requirements
See `requirements.txt` for complete package list.

---

**Complete iris classification with EDA!** 🌸
