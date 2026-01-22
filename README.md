# Customer Segmentation using K-Means Clustering

This project implements **Customer Segmentation** using the **K-Means Clustering algorithm** on mall customer data.  
The goal is to group customers into distinct clusters based on their **Annual Income** and **Spending Score**, enabling businesses to better understand customer behavior and target them effectively.

---

## 📌 Project Overview

Customer segmentation is a key application of unsupervised machine learning in marketing analytics.  
In this project:

- The dataset is analyzed and preprocessed
- The optimal number of clusters is determined using the **Elbow Method**
- Customers are segmented using **K-Means**
- Results are visualized with clusters and centroids

---

## 📂 Dataset

- **Dataset Name:** Mall_Customers.csv  
- **Source:** Mall customer records  
- **Features Used:**
  - Annual Income
  - Spending Score

---

## 🛠️ Technologies Used

- Python 3
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## ⚙️ Project Workflow

1. Load the dataset using Pandas
2. Select relevant features (Annual Income & Spending Score)
3. Apply the **Elbow Method** to determine optimal clusters
4. Train the **K-Means Clustering model**
5. Assign each customer to a cluster
6. Visualize clusters along with their centroids

---

## 📊 Elbow Method

The **Elbow Method** is used to identify the optimal number of clusters by plotting the **WCSS (Within-Cluster Sum of Squares)** against different cluster counts.

The elbow point in the graph indicates the best value of **K = 5**.

---

## 📈 Visualization

- Each cluster is represented using a distinct color
- Centroids are highlighted separately
- X-axis: Annual Income
- Y-axis: Spending Score

---

## 🧠 Model Used

**K-Means Clustering**

- Initialization: k-means++
- Number of clusters: 5
- Random state used for reproducibility

---

## ▶️ How to Run the Project

1. Clone the repository:
   ```bash
   git clone <repository-url>

2.	Navigate to the project directory:
cd Customer-Segmentation

3.	Install dependencies:
pip install pandas numpy matplotlib seaborn scikit-learn

4.	Run the Python script:
python customer_segmentation.py


⸻

📌 Output
	•	Elbow Curve for cluster selection
	•	Scatter plot of customer segments
	•	Clearly marked centroids

⸻

🎯 Use Cases
	•	Targeted marketing strategies
	•	Customer behavior analysis
	•	Business decision-making
	•	Market segmentation

⸻

📜 License

This project is open-source and available for educational and learning purposes.

⸻

✨ Author

Vansh Saxena
Machine Learning Enthusiast