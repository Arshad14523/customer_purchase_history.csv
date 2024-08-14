import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load the CSV file with error handling
def load_data(filename):
    try:
        # Load the data, skipping bad lines
        df = pd.read_csv(filename, on_bad_lines='skip')
        return df
    except pd.errors.ParserError as e:
        print("Error parsing the CSV file:", e)
        return None

# Load the dataset
df = load_data('customer_purchase_history.csv')

# Proceed only if the DataFrame is loaded successfully
if df is not None:
    # Display the first few rows to confirm proper loading
    print("Data loaded successfully:")
    print(df.head())

    # Check columns and data types
    print("Columns and data types:")
    print(df.dtypes)

    # Features for clustering
    features = df[['TotalPurchases', 'FrequencyOfPurchases', 'AvgPurchaseAmount']]

    # Standardize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)  # Adjust the number of clusters if needed
    df['Cluster'] = kmeans.fit_predict(features_scaled)

    # Print cluster centers
    print("Cluster Centers:")
    print(kmeans.cluster_centers_)

    # Print first few rows with cluster labels
    print(df.head())

    # Plotting clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='TotalPurchases', y='FrequencyOfPurchases', hue='Cluster', palette='Set2')
    plt.title('K-means Clustering of Customers')
    plt.xlabel('Total Purchases')
    plt.ylabel('Frequency of Purchases')
    plt.legend(title='Cluster')
    plt.show()
else:
    print("Failed to load data.")
