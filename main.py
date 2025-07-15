import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# I am using Google Colab platform
#First make a folder named "Content" in google drive then add a "Sample_price_data.csv" file in that folder and if you are using any other platform then skip these 3 line code below
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/Content

# Load CSV file
df = pd.read_csv("Sample_price_data.csv")  

# Remove unwanted columns
columns_to_drop = ["base_price", "zip_multiplier"]
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

# Set X and y
X = df.drop(columns=["price"])
y = df["price"]

# Define preprocessing
categorical_cols = ["zip_code"]
numerical_cols = [col for col in X.columns if col not in categorical_cols]

preprocessor = ColumnTransformer([
    ("onehot", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", "passthrough", numerical_cols)
])

# Create the pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
rmse = mean_squared_error(y_test, y_pred) ** 0.5  
print("RMSE:", rmse)

#  Predicting from a sample input
sample_input = pd.DataFrame([{
    "num_rooms": 4,
    "floor_area": 1200,
    "zip_code": "400005",
    "school_distance": 2.5,
    "crime_rate": 2.0,
    "year_built": 2018
}])

predicted_price = model.predict(sample_input)[0]

# Convert to INR
inr_price = 83.00 * predicted_price
print(f"Predicted House Price: ₹{inr_price:,.2f}")





#Let's Visualize the data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/Content

# Load CSV file
df = pd.read_csv("Sample_price_data.csv") 

# Convert relevant columns to numeric
df["school_distance"] = pd.to_numeric(df["school_distance"], errors="coerce")
df["crime_rate"] = pd.to_numeric(df["crime_rate"], errors="coerce")
df.dropna(subset=["school_distance", "crime_rate"], inplace=True)

# Set theme
sns.set_theme(style="whitegrid")

# Visualize distributions of numerical columns
numerical_cols = [col for col in X.columns if col != 'zip_code']

plt.figure(figsize=(15, 10))
for idx, col in enumerate(numerical_cols):
    plt.subplot(2, 3, idx + 1)
    sns.histplot(df[col], kde=True, bins=30, color='skyblue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Price vs Floor Area
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="floor_area", y="price", hue="num_rooms", palette="viridis")
plt.title("Price vs Floor Area (by Rooms)")
plt.tight_layout()
plt.show()

# Price vs Zip Code
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="zip_code", y="price")
plt.title("Price by Zip Code")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Price vs Number of Rooms
plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x="num_rooms", y="price")
plt.title("Price vs Number of Rooms")
plt.tight_layout()
plt.show()


# Price vs School Distance
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="school_distance", y="price", hue="num_rooms", palette="coolwarm")
plt.title("Price vs School Distance")
plt.tight_layout()
plt.show()

# Price vs Crime Rate
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="crime_rate", y="price", hue="num_rooms", palette="Spectral")
plt.title("Price vs Crime Rate")
plt.tight_layout()
plt.show()
