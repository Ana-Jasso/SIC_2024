import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Read the data
train = pd.read_csv('train.csv')  # Adjust the path as needed

# Prepare the data
X = train.drop(['efs', 'efs_time', 'y_nel', 'ID'], axis=1)  # Remove target and ID columns
y = train['efs']  # Use event-free survival as target

# Handle categorical variables
le = LabelEncoder()
for column in X.columns:
    if X[column].dtype == 'object':
        X[column] = le.fit_transform(X[column].astype(str))

# Train a Random Forest to get feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Get feature importance scores
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
})

# Sort features by importance
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Select top 25 features
top_25_features = feature_importance.head(25)['feature'].tolist()

# Create a new DataFrame with only the selected features
df_selected = train[top_25_features]

# Calculate the correlation matrix
corr_matrix = df_selected.corr()

# Create the heatmap
plt.figure(figsize=(15, 12))
sns.heatmap(corr_matrix, annot=True, cmap='viridis', fmt='.2f', square=True)
plt.title('Correlation Heatmap of 25 Most Important Features')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Print the selected features and their importance scores
print("\nTop 25 Most Important Features:")
print(feature_importance.head(25)) 