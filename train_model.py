import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load and preprocess data (same as before)
df = pd.read_csv("career_data.csv")
df['Skills'] = df['Skills'].apply(lambda x: x.split(';'))

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
skills_encoded = mlb.fit_transform(df['Skills'])
skills_df = pd.DataFrame(skills_encoded, columns=mlb.classes_)

df_encoded = pd.get_dummies(df[['Degree', 'Interest']])
X = pd.concat([skills_df, df_encoded, df[['GPA']]], axis=1)
y = df['Recommended_Career']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict & evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained with accuracy: {acc:.2f}")

# Save model and label encoder
joblib.dump(model, 'career_model.pkl')
joblib.dump(mlb, 'skills_encoder.pkl')

print("✅ Model + encoder saved as .pkl files!")
