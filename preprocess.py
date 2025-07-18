import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# Step 1: Load the data
df = pd.read_csv("career_data.csv")

# Step 2: Process skills (split by ;)
df['Skills'] = df['Skills'].apply(lambda x: x.split(';'))
mlb = MultiLabelBinarizer()
skills_encoded = mlb.fit_transform(df['Skills'])
skills_df = pd.DataFrame(skills_encoded, columns=mlb.classes_)

# Step 3: One-hot encode Degree and Interest
df_encoded = pd.get_dummies(df[['Degree', 'Interest']])

# Step 4: Combine all features
X = pd.concat([skills_df, df_encoded, df[['GPA']]], axis=1)
y = df['Recommended_Career']

print("✅ Preprocessing Complete!")
print("Feature columns:", X.columns.tolist())
print("Target example:", y[:5].tolist())
