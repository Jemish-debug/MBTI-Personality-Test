import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("data\\mbti_1.csv")  # Replace with your file path

# Split into train/test sets (stratified split to preserve class distribution)
train_df, test_df = train_test_split(
    df,
    test_size=0.3,           # 30% for testing
    stratify=df['type'],      # Preserve class balance
    random_state=42          # For reproducibility
)

# Save to CSV files
train_df.to_csv("data\\train.csv", index=False)
test_df.to_csv("data\\test.csv", index=False)

print(f"Training samples: {len(train_df)}")
print(f"Testing samples: {len(test_df)}")