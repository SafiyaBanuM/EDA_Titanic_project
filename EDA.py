# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv(r"C:/Users/SAFIYA BANU/Downloads/titanic.csv")

# Display first 5 rows
print(df.head())

# Dataset Information
print("\nDataset Information:")
print(df.info())

# Statistical Summary
print("\nStatistical Summary:")
print(df.describe())

# Shape of Dataset
print("\nShape:", df.shape)

# Column Names
print("\nColumns:")
print(df.columns)

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate Values
print("\nDuplicate Rows:", df.duplicated().sum())

# Survival count
survival = df["Survived"].value_counts()

plt.figure(figsize=(6,4))
plt.bar(["Not Survived", "Survived"], survival)
plt.title("Survival Count")
plt.xlabel("Survival Status")
plt.ylabel("Number of Passengers")
plt.show()

#Passenger class distribution
pclass = df["Pclass"].value_counts().sort_index()

plt.figure(figsize=(6,4))
plt.bar(pclass.index.astype(str), pclass.values)
plt.title("Passenger Class Distribution")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.show()

#Gender Distribution
gender = df["Sex"].value_counts()

plt.figure(figsize=(6,4))
plt.pie(gender.values,
        labels=gender.index,
        autopct='%1.1f%%',
        startangle=90)

plt.title("Gender Distribution")
plt.show()

#Age Distribution
plt.figure(figsize=(7,4))
plt.hist(df["Age"].dropna(), bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

#Fair Distribution
plt.figure(figsize=(7,4))
plt.hist(df["Fare"], bins=20)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.show()

#Age Box Plot
plt.figure(figsize=(6,4))
plt.boxplot(df["Age"].dropna())
plt.title("Age Box Plot")
plt.ylabel("Age")
plt.show()

#Fair box plot
plt.figure(figsize=(6,4))
plt.boxplot(df["Fare"])
plt.title("Fare Box Plot")
plt.ylabel("Fare")
plt.show()

#correlation matrix
corr = df.corr(numeric_only=True)

print(corr)

#Correlation Heatmap
plt.figure(figsize=(8,6))
plt.imshow(corr, cmap='coolwarm')

plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)

plt.colorbar()
plt.title("Correlation Heatmap")

plt.show()
