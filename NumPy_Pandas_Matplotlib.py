import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

arr1 = np.array([14, 74, 47, 58, 36, 81,27, 712, 982])
arr2 = np.array([[21, 76, 83, 37], [47, 25, 68, 87]])

print("Original Array:", arr1)
print("Array + 47:", arr1 + 47)
print("Sliced Array:", arr1[2:5])
print("Reshaped 2D Array:\n", arr2.reshape(4, 2))


data = {
    'Name': ['Jay', 'Kaelyn', 'Rafayel','Satomi','Eula'],
    'Age': [23, 19, 24,21,25],
    'Score': [85, 90, 88,87,78]
}

df = pd.DataFrame(data)

print("\nDataFrame:")
print(df)

print("\nDataFrame Info:")
print(df.info())

print("\nStatistics:")
print(df.describe())


# Line Plot
plt.figure()
plt.plot(df['Name'], df['Score'],color = '#AF69ED', marker='o')
plt.title('Scores of Students')
plt.xlabel('Name')
plt.ylabel('Score')
plt.grid(True)
plt.show()

# Bar Plot
plt.figure()
plt.bar(df['Name'], df['Age'], color='#F2C1D1')
plt.title('Age of Students')
plt.xlabel('Name')
plt.ylabel('Age')
plt.show()

# Pie Chart
plt.figure()
color_list=['#2E8B57','#F0AF87','#0096C7','#E0B0FF','#FFEE8C']
plt.pie(df['Score'], labels=df['Name'], autopct='%1.1f%%', startangle=90,colors=color_list)
plt.title('Score Distribution')
plt.show()
