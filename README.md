performing exploratory data analysis for the given dataset and giving the results on the products which gives the most purchased product 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('customers.csv')
datatype= df.dtypes
print(datatype)

num_rows, num_columns = df.shape
print("\nNumber of rows:", num_rows)
print("Number of columns:", num_columns)

missing_values = df.isnull().sum()
print("\nNumber of missing values in each column:")
print(missing_values)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("customers.csv")

# Select only continuous variables (assuming all numeric columns are continuous)
continuous_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Detect outliers using boxplots
plt.figure(figsize=(12, 6))
for col in continuous_columns:
    plt.subplot(1, len(continuous_columns), list(continuous_columns).index(col) + 1)
    sns.boxplot(y=df[col], color='skyblue')
    plt.title(col)
plt.tight_layout()
plt.show()

# Clip the data between the 5th and 95th percentiles
clipped_df = df.copy()
for col in continuous_columns:
    lower_bound = df[col].quantile(0.05)
    upper_bound = df[col].quantile(0.95)
    clipped_df[col] = np.clip(df[col], lower_bound, upper_bound)

# Visualize the clipped data using boxplots
plt.figure(figsize=(12, 6))
for col in continuous_columns:
    plt.subplot(1, len(continuous_columns), list(continuous_columns).index(col) + 1)
    sns.boxplot(y=clipped_df[col], color='skyblue')
    plt.title(col)
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("customers.csv")

# Select only continuous variables (assuming all numeric columns are continuous)
continuous_columns = df.select_dtypes(include=['float64', 'int64']).columns



# Clip the data between the 5th and 95th percentiles
clipped_df = df.copy()
for col in continuous_columns:
    lower_bound = df[col].quantile(0.05)
    upper_bound = df[col].quantile(0.95)
    clipped_df[col] = np.clip(df[col], lower_bound, upper_bound)

# Visualize the clipped data using boxplots
plt.figure(figsize=(12, 6))
for col in continuous_columns:
    plt.subplot(1, len(continuous_columns), list(continuous_columns).index(col) + 1)
    sns.boxplot(y=clipped_df[col], color='skyblue')
    plt.title(col)
plt.tight_layout()
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Import the dataset
# Assuming the dataset is in CSV format and named "customers.csv"
df = pd.read_csv("customers.csv")

# Relationship between categorical variables and output variable using count plots
categorical_variables = ['MaritalStatus', 'Gender']  # Add more if needed
output_variable = 'Product'  # Assuming the name of the output variable
plt.figure(figsize=(12, 6))
for col in categorical_variables:
    plt.subplot(1, len(categorical_variables), list(categorical_variables).index(col) + 1)
    sns.countplot(x=col, hue=output_variable, data=df)
    plt.title(f"{col} vs {output_variable}")
plt.tight_layout()
plt.show()

# Relationship between continuous variables and output variable using scatter plots
continuous_variables = ['Age']  # Add more if needed
plt.figure(figsize=(12, 6))
for col in continuous_variables:
    plt.subplot(1, len(continuous_variables), list(continuous_variables).index(col) + 1)
    sns.scatterplot(x=col, y=output_variable, data=df)
    plt.title(f"{col} vs {output_variable}")
plt.tight_layout()
plt.show()

import pandas as pd

# Import the dataset
# Assuming the dataset is in CSV format and named "customers.csv"
df = pd.read_csv("customers.csv")

# Find the marginal probability (what percent of customers have purchased KP281, KP481, or KP781)
marginal_prob = df['Product'].value_counts(normalize=True)
print("Marginal probability of each product:")
print(marginal_prob)

# Find the probability that the customer buys a product based on each column
# Assuming the columns represent different features such as marital status, gender, etc.
conditional_prob = pd.crosstab(index=df['Product'], columns=df['MaritalStatus'], normalize='index')
print("\nProbability of buying each product based on marital status:")
print(conditional_prob)
# Find the conditional probability that an event occurs given that another event has occurred
# Example: given that a customer is female, what is the probability she’ll purchase a KP481
female_prob = df[df['Gender'] == 'Female']['Product'].value_counts(normalize=True)
male_prob = df[df['Gender'] == 'Male']['Product'].value_counts(normalize=True)
prob_female_kp281 = female_prob.get('KP281', 0)
prob_male_kp281 = male_prob.get('KP281',0)
print("\nConditional probability of purchasing KP281 given that the customer is female:")
print(prob_female_kp281)
print("\nConditional probability of purchasing KP281 given that the customer is male:")
print(prob_male_kp281)

import pandas as pd

# Import the dataset
# Assuming the dataset is in CSV format and named "customers.csv"
df = pd.read_csv("customers.csv")

# Customer profiling for product KP281
kp281_customers = df[df['Product'] == 'KP281']

# Age profile
age_profile = kp281_customers['Age'].describe()

# Gender profile
gender_profile = kp281_customers['Gender'].value_counts(normalize=True)

# Income group profile (assuming income is a column in the dataset)
income_profile = kp281_customers['Income'].value_counts(normalize=True)

print("Customer Profiling for Product KP281:")
print("Age Profile:")
print(age_profile)
print("\nGender Profile:")
print(gender_profile)
print("\nIncome Group Profile:")
print(income_profile)

