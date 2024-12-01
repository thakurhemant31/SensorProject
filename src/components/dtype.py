import pandas as pd

# Load your dataset (example)
data = pd.read_csv('D:/sensor_pro/notebook/Wafer_14012020_113045.csv')

# Check the data types of columns
print(data.dtypes)

# Check for any rows where a column is a string but should be numeric
for column in data.columns:
    if data[column].dtype == 'object':  # Object dtype usually indicates strings
        print(f"Column: {column}")
        print(data[column].unique())  # This will show unique values in the column
