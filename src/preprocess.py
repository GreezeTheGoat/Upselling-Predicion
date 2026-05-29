import pandas as pd
from sklearn.model_selection import train_test_split

# ---------- Opening Raw data and removing useless columns ----------

df = pd.read_csv("../data/raw/Telco_data.csv")

df = df[df["Churn Value"] == 0]
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('future.no_silent_downcasting', True)

useless_columns = ["Count", "Country", "State", "City", "Zip Code", "Lat Long", "Latitude", "Longitude","Churn Value", "Churn Score", "Churn Label", "Churn Reason", "Monthly Charges", "Total Charges"]
df = df.drop(columns = useless_columns)


# ---------- Replacing strings with numbers ----------

numeric_map = {
    "Yes": 1,
    "Male": 1,
    "No": 0,
    "No internet service": 0,
    "No phone service": 0,
    "Female": 0,
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2,
    "Bank transfer (automatic)": 0,
    "Credit card (automatic)": 0,
    "Electronic check": 1,
    "Mailed check": 2,
    "DSL": 1,
    "Fiber optic": 2,
}

df = df.replace(numeric_map)

# ---------- Creating "Total Extra Services" and "is_fiber" columns ----------

extra_services_columns = ["Phone Service", "Multiple Lines", "Online Security", "Online Backup", "Device Protection", "Streaming TV", "Streaming Movies"]

df["Total Extra Services"] = df[extra_services_columns].sum(axis = 1)


df["is_fiber"] = (df["Internet Service"] == 2).astype(int) # Removing ghost clients

# ---------- Creating training and testing processed data

X = df.drop(columns = ["is_fiber", "Internet Service", "CustomerID"])
y = df["is_fiber"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

path_processed = "../data/processed/"

X_train.to_csv(f"{path_processed}X_train.csv", index=False)
X_test.to_csv(f"{path_processed}X_test.csv", index=False)
y_train.to_csv(f"{path_processed}y_train.csv", index=False)
y_test.to_csv(f"{path_processed}y_test.csv", index=False)

# ---------- Creating no fiber clients dataset ----------

df_nofiber = df[df["is_fiber"] == 0].copy()
df_nofiber = df_nofiber.drop(columns = ["is_fiber", "Internet Service"])

optimized_columns = [
    "CustomerID",
    'CLTV',
 'Tenure Months',
 'Total Extra Services',
 'Payment Method',
 'Multiple Lines',
 'Contract',
 'Streaming TV',
 'Phone Service',
 'Paperless Billing',
 'Streaming Movies',
 'Senior Citizen',
 'Tech Support',
 'Online Security']

df_nofiber = df_nofiber[optimized_columns]

df_nofiber.to_csv(f"{path_processed}nf_clients.csv", index=False)

print("---------- Processed raw data with sucsess ----------")
