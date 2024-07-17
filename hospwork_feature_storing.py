import hopsworks
import pandas as pd

project = hopsworks.login()

fs = project.get_feature_store()

temp = pd.read_csv('existing_data.csv',index_col = 'Date')

# Get or create the 'transactions_fraud_batch_fg' feature group
trans_fg = fs.get_or_create_feature_group(
    name="Gold_price_prediction_features",
    version=1,
    description="Gold_price_data",
    primary_key=["Date"],
    event_time="datetime",
)

# Insert data into feature group
trans_fg.insert(temp)

# Update feature descriptions
feature_descriptions = [
    {"name": "Date", "description": "relevant date"},
    {"name": "24 Carat 1 Gram", "description": "price according to dates"}
]

for desc in feature_descriptions: 
    trans_fg.update_feature_description(desc["name"], desc["description"])