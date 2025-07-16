import pandas as pd

# Load and preprocess the dataset
df = pd.read_csv("data/train.csv")

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Combine Manufacturer and Model into a unified identifier
df['Car_Name'] = (df['Manufacturer'].astype(str) + ' ' + df['Model'].astype(str)).str.lower().str.strip()

def get_car_info(predicted_name):
    name = predicted_name.lower().strip()
    matches = df[df['Car_Name'].str.contains(name)]

    if matches.empty:
        return None  # Return None for app.py to handle

    avg_price = matches['Price'].mean()

    top_match = matches.iloc[0]
    info = {
        "Model": top_match['Car_Name'].title(),
        "Year": int(top_match['Prod._year']),
        "Fuel": top_match['Fuel_type'],
        "Transmission": top_match['Gear_box_type'],
        "Drive": top_match['Drive_wheels'],
        "KM Driven": top_match['Mileage'],
        "Average_Price_(₹)": f"{avg_price:,.0f}"
    }
    return info

