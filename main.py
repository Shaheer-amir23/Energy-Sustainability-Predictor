#PUT ALL IMPORTS UP HEREEE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

#SECTION 2: DATA CLEANING AND FEATURE ENGINEERING
#Show dataframe attributes and clean up data that is irrelevant to our goal
df = pd.read_csv('global-data-on-sustainable-energy (1).csv') 
df = df.drop(columns = ['Latitude', 'Longitude', 'Land Area(Km2)', 'Electricity from nuclear (TWh)', 'Access to clean fuels for cooking', 'Financial flows to developing countries (US $)', 'Renewables (% equivalent primary energy)'])

df.head()

#Keeping rows of only the 10 countries that will be used for our analyses 
countries = ['Bahrain', 'Libya', 'Jordan', 'Iraq', 'Lebanon', 'United Arab Emirates', 'Kuwait', 'Oman', 'Qatar']

data = df[df['Entity'].isin(countries)]

data = data.dropna()
'''
MAY OR MAY NOT NEED 
# Calculate year-over-year CO2 emissions change
data['CO2_Gain_YOY'] = data.groupby('Entity')['Value_co2_emissions_kt_by_country'].diff()

# Display the updated DataFrame to check the results
data[['Entity', 'Year', 'Value_co2_emissions_kt_by_country', 'CO2_Gain_YOY']].head(10)
'''
print(len(data))
data.head(210)

#CONSTRUCT FEATURES THAT WILL BE USED IN ANALYSYS

# SECTION 3: DATA VISUALIZATION
# Select only the numerical features from the dataset
numerical_features = df.select_dtypes(include=['float64', 'int64'])

# Compute the correlation matrix for numerical features
correlation_matrix = numerical_features.corr()

# Display the correlation matrix
correlation_matrix

# Plot a heatmap to visualize the correlation between numerical features
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

# Identify the top 5 most correlated features with the target variable CO2 Emissions ('Value_co2_emissions_kt_by_country')
target = 'Value_co2_emissions_kt_by_country'
most_correlated_features = correlation_matrix[target].sort_values(ascending=False)[1:6]  # Skip the target itself
print("Top 5 most correlated features with 'Value_co2_emissions_kt_by_country':")

# Iterate through these features
for i, feature in enumerate(most_correlated_features.index, start=1):
    print(f"{i}. {feature}: Correlation = {most_correlated_features[feature]:.4f}")

print() #empty line

# Identify the top 5 most correlated features with the target variable gdp growth per capita ('gdp_per_capita')
target = 'gdp_per_capita'
most_correlated_features = correlation_matrix[target].sort_values(ascending=False)[1:6]  # Skip the target itself
print("Top 5 most correlated features with 'gdp_per_capita':")

# Iterate through these features
for i, feature in enumerate(most_correlated_features.index, start=1):
    print(f"{i}. {feature}: Correlation = {most_correlated_features[feature]:.4f}")

print() #empty line

# Identify the top 5 most correlated features with the target variable primary energy consumption per consumption per capita ('Primary energy consumption per capita (kWh/person)')
target = 'Primary energy consumption per capita (kWh/person)'
most_correlated_features = correlation_matrix[target].sort_values(ascending=False)[1:6]  # Skip the target itself
print("Top 5 most correlated features with 'Primary energy consumption per capita (kWh/person)':")

# Iterate through these features
for i, feature in enumerate(most_correlated_features.index, start=1):
    print(f"{i}. {feature}: Correlation = {most_correlated_features[feature]:.4f}")


#SECTION 4: MODELING
import seaborn as sns
import matplotlib.pyplot as plt

# Assume the CO2 emissions column is named 'Value_co2_emissions_kt_by_country'
target_variable = 'Value_co2_emissions_kt_by_country'

# Select only numerical columns, excluding the target variable
numerical_columns = data.select_dtypes(include=['float64', 'int64']).drop(columns=[target_variable, 'Year'])

# Calculate correlations of all features with the target variable (CO2 emissions)
correlations_with_co2 = numerical_columns.corrwith(data[target_variable])

# Convert to a DataFrame for easier plotting
correlations_df = correlations_with_co2.to_frame().reset_index()
correlations_df.columns = ['Feature', 'Correlation with CO2 Emissions']

# Sort the DataFrame by the absolute value of correlations
correlations_df['Absolute Correlation'] = correlations_df['Correlation with CO2 Emissions'].abs()
correlations_df_sorted = correlations_df.sort_values(by='Absolute Correlation', ascending=False).drop(columns='Absolute Correlation')

print("\nCorrelations with CO2 Emissions:")
for index, row in correlations_df_sorted.iterrows():
    print(f"{row['Feature']:<70} {row['Correlation with CO2 Emissions']:.6f}")

# Plot the correlations
plt.figure(figsize=(10, 8))
sns.barplot(x='Correlation with CO2 Emissions', y='Feature', data=correlations_df, palette='coolwarm')
plt.title('Correlation of Each Feature with CO2 Emissions'), 
plt.xlabel('Correlation Coefficient')
plt.ylabel('Features')
plt.show()

#SECTION 5: GRADIENT BOOSTING
# Convert to DataFrame
df = pd.DataFrame(data)

# Features and target variable
X = df[["Electricity from fossil fuels (TWh)","Electricity from renewables (TWh)", "gdp_per_capita","Year","Primary energy consumption per capita (kWh/person)"]]
y = df["Value_co2_emissions_kt_by_country"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Initialize and train the Gradient Boosting model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.2, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
# Predicting and evaluating model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.2f}")


selected_countries = ['Bahrain', 'Libya', 'Jordan', 'Iraq', 'Lebanon', 'United Arab Emirates', 'Kuwait', 'Oman', 'Qatar']
predicted_year = 2040

# Filter the data for the selected countries and update the year
predicted_data = df[df["Entity"].isin(selected_countries)].copy()
predicted_data["Year"] = predicted_year  # Update year for projection


# Features are present
X_future = predicted_data[[
    "Electricity from fossil fuels (TWh)",
    "Electricity from renewables (TWh)",
    "gdp_per_capita",
    "Year",
    "Primary energy consumption per capita (kWh/person)"
]]

# Make predictions
predicted_data["Value_co2_emissions_kt_by_country"] = model.predict(X_future)

# Display projections
print("\nProjected CO2 Emissions for Selected Countries in 20 Years:")
print(predicted_data[["Entity", "Year", "Value_co2_emissions_kt_by_country"]])

# Removing commas to make it numerical
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Dropping rows where the target column is NaN
data_cleaned = data.dropna(subset=[target_variable])

# Defining features and target
features = data_cleaned.drop(columns=[target_variable, 'Entity', 'Year'], errors='ignore')
target = data_cleaned[target_variable]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Training the Random Forest Regressor but reducing overfitting
rf_model = RandomForestRegressor(
    n_estimators=100, 
    max_depth=10,  # Limiting tree depth
    min_samples_split=10,  # Minimum samples needed to split an internal node
    min_samples_leaf=5,  # Minimum samples in leaf nodes
    random_state=42
)
rf_model.fit(X_train, y_train)

# Evaluating feature importance
feature_importances = pd.DataFrame({
    'Feature': features.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Displaying feature importance
print("Feature Importance Ranking:")
print(feature_importances)

# Predicting and evaluating model performance
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

# Plotting feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance in Random Forest')
plt.gca().invert_yaxis()
plt.show()


#SECTION 6: FUTURE PROJECTIONS
correlations = {
    'Electricity from fossil fuels (TWh)': [0.922507, 0.9897, 0.934235],
    'Renewable energy share in the total final energy consumption (%)': [-0.412145], #random forest correlation not applicable here 
    'gdp_per_capita': [0.359253],
    'Electricity from renewables (TWh)': [0.8563, 0.317732],
    'Primary energy consumption per capita (kWh/person)': [0.180025],
    'Access to electricity (% of population)': [0.1270, 0.059212 ],
}

# Compute average correlations
average_correlations = {feature: sum(values) / len(values) for feature, values in correlations.items()}

# Sort features by absolute average correlation
sorted_correlations = sorted(average_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
print("Sorted Features by Average Correlation:")
for feature, corr in sorted_correlations:
    print(f"{feature}: {corr:.4f}")

# Selecting top 5 features based on average correlation
top_features = [feature for feature, _ in sorted_correlations[:5]]

# Step 1: Forecasted feature values for 2020 to 2040
years = np.arange(2020, 2041)  # Years from 2020 to 2040

# Assuming linear growth projections for each feature
future_values = pd.DataFrame({
    'Year': years,
    'Electricity from fossil fuels (TWh)': np.linspace(df['Electricity from fossil fuels (TWh)'].iloc[-1], df['Electricity from fossil fuels (TWh)'].iloc[-1] * 1.05, len(years)),  # 5% growth
    'gdp_per_capita': np.linspace(df['gdp_per_capita'].iloc[-1], df['gdp_per_capita'].iloc[-1] * 1.03, len(years)),  # 3% growth
    'Electricity from renewables (TWh)': np.linspace(df['Electricity from renewables (TWh)'].iloc[-1], df['Electricity from renewables (TWh)'].iloc[-1] * 1.04, len(years)),  # 4% growth
    'Primary energy consumption per capita (kWh/person)': np.linspace(df['Primary energy consumption per capita (kWh/person)'].iloc[-1], df['Primary energy consumption per capita (kWh/person)'].iloc[-1] * 1.02, len(years)),  # 2% growth
    'Access to electricity (% of population)': np.linspace(df['Access to electricity (% of population)'].iloc[-1], df['Access to electricity (% of population)'].iloc[-1] * 1.01, len(years)),  # 1% growth
})

# Check columns in the training dataset (df)
print("Training Data Columns (df):", df.columns)

# Check columns in the forecasted feature values (future_values)
print("Future Data Columns (future_values):", future_values.columns)

# Step 3: Add the missing feature (Renewable energy share in the total final energy consumption (%))
last_value_renewable_energy_share = df['Renewable energy share in the total final energy consumption (%)'].iloc[-1]

# Add this feature to the future_values DataFrame with a linear growth assumption (adjust as needed)
future_values['Renewable energy share in the total final energy consumption (%)'] = np.linspace(
    last_value_renewable_energy_share,
    last_value_renewable_energy_share * 1.02,  # Adjust growth rate (e.g., 2% annual increase)
    len(years)
)

# Check updated columns
print("Updated Future Data Columns (future_values):", future_values.columns)

# Step 4: Impute missing values (if necessary)
imputer = SimpleImputer(strategy='mean')
X_future_imputed = imputer.fit_transform(future_values.drop(columns=['Year']))


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Assuming df contains the historical data for the selected countries
selected_countries = ['Bahrain', 'Libya', 'Jordan', 'Iraq', 'Lebanon', 'United Arab Emirates', 'Kuwait', 'Oman', 'Qatar']
df_selected = df[df['Entity'].isin(selected_countries)]

# Step 1: Initialize Imputer to handle missing values
imputer = SimpleImputer(strategy='mean')

# Step 2: Define the years for future predictions
years = np.arange(2020, 2041)  # Years 2020 to 2040

# Step 3: Prepare the feature names
features = [
    'Electricity from fossil fuels (TWh)', 
    'Renewable energy share in the total final energy consumption (%)', 
    'gdp_per_capita', 
    'Electricity from renewables (TWh)', 
    'Primary energy consumption per capita (kWh/person)', 
    'Access to electricity (% of population)'
]

# Step 4: Train separate models for each country and predict future CO2 emissions
results = []

for country in selected_countries:
    country_data = df_selected[df_selected['Entity'] == country]
    
    # Filter out rows with NaN CO2 emissions (from 2020)
    country_data = country_data[country_data['Value_co2_emissions_kt_by_country'].notna()]
    
    # Check if there is enough data for training (at least a few data points)
    if len(country_data) > 1:
        # Step 5: Prepare the features (X) and target (y)
        X = country_data[features]
        y = country_data['Value_co2_emissions_kt_by_country']
        
        # Impute missing values in the features if necessary
        X_imputed = imputer.fit_transform(X)

        # Step 6: Train the MLR model for this country
        mlr_model = LinearRegression()
        mlr_model.fit(X_imputed, y)

        # Step 7: Prepare future feature values (for the years 2020-2040)
        last_values = country_data[features].iloc[-1]
        future_values = pd.DataFrame({
            'Year': years,
            'Electricity from fossil fuels (TWh)': np.linspace(last_values['Electricity from fossil fuels (TWh)'], last_values['Electricity from fossil fuels (TWh)'] * 1.05, len(years)),
            'Renewable energy share in the total final energy consumption (%)': np.linspace(last_values['Renewable energy share in the total final energy consumption (%)'], last_values['Renewable energy share in the total final energy consumption (%)'] * 1.02, len(years)),
            'gdp_per_capita': np.linspace(last_values['gdp_per_capita'], last_values['gdp_per_capita'] * 1.03, len(years)),
            'Electricity from renewables (TWh)': np.linspace(last_values['Electricity from renewables (TWh)'], last_values['Electricity from renewables (TWh)'] * 1.04, len(years)),
            'Primary energy consumption per capita (kWh/person)': np.linspace(last_values['Primary energy consumption per capita (kWh/person)'], last_values['Primary energy consumption per capita (kWh/person)'] * 1.02, len(years)),
            'Access to electricity (% of population)': np.linspace(last_values['Access to electricity (% of population)'], last_values['Access to electricity (% of population)'] * 1.01, len(years)),
        })
        
        # Impute missing values for the future data if necessary
        future_values_imputed = imputer.transform(future_values.drop(columns=['Year']))

        # Step 8: Predict CO2 emissions for the years 2020-2040
        predicted_co2 = mlr_model.predict(future_values_imputed)

        # Step 9: Prepare the results for this country
        country_predictions = pd.DataFrame({
            'Year': years,
            'Predicted CO2 Emissions (MtCO2)': predicted_co2,
            'Country': country
        })

        # Append the predictions to the results list
        results.append(country_predictions)

# Step 10: Combine all country predictions into a single DataFrame
final_predictions = pd.concat(results)

# Step 11: Save the predictions to a CSV file
final_predictions.to_csv('co2_emissions_predictions_by_country_2020_2040.csv', index=False)

# Display the final predictions
print(final_predictions)


# using gradiant boosting 

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer

# Assuming df contains the historical data for the selected countries
selected_countries = ['Bahrain', 'Libya', 'Jordan', 'Iraq', 'Lebanon', 'United Arab Emirates', 'Kuwait', 'Oman', 'Qatar']
df_selected = df[df['Entity'].isin(selected_countries)]

# Step 1: Initialize Imputer to handle missing values
imputer = SimpleImputer(strategy='mean')

# Step 2: Define the years for future predictions
years = np.arange(2020, 2041)  # Years 2020 to 2040

# Step 3: Prepare the feature names
features = [
    'Electricity from fossil fuels (TWh)', 
    'Renewable energy share in the total final energy consumption (%)', 
    'gdp_per_capita', 
    'Electricity from renewables (TWh)', 
    'Primary energy consumption per capita (kWh/person)', 
    'Access to electricity (% of population)'
]

# Step 4: Train separate models for each country and predict future CO2 emissions
results = []

for country in selected_countries:
    country_data = df_selected[df_selected['Entity'] == country]
    
    # Filter out rows with NaN CO2 emissions (from 2020)
    country_data = country_data[country_data['Value_co2_emissions_kt_by_country'].notna()]
    
    # Check if there is enough data for training (at least a few data points)
    if len(country_data) > 1:
        # Step 5: Prepare the features (X) and target (y)
        X = country_data[features]
        y = country_data['Value_co2_emissions_kt_by_country']
        
        # Impute missing values in the features if necessary
        X_imputed = imputer.fit_transform(X)

        # Step 6: Train the Gradient Boosting model for this country
        gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
        gb_model.fit(X_imputed, y)

        # Step 7: Prepare future feature values (for the years 2020-2040)
        last_values = country_data[features].iloc[-1]
        future_values = pd.DataFrame({
            'Year': years,
            'Electricity from fossil fuels (TWh)': np.linspace(last_values['Electricity from fossil fuels (TWh)'], last_values['Electricity from fossil fuels (TWh)'] * 1.05, len(years)),
            'Renewable energy share in the total final energy consumption (%)': np.linspace(last_values['Renewable energy share in the total final energy consumption (%)'], last_values['Renewable energy share in the total final energy consumption (%)'] * 1.02, len(years)),
            'gdp_per_capita': np.linspace(last_values['gdp_per_capita'], last_values['gdp_per_capita'] * 1.03, len(years)),
            'Electricity from renewables (TWh)': np.linspace(last_values['Electricity from renewables (TWh)'], last_values['Electricity from renewables (TWh)'] * 1.04, len(years)),
            'Primary energy consumption per capita (kWh/person)': np.linspace(last_values['Primary energy consumption per capita (kWh/person)'], last_values['Primary energy consumption per capita (kWh/person)'] * 1.02, len(years)),
            'Access to electricity (% of population)': np.linspace(last_values['Access to electricity (% of population)'], last_values['Access to electricity (% of population)'] * 1.01, len(years)),
        })
        
        # Impute missing values for the future data if necessary
        future_values_imputed = imputer.transform(future_values.drop(columns=['Year']))

        # Step 8: Predict CO2 emissions for the years 2020-2040
        predicted_co2 = gb_model.predict(future_values_imputed)

        # Step 9: Prepare the results for this country
        country_predictions = pd.DataFrame({
            'Year': years,
            'Predicted CO2 Emissions (MtCO2)': predicted_co2,
            'Country': country
        })

        # Append the predictions to the results list
        results.append(country_predictions)

# Step 10: Combine all country predictions into a single DataFrame
final_predictions = pd.concat(results)

# Step 11: Save the predictions to a CSV file
final_predictions.to_csv('co2_emissions_predictions_gradient_boosting_by_country_2020_2040.csv', index=False)

# Display the final predictions
print(final_predictions)


#using a nn and predicting values with 

selected_countries = ['Bahrain', 'Libya', 'Jordan', 'Iraq', 'Lebanon', 'United Arab Emirates', 'Kuwait', 'Oman', 'Qatar']
df_selected = df[df['Entity'].isin(selected_countries)]

# Filter out rows for the year 2020 where 'CO2 emissions' is NaN
df_selected = df_selected[df_selected['Value_co2_emissions_kt_by_country'].notna()]

# Define the features and target
X = df_selected[['Electricity from fossil fuels (TWh)', 'Renewable energy share in the total final energy consumption (%)',
                 'gdp_per_capita', 'Electricity from renewables (TWh)',
                 'Primary energy consumption per capita (kWh/person)', 'Access to electricity (% of population)']]
y = df_selected['Value_co2_emissions_kt_by_country']

# Impute missing values if necessary
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Scale features for better NN performance
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X_imputed)

# Scale the target variable y (CO2 emissions)
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))  # Reshape y to 2D for scaling

# Train a separate Neural Network model for each country
results = []

# Iterate through each country and train a model
for country in selected_countries:
    country_data = df_selected[df_selected['Entity'] == country]
    
    # Define features and target for this country's model
    X_country = country_data[['Electricity from fossil fuels (TWh)', 'Renewable energy share in the total final energy consumption (%)',
                              'gdp_per_capita', 'Electricity from renewables (TWh)',
                              'Primary energy consumption per capita (kWh/person)', 'Access to electricity (% of population)']]
    y_country = country_data['Value_co2_emissions_kt_by_country']
    
    # Impute and scale the country's data
    X_country_imputed = imputer.transform(X_country)
    X_country_scaled = scaler_X.transform(X_country_imputed)
    
    # Scale the target variable for this country
    y_country_scaled = scaler_y.transform(y_country.values.reshape(-1, 1))
    
    # Train the Neural Network model
    nn_model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    nn_model.fit(X_country_scaled, y_country_scaled.ravel())  # Fit the model on scaled target
    
    # Predict CO2 emissions for the years 2020-2040
    future_values = np.array([np.linspace(X_country[col].min(), X_country[col].max(), num=21) for col in X_country.columns]).T
    future_values_imputed = imputer.transform(future_values)
    future_values_scaled = scaler_X.transform(future_values_imputed)
    
    # Predict future CO2 emissions (scaled)
    predicted_co2_scaled = nn_model.predict(future_values_scaled)
    
    # Inverse transform the predictions back to the original scale of CO2 emissions
    predicted_co2 = scaler_y.inverse_transform(predicted_co2_scaled.reshape(-1, 1))
    
    # Store results for this country
    country_results = pd.DataFrame({
        'Year': np.arange(2020, 2041),
        'Predicted CO2 Emissions (kt)': predicted_co2.flatten(),  # Flatten to get a 1D array
        'Country': country
    })
    
    results.append(country_results)

# Combine all results into one DataFrame
final_predictions = pd.concat(results)

# Save results to CSV
final_predictions.to_csv('predicted_co2_emissions_2020_2040_nn.csv', index=False)

# Display the final predictions
print(final_predictions)


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer

# Assuming df is the data frame containing the historical data for selected countries
# countries = ['Bahrain', 'Libya', 'Jordan', 'Iraq', 'Lebanon', 'United Arab Emirates', 'Kuwait', 'Oman', 'Qatar']
selected_countries = ['Bahrain', 'Libya', 'Jordan', 'Iraq', 'Lebanon', 'United Arab Emirates', 'Kuwait', 'Oman', 'Qatar']
df_selected = df[df['Entity'].isin(selected_countries)]

# Filter out rows for the year 2020 where 'CO2 emissions' is NaN
df_selected = df_selected[df_selected['Value_co2_emissions_kt_by_country'].notna()]

# Define the features and target
X = df_selected[['Electricity from fossil fuels (TWh)', 'Renewable energy share in the total final energy consumption (%)',
                 'gdp_per_capita', 'Electricity from renewables (TWh)',
                 'Primary energy consumption per capita (kWh/person)', 'Access to electricity (% of population)']]
y = df_selected['Value_co2_emissions_kt_by_country']

# Impute missing values if necessary
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# **Scale the "Electricity from fossil fuels (TWh)" feature** by multiplying it with a scaling factor
scaling_factor = 2  # Increase the weight of this feature
X_imputed[:, 0] *= scaling_factor  # Multiply the "Electricity from fossil fuels (TWh)" column by scaling factor

# Scale the features for better Neural Network performance
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X_imputed)

# Scale the target variable y (CO2 emissions)
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))  # Reshape y to 2D for scaling

# Train a separate Neural Network model for each country
results = []

# Iterate through each country and train a model
for country in selected_countries:
    country_data = df_selected[df_selected['Entity'] == country]
    
    # Define features and target for this country's model
    X_country = country_data[['Electricity from fossil fuels (TWh)', 'Renewable energy share in the total final energy consumption (%)',
                              'gdp_per_capita', 'Electricity from renewables (TWh)',
                              'Primary energy consumption per capita (kWh/person)', 'Access to electricity (% of population)']]
    y_country = country_data['Value_co2_emissions_kt_by_country']
    
    # Impute and scale the country's data
    X_country_imputed = imputer.transform(X_country)
    X_country_imputed[:, 0] *= scaling_factor  # Apply the same scaling factor to "Electricity from fossil fuels" for this country
    X_country_scaled = scaler_X.transform(X_country_imputed)
    
    # Scale the target variable for this country
    y_country_scaled = scaler_y.transform(y_country.values.reshape(-1, 1))
    
    # Train the Neural Network model
    nn_model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    nn_model.fit(X_country_scaled, y_country_scaled.ravel())  # Fit the model on scaled target
    
    # Predict CO2 emissions for the years 2020-2040
    future_values = np.array([np.linspace(X_country[col].min(), X_country[col].max(), num=21) for col in X_country.columns]).T
    future_values_imputed = imputer.transform(future_values)
    future_values_imputed[:, 0] *= scaling_factor  # Apply the scaling factor to future values of "Electricity from fossil fuels"
    future_values_scaled = scaler_X.transform(future_values_imputed)
    
    # Predict future CO2 emissions (scaled)
    predicted_co2_scaled = nn_model.predict(future_values_scaled)
    
    # Inverse transform the predictions back to the original scale of CO2 emissions
    predicted_co2 = scaler_y.inverse_transform(predicted_co2_scaled.reshape(-1, 1))
    
    # Store results for this country
    country_results = pd.DataFrame({
        'Year': np.arange(2020, 2041),
        'Predicted CO2 Emissions (kt)': predicted_co2.flatten(),  # Flatten to get a 1D array
        'Country': country
    })
    
    results.append(country_results)

# Combine all results into one DataFrame
final_predictions = pd.concat(results)

# Save results to CSV
final_predictions.to_csv('predicted_co2_emissions_2020_2040_nn_fossil_scaling_v2.csv', index=False)

# Display the final predictions
print(final_predictions)


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer

# Ensure the selected countries list is consistent with the data
selected_countries = ['Bahrain', 'Libya', 'Jordan', 'Iraq', 'Lebanon', 
                      'United Arab Emirates', 'Kuwait', 'Oman', 'Qatar']

# Filter the dataset for the selected countries
df_selected = df[df['Entity'].isin(selected_countries)]

# Debugging: Ensure the filtered dataset is not empty
print("Number of rows in df_selected:", len(df_selected))
if df_selected.empty:
    raise ValueError("The filtered DataFrame 'df_selected' is empty. Check your filtering criteria.")

# Filter out rows where 'Value_co2_emissions_kt_by_country' is NaN
df_selected = df_selected[df_selected['Value_co2_emissions_kt_by_country'].notna()]

# Define the features and target
X = df_selected[['Electricity from fossil fuels (TWh)', 
                 'Renewable energy share in the total final energy consumption (%)', 
                 'gdp_per_capita', 
                 'Electricity from renewables (TWh)', 
                 'Primary energy consumption per capita (kWh/person)', 
                 'Access to electricity (% of population)']]
y = df_selected['Value_co2_emissions_kt_by_country']

# Debugging: Ensure the features (X) and target (y) have data
print("Columns in X:", X.columns)
print("Number of rows in X:", len(X))
print("Missing values in X:\n", X.isnull().sum())
print("Number of rows in y:", len(y))

# Handle missing values with SimpleImputer
if X.empty:
    raise ValueError("The feature matrix 'X' is empty after filtering. Check your data or filtering criteria.")
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Debugging: Check shape after imputation
print("Shape of X_imputed:", X_imputed.shape)

# Scale the features for better Neural Network performance
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X_imputed)

# Scale the target variable y (CO2 emissions)
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))  # Reshape y to 2D for scaling

# Train the Neural Network model
nn_model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
nn_model.fit(X_scaled, y_scaled.ravel())  # Fit the model on scaled data

# Debugging: Output model performance on the training data
y_pred_scaled = nn_model.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Neural Network Performance with Increased Feature Weights:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

# Predict future CO2 emissions for 2020-2040
future_years = np.arange(2020, 2041)

# Generate future feature values
future_values = np.array([
    np.linspace(X.iloc[:, col].min(), X.iloc[:, col].max(), len(future_years)) 
    for col in range(X.shape[1])
]).T  # Transpose to match the required shape

# Apply increased weights to fossil fuels and renewables
future_values[:, 0] *= 2.0  # Increase weight for fossil fuels
future_values[:, 1] *= 1.5  # Increase weight for renewables

# Scale and predict future emissions
future_values_scaled = scaler_X.transform(future_values)
future_predictions_scaled = nn_model.predict(future_values_scaled)
future_predictions = scaler_y.inverse_transform(future_predictions_scaled.reshape(-1, 1))

# Store predictions in a DataFrame
future_results = pd.DataFrame({
    'Year': future_years,
    'Predicted CO2 Emissions (kt)': future_predictions.flatten()
})

# Display and plot the results
print("Predicted CO2 Emissions for 2020-2040 with Adjusted Weights:")
print(future_results)

# Plot the predictions
plt.figure(figsize=(10, 6))
plt.plot(future_results['Year'], future_results['Predicted CO2 Emissions (kt)'], label="Predicted CO2 Emissions")
plt.title("CO2 Emissions Prediction (2020-2040) with Adjusted Feature Weights")
plt.xlabel("Year")
plt.ylabel("CO2 Emissions (kt)")
plt.legend()
plt.show()
