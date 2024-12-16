import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Title Section
st.title("ML Model Generator")

# Sidebar Section
with st.sidebar:
    # Data Source Section
    st.header("Data Source")
    data_source = st.radio("Choose data source:", ["Generate Synthetic Data", "Upload Dataset"])

    if data_source == "Generate Synthetic Data":
        st.subheader("Synthetic Data Generation")
        st.write("Define parameters for synthetic data generation below.")
        
        # Data Generation Parameters Section
        st.subheader("Data Generation Parameters")
        
        # Input for feature names
        features_input = st.text_input("Enter feature names (comma-separated)", "length (mm), width (mm), density (g/cm³)")
        features = [f.strip() for f in features_input.split(",")]

        # Input for class names
        classes_input = st.text_input("Enter class names (comma-separated)", "Ampalaya, Banana, Cabbage")
        classes = [c.strip() for c in classes_input.split(",")]

        # Class-Specific Settings
        st.subheader("Class-Specific Settings")
        class_data = []  # List to store data for all classes

        for class_name in classes:
            with st.expander(f"{class_name} Settings", expanded=False):
                st.checkbox(f"Set specific values for {class_name}", value=True)

                # Inputs for mean and std dev of each feature
                mean_values = [st.number_input(f"Mean for {feature} ({class_name})", value=100.0) for feature in features]
                std_values = [st.number_input(f"Std Dev for {feature} ({class_name})", value=10.0) for feature in features]

                # Input for number of samples
                num_samples = st.number_input(f"Number of samples for {class_name}", value=250, min_value=1)
                data = np.random.normal(loc=mean_values, scale=std_values, size=(num_samples, len(features)))
                class_labels = np.full((num_samples, 1), class_name)  # Class label column
                class_data.append(np.hstack([data, class_labels]))  # Combine data and labels

    # Button to Generate Data and Train Model
    generate_data_button = st.button("Generate Data and Train Model")

# Main Section - Output Area
if generate_data_button:
    try:
        # Combine data for all classes
        all_data = np.vstack(class_data)

        # Extract feature data and labels
        feature_data = all_data[:, :-1]
        labels = all_data[:, -1]

        # Check for feature mismatch
        if feature_data.shape[1] != len(features):
            st.error(f"Mismatch: Data has {feature_data.shape[1]} columns but {len(features)} features.")
            features = [f"Feature {i+1}" for i in range(feature_data.shape[1])]
            st.warning(f"Adjusted features: {features}")

        # Create DataFrame for class data
        class_df = pd.DataFrame(feature_data, columns=features)
        class_df['Class'] = labels

        # Dataset Split Information
        total_samples = len(class_df)
        train_samples = int(0.8 * total_samples)
        test_samples = total_samples - train_samples

        st.subheader("Dataset Split Information")
        st.write(f"Total Samples: {total_samples}")
        st.write(f"Training Samples: {train_samples}")
        st.write(f"Testing Samples: {test_samples}")

        # Split the data
        train_data = class_df.sample(train_samples, random_state=42)
        test_data = class_df.drop(train_data.index)

        # Scale the feature data using MinMaxScaler
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(class_df[features])
        scaled_df = pd.DataFrame(scaled_data, columns=features)
        scaled_df['Class'] = labels

        # Display Generated Data
        st.subheader("Generated Data Sample")
        st.write("Original Data:")
        st.dataframe(class_df)  # Display original data
        st.write("Scaled Data:")
        st.dataframe(scaled_df)  # Display scaled data

        # Feature Visualization
        st.subheader("Feature Visualization")
        visualization_type = st.radio("Select Visualization Type", ["2D", "3D"])

        if visualization_type == "2D":
            x_feature = st.selectbox("Select X-Axis Feature", features)
            y_feature = st.selectbox("Select Y-Axis Feature", features)
            plt.figure(figsize=(10, 6))
            for class_name in classes:
                subset = class_df[class_df['Class'] == class_name]
                plt.scatter(subset[x_feature], subset[y_feature], label=class_name, alpha=0.6)
            plt.xlabel(x_feature)
            plt.ylabel(y_feature)
            plt.legend()
            st.pyplot(plt)

        elif visualization_type == "3D":
            x_feature = st.selectbox("Select X-Axis Feature", features, key="3d_x")
            y_feature = st.selectbox("Select Y-Axis Feature", features, key="3d_y")
            z_feature = st.selectbox("Select Z-Axis Feature", features, key="3d_z")
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            for class_name in classes:
                subset = class_df[class_df['Class'] == class_name]
                ax.scatter(subset[x_feature], subset[y_feature], subset[z_feature], label=class_name, alpha=0.6)
            ax.set_xlabel(x_feature)
            ax.set_ylabel(y_feature)
            ax.set_zlabel(z_feature)
            ax.legend()
            st.pyplot(fig)

    except ValueError as e:
        st.error(f"Error: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")