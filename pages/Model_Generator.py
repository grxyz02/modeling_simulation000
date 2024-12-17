import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
import io
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


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

        # Store mean and std values for each class
        mean_values_dict = {}
        std_values_dict = {}

        for class_name in classes:
            with st.expander(f"{class_name} Settings", expanded=False):
                st.checkbox(f"Set specific values for {class_name}", value=True)

                col1, col2 = st.columns(2)
                with col1:
                    mean_values_dict[class_name] = [
                        int(st.number_input(f"Mean for {feature} ({class_name})", value=100.0)) for feature in features
                    ]
                with col2:
                    std_values_dict[class_name] = [
                        int(st.number_input(f"Std Dev for {feature} ({class_name})", value=10.0)) for feature in features
                    ]

        st.subheader("Sample Size & Train/Test Split Configuration")   
        
        col1, col2 = st.columns(2)
        with col1:
            total_sample_size = st.slider(
                "Number of samples", 
                max_value = 50000, 
                min_value=500,
                step=500
            )
        with col2:
            train_test_split_percent = st.slider(
                "Train-Test Split (%)",
                min_value=10,
                max_value=50,
                step=5
            )

        samples_per_class = total_sample_size // len(classes)
        
        # Generate synthetic data for each class
        for class_name in classes:
            mean_values = mean_values_dict[class_name]
            std_values = std_values_dict[class_name]
            data = np.random.normal(
                loc=mean_values,
                scale=std_values,
                size=(samples_per_class, len(features))
            )
            class_labels = np.full((samples_per_class, 1), class_name)  # Class label column
            class_data.append(np.hstack([data, class_labels]))  # Combine data and labels

    generate_data_button = st.button("Generate Data and Train Model")

# Main Section - Output Area
if generate_data_button or 'generated' not in st.session_state:
    try:
        all_data = np.vstack(class_data)
        np.random.shuffle(all_data)

        # Split data into train and test
        train_size = train_test_split_percent / 100
        #train_data, test_data = train_test_split(all_data, test_size=1 - train_size, stratify=all_data[:, -1], random_state=42)

        # Extract feature data and labels
        feature_data = all_data[:, :-1]
        labels = all_data[:, -1]

        # Create DataFrame for class data
        class_df = pd.DataFrame(feature_data, columns=features)
        class_df['Target'] = labels

        # Total sample size from user input
        total_samples = int(total_sample_size)
        train_samples = int((train_test_split_percent / 100) * total_samples)
        test_samples = total_samples - train_samples

        st.subheader("Dataset Split Information")
        col1, col2, col3 = st.columns(3)

        # Total Samples Column
        with col1:
            st.markdown("Total Samples")
            st.subheader(total_samples)

        # Training Samples Column
        with col2:
            st.markdown("Training Samples")
            st.subheader(f"{test_samples} ({100 - train_test_split_percent}%)")
        
        # Testing Samples Column
        with col3:
            st.markdown("Testing Samples") 
            st.subheader(f"{train_samples} ({train_test_split_percent}%)")
            


        # Scale the feature data using MinMaxScaler
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(class_df[features])  # Scale only the feature columns
        scaled_df = pd.DataFrame(scaled_data, columns=features)
        scaled_df['Target'] = labels  # Keep the 'Target' column with class labels



        st.subheader("Generated Data Sample")

        # Define the column widths
        col1, col2 = st.columns([10, 10])  # Both columns take equal width

        with col1:
            st.write("Original Data (Random samples from each class):")
            # Adjust the dataframe width to fit the container width and take up more space
            st.dataframe(class_df, use_container_width=True)

        with col2:
            st.write("Scaled Data (using best model's scaler):")
            # Adjust the dataframe width to fit the container width and take up more space
            st.dataframe(scaled_df, use_container_width=True)


        # Feature Visualization

        st.subheader("Feature Visualization")
        features = class_df.columns[:-1]  # Exclude 'Target' for plotting

        # Convert all features to numeric, coercing errors
        for feature in features:
            class_df[feature] = pd.to_numeric(class_df[feature], errors='coerce')

        # List of unique class labels
        classes = class_df['Target'].unique()

        # Select visualization type
        visualization_type = st.radio("Select Visualization Type", ["2D", "3D"])

        if visualization_type == "2D":
            # Dropdowns to select features for X and Y axes
            col1, col2 = st.columns(2)
            with col1:
                x_feature = st.selectbox("Select X-Axis Feature", features)
            with col2:    
                y_feature = st.selectbox("Select Y-Axis Feature", features)

            # Check if selected features are numeric, or convert if necessary
            if pd.to_numeric(class_df[x_feature], errors='coerce').isnull().any() or pd.to_numeric(class_df[y_feature], errors='coerce').isnull().any():
                st.error("Selected features should be numeric for scatter plot.")
            else:
                # Create the 2D scatter plot using Plotly Express
                fig = px.scatter(
                    class_df, 
                    x=x_feature, 
                    y=y_feature, 
                    color='Target',  # Color points by class
                    title=f"Scatter Plot of {x_feature} vs {y_feature}",
                    labels={x_feature: x_feature, y_feature: y_feature}
                )
                # Display the plot in the Streamlit app
                st.plotly_chart(fig, use_container_width=True)


        elif visualization_type == "3D":
            # Dropdowns to select features for X, Y, and Z axes
            col1, col2, col3 = st.columns(3)
            with col1:
                x_feature = st.selectbox("Select X-Axis Feature", features, key="x_3d")
            with col2:
                y_feature = st.selectbox("Select Y-Axis Feature", features, key="y_3d")
            with col3:
                z_feature = st.selectbox("Select Z-Axis Feature", features, key="z_3d")

    # Ensure the selected features are numeric before plotting
            if (
                pd.to_numeric(class_df[x_feature], errors='coerce').isnull().any() or
                pd.to_numeric(class_df[y_feature], errors='coerce').isnull().any() or
                pd.to_numeric(class_df[z_feature], errors='coerce').isnull().any()
            ):
                st.error("Selected features should be numeric for 3D scatter plot.")
            else:
        # Create the 3D scatter plot using Plotly Express
                fig = px.scatter_3d(
                    class_df,
                    x=x_feature,
                    y=y_feature,
                    z=z_feature,
                    color='Target',  # Color points by class
                    title=f"3D Scatter Plot of {x_feature}, {y_feature}, {z_feature}",
                    labels={x_feature: x_feature, y_feature: y_feature, z_feature: z_feature}
                )
                # Display the plot in the Streamlit app
                st.plotly_chart(fig, use_container_width=True)

        #Download Dataset
        st.subheader("Download Dataset")

        # Function to convert DataFrame to CSV
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        # Generate CSV files for original and scaled datasets
        original_csv = convert_df_to_csv(class_df)
        scaled_csv = convert_df_to_csv(scaled_df)

        # Create download buttons
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download Original Dataset (CSV)",
                data=original_csv,
                file_name="original_dataset.csv",
                mime="text/csv"
            )
        with col2:
            st.download_button(
                label="Download Scaled Dataset (CSV)",
                data=scaled_csv,
                file_name="scaled_dataset.csv",
                mime="text/csv"
            )

        with st.expander("Dataset Statistics"):
            st.subheader("Dataset Statistics Overview")

         # Create columns for Original and Scaled dataset statistics
            col1, col2 = st.columns(2)

            # Display statistics for the Original dataset in Column 1
            with col1:
                st.write("**Original Dataset**")
                st.dataframe(class_df.describe())

            # Display statistics for the Scaled dataset in Column 2
            with col2:
                st.write("**Scaled Dataset**")
                st.dataframe(scaled_df.describe())

        


        #TRIAL ... (TRAIN)
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(scaled_data, labels, test_size=1-train_size, stratify=labels)

        # Models
        models = {
            "Random Forest": RandomForestClassifier(),
            "Logistic Regression": LogisticRegression(),
            "SVM": SVC()
        }

        best_model = None
        best_score = 0
        model_results = {}

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            model_results[model_name] = {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1
            }

            if accuracy > best_score:
                best_score = accuracy
                best_model = model

        best_model.fit(X_train, y_train)  # Fit the best model
        y_pred = best_model.predict(X_test)  # Get predictions from the best model

        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        # Show model comparison
        st.subheader("Best Model Performance")
        st.write(f"Best Model: {best_model.__class__.__name__}")
        st.write(f"Accuracy: {best_score:.4f}")
        
        report_df = pd.DataFrame(report).transpose()  # Convert to DataFrame and transpose for easier viewing
        st.dataframe(report_df)

        # Show model comparison
        st.subheader("Model Comparison")
        model_comparison_df = pd.DataFrame(model_results).T
        st.dataframe(model_comparison_df)

        

        # Display confusion matrix
        #st.subheader("Confusion Matrix")
        #y_pred_best = best_model.predict(X_test)
        #cm = confusion_matrix(y_test, y_pred_best)
        #fig, ax = plt.subplots()
        #ax.matshow(cm, cmap='Blues')
        #st.pyplot(fig)

        # Save and download the best model
        st.subheader("Download Best Model")

        model_filename = "best_model.pkl"
        joblib.dump(best_model, model_filename)

        with open(model_filename, "rb") as f:
            st.download_button(
                label="Download Model",
                data=f,
                file_name=model_filename,
                mime="application/octet-stream"
            )


    except ValueError as e:
        st.error(f"Error: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")