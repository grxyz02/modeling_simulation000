import streamlit as st
import random
import joblib
import plotly.express as px

def load_files():
    st.sidebar.title("Upload Model Files")
    model_file = st.sidebar.file_uploader("Upload Model (.pkl)", type="pkl")
    scaler_file = st.sidebar.file_uploader("Upload Scaler (.pkl)", type="pkl")

    if model_file and scaler_file:
        try:
            model = joblib.load(model_file)
            scaler = joblib.load(scaler_file)
            st.sidebar.success("Model and scaler loaded successfully!")
            return model, scaler
        except Exception as e:
            st.sidebar.error(f"Error loading files: {e}")
            return None, None
    else:
        st.sidebar.info("Please upload both model and scaler files to proceed.")
        return None, None

def predict_and_visualize(model, scaler, input_features):
    try:
        # Preprocess the inputs
        input_array = [input_features]
        input_scaled = scaler.transform(input_array)

        # Make predictions
        prediction = model.predict(input_scaled)
        probabilities = model.predict_proba(input_scaled)[0]
        class_labels = model.classes_

        # Display results
        st.subheader("Prediction Results")
        st.write(f"**Predicted Class:** {prediction[0]}")
        prob_fig = px.bar(
            x=class_labels,
            y=probabilities,
            labels={"x": "Class", "y": "Probability"},
            title="Class Probabilities",
        )
        st.plotly_chart(prob_fig)
    except Exception as e:
        st.error(f"Error during prediction: {e}")

def main():
    st.title("ML Model Implementation")
    model, scaler = load_files()
    if not model or not scaler:
        return

    feature_names = ["length (mm)", "width (mm)", "density (g/cm³)", "pH"]
    input_features = []

    st.subheader("Enter Feature Values")
    col1, col2 = st.columns([2, 3])  # Left: inputs, Right: outputs
    with col1:
        if st.button("🎲 Randomize"):
            # Randomize input features using the `random` module
            random_values = [random.uniform(0, 100) for _ in feature_names]
            for i, value in enumerate(random_values):
                st.session_state[f"input_{i}"] = value
            st.session_state["randomized"] = True

        # Feature Input Section
        for idx, feature in enumerate(feature_names):
            value = st.number_input(
                label=feature,
                min_value=0.0,
                max_value=1000.0,
                value=st.session_state.get(f"input_{idx}", 0.0),
                step=0.1,
                key=f"input_{idx}"
            )
            input_features.append(value)

        if st.button("Make Prediction"):
            st.session_state["make_prediction"] = True

    with col2:
        # Handle Randomization Results
        if "randomized" in st.session_state and st.session_state["randomized"]:
            st.subheader("Randomized Input Prediction")
            predict_and_visualize(model, scaler, input_features)
            st.session_state["randomized"] = False  

        # Handle Manual Prediction Results
        if st.session_state.get("make_prediction"):
            st.subheader("Manual Input Prediction")
            predict_and_visualize(model, scaler, input_features)
            st.session_state["make_prediction"] = False  
            
if __name__ == "__main__":
    main()
