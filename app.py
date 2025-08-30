import streamlit as st
import joblib
import numpy as np
import pandas as pd


st.set_page_config(page_title="Normal vs Abnormal Classifier", 
                   page_icon="🧠", 
                   layout="wide")


model = joblib.load("classifier.pkl")


st.sidebar.title("🔎 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict", "Upload CSV", "About"])



if page == "Home":
    st.title("🧠 Normal vs Abnormal Classifier")
    st.markdown("""
    Welcome to the **Normal vs Abnormal Classifier App**!  
    - 🔮 Predict whether a patient is **Normal** or **Abnormal**  
    - 📂 Upload CSV files for batch predictions  
    - 📊 Visualize results easily  

    Built with ❤️ using **Streamlit** + **Machine Learning**.
    """)
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=250)



elif page == "Predict":
    st.header("🔮 Make a Prediction")

    st.write("Enter patient spine data below:")

    # Your 6 features
    feature1 = st.number_input("Enter Pelvic Incidence")
    feature2 = st.number_input("Enter Pelvic Tilt")
    feature3 = st.number_input("Enter Lumbar Lordosis Angle")
    feature4 = st.number_input("Enter Sacral Slope")
    feature5 = st.number_input("Enter Pelvic Radius")
    feature6 = st.number_input("Enter Grade of Spondylolisthesis")

    if st.button("Predict"):
        input_data = np.array([[feature1, feature2, feature3, feature4, feature5, feature6]])
        prediction = model.predict(input_data)[0]

        st.subheader("Result:")
        if prediction == 1:
            st.metric("Prediction", "Abnormal ❌", delta="-1")
            st.error("⚠️ Patient is **Abnormal**")
        else:
            st.metric("Prediction", "Normal ✅", delta="+1")
            st.success("🎉 Patient is **Normal**")



elif page == "Upload CSV":
    st.header("📂 Upload CSV for Batch Prediction")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("✅ File uploaded successfully!")
        st.dataframe(data.head())

        # Predictions
        preds = model.predict(data)
        data["Prediction"] = preds

        # Show results
        st.subheader("🔮 Predictions:")
        st.dataframe(data)

        # Count Normal vs Abnormal
        st.subheader("📊 Prediction Distribution")
        counts = data["Prediction"].value_counts().rename({0: "Normal", 1: "Abnormal"})
        st.bar_chart(counts)

        # Download option
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Predictions", csv, "predictions.csv", "text/csv")



elif page == "About":
    st.header("ℹ️ About This App")
    st.markdown("""
    This is a **Machine Learning Web App** built with **Streamlit**.  
    - 👨‍💻 Developer: Athar Sharma  
    - 📊 Purpose: Classify patients as **Normal (0)** or **Abnormal (1)**  
    - ⚡ Features:
        - Manual input prediction (6 medical features)
        - Batch CSV prediction
        - Prediction distribution chart
        - Clean UI with sidebar navigation
    """)
    st.success("Made with ❤️ by Athar Sharma")
