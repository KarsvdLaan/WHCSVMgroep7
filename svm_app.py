
import pandas as pd
import streamlit as st
from sklearn.svm import SVR
import joblib
import os

# Bestandsnamen
csv_file = "piemol csv.csv"
model_file = "svm_model.joblib"

st.title("🌿 SVM Voorspelling van Temperatuur in de Kas")

# Inlezen van de dataset
df = pd.read_csv(csv_file, delimiter=";", quotechar='"')
X = df.drop(columns=["prediction(Tempratuur in de Kas)"])
y = df["prediction(Tempratuur in de Kas)"]

# Laden of trainen van het model
if os.path.exists(model_file):
    model = joblib.load(model_file)
    st.success("✅ Bestaand model geladen.")
else:
    model = SVR(kernel='rbf')
    model.fit(X, y)
    joblib.dump(model, model_file)
    st.success("✅ Model getraind en opgeslagen.")

# Interactieve invoer van gebruiker
st.header("📥 Voer inputwaarden in om voorspelling te doen:")

user_input = []
for column in X.columns:
    value = st.number_input(f"{column}", value=float(X[column].mean()))
    user_input.append(value)

# Voorspelling uitvoeren
if st.button("Voorspel"):
    prediction = model.predict([user_input])[0]
    st.subheader(f"🌡️ Voorspelde temperatuur in de kas: {prediction:.2f} °C")
