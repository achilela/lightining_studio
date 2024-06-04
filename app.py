import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_name = "amiguel/fintune_naming_model"  # Replace with your model repo
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def classify_review(text, model, tokenizer, device, max_length=512):
    model.eval()
    inputs = tokenizer.encode_plus(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors="pt"
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=-1).item()
    return "Proper Naming otfcn" if predicted_label == 1 else "Wrong Naming notfcn"

def main():
    st.title("Notifications Naming Classifier")

    input_option = st.radio("Select input option", ("Single Text Query", "Upload Table"))

    if input_option == "Single Text Query":
        text_query = st.text_input("Enter text query")
        if st.button("Classify"):
            if text_query:
                predicted_label = classify_review(text_query, model, tokenizer, device)
                st.write("Predicted Label:")
                st.write(predicted_label)
            else:
                st.warning("Please enter a text query.")

    elif input_option == "Upload Table":
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
        if uploaded_file is not None:
            import pandas as pd
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            text_column = st.selectbox("Select the text column", df.columns)
            predicted_labels = [classify_review(text, model, tokenizer, device) for text in df[text_column]]
            df["Predicted Label"] = predicted_labels
            st.write(df) 

if __name__ == "__main__":
    main()
