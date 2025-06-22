# app.py
import streamlit as st
import torch
from model import DigitGeneratorNet
import matplotlib.pyplot as plt

st.set_page_config(page_title="Digit Generator", layout="centered")

# Load model
device = torch.device("cpu")
model = DigitGeneratorNet()
model.load_state_dict(torch.load("generator.pth", map_location=device))
model.eval()

st.title("MNIST Digit Image Generator")
digit = st.selectbox("Choose a digit (0-9)", list(range(10)))

if st.button("Generate Images"):
    z = torch.randn(5, 64)
    labels = torch.tensor([digit] * 5)
    with torch.no_grad():
        images = model(z, labels).squeeze().numpy()

    cols = st.columns(5)
    for i in range(5):
        fig, ax = plt.subplots()
        img = images[i].squeeze()  # Shape becomes (28, 28)
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        st.pyplot(fig)
