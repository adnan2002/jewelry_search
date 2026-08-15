import io

import streamlit as st
import chromadb

from PIL import Image

import numpy as np

import onnxruntime as ort

from sklearn.metrics.pairwise import cosine_similarity

IMG_SIZE = (224, 224)
TOP_K = 25
MODEL_PATH = "model/embedding_model.onnx"
CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "image_embeddings"


@st.cache_resource
def load_embedding_model():
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

    return session


@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    return collection


def get_embedding(embedding_model, image):
    image = image.convert("RGB").resize(IMG_SIZE)

    image = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    image = np.expand_dims(image, axis=0)

    embedding = embedding_model.run(None, {"input_1": image})[0]

    return embedding


def main():
    st.title("Jewelry Search")

    embedding_model = load_embedding_model()
    collection = load_collection()

    source = st.radio(
        "Image source",
        ["Upload image", "Take a photo"],
    )

    image = None

    if source == "Upload image":
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
    else:
        camera_image = st.camera_input("Take a photo")
        if camera_image is not None:
            image = Image.open(io.BytesIO(camera_image.getvalue()))

    if image is None:
        st.stop()

    st.image(image, caption="Query image")

    threshold = st.slider("Minimum similarity (%)", 0, 100, 60) / 100.0

    embedding = get_embedding(embedding_model, image)

    results = collection.query(
        query_embeddings=embedding.tolist(),
        n_results=TOP_K,
        include=["embeddings", "metadatas"],
    )

    doc_embeddings = np.array(results["embeddings"][0])
    metadatas = results["metadatas"][0]

    similarities = cosine_similarity(embedding, doc_embeddings)[0]
    order = np.argsort(similarities)[::-1]

    matches = [
        (metadatas[i], similarities[i])
        for i in order
        if similarities[i] >= threshold
    ]

    if not matches:
        st.warning("No similar items found. Try another image or lower the threshold.")
        return

    st.subheader(f"Top {len(matches)} matches")

    for row_start in range(0, len(matches), 5):
        cols = st.columns(5)

        for col, (metadata, similarity) in zip(cols, matches[row_start:row_start + 5]):
            path = metadata["path"]
            filename = path.split("/")[-1]

            col.image(path)
            col.caption(f"{filename} - {similarity * 100:.2f}%")


if __name__ == "__main__":
    main()
