import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import chromadb

import numpy as np







def main():
        
    dataset = tf.keras.utils.image_dataset_from_directory(
        "data/Jewellery_Data",
        image_size=(224, 224),
        batch_size=32,
        shuffle=False
    )

    base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
    )
    
    embedding_model = Model(
        inputs=base_model.input,
        outputs=GlobalAveragePooling2D()(base_model.output)
    )


    embeddings = []

    for images, _ in dataset:
        images = preprocess_input(images)
        emb = embedding_model.predict(images, verbose=0)
        embeddings.append(emb)

    embeddings = np.concatenate(embeddings, axis=0)

    path = "data/chroma"
    client = chromadb.PersistentClient(path=path)
    collection = client.get_or_create_collection(name="image_embeddings")

    collection.add(
        ids=[f"{i}_{img.split('/')[-1]}" for i, img in enumerate(dataset.file_paths)],
        embeddings=embeddings.tolist(),
        metadatas=[{"path": img} for img in dataset.file_paths]
    )
    print(f"Vector db saved on: {path}")


if __name__ == "__main__":
    main()



