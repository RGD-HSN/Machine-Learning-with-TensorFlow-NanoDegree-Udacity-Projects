import argparse
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json


def process_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224)) 
    img_array = image.img_to_array(img) 
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0  
    return img_array


def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def predict(image_path, model, top_k=5):
    img_array = process_image(image_path)
    predictions = model.predict(img_array)
    top_k_probs, top_k_indices = tf.nn.top_k(predictions, k=top_k)
    return top_k_probs.numpy(), top_k_indices.numpy()


def load_category_names(category_names_path):
    with open(category_names_path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Classify images using a trained model")
    parser.add_argument('image_path', type=str, help="Path to the input image")
    parser.add_argument('model_path', type=str, help="Path to the saved model")
    parser.add_argument('--top_k', type=int, default=5, help="Return the top K most likely classes")
    parser.add_argument('--category_names', type=str, help="Path to a JSON file mapping labels to flower names")
    
    args = parser.parse_args()
    

    model = load_model(args.model_path)
    model = tf.keras.models.load_model('saved_model.keras')

    category_names = None
    if args.category_names:
        category_names = load_category_names(args.category_names)
    

    probs, indices = predict(args.image_path, model, top_k=args.top_k)
    

    print("Predictions:")
    for i in range(args.top_k):
        class_index = indices[0][i]
        prob = probs[0][i]
        flower_name = category_names.get(str(class_index), str(class_index)) if category_names else f"Class {class_index}"
        print(f"{flower_name}: {prob:.4f}")

if __name__ == "__main__":
    main()
