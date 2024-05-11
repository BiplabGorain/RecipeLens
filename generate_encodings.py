# This script generates encodings for all the images in the dataset and
# saves it in a txt file and the names of the recipes in another txt file.
# Make sure to change the file paths before running the script.
import os
import numpy as np
import tensorflow as tf
# from tensorflow.keras.applications import DenseNet201
# from tensorflow.keras.preprocessing import image
# from keras import models  # Import Keras for model loading (if needed)

def get_encodings(img):
  """Preprocesses and encodes an image using DenseNet201."""
  img_array = tf.keras.preprocessing.image.img_to_array(
      img)  # Convert to NumPy array
  img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
  processed_img = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False)(
      img_array)  # Preprocess and extract features
  return processed_img


if __name__ == '__main__':
  image_dir = "Dataset/images"
  encodings_list = []

  for filename in os.listdir(image_dir):
    image_path = os.path.join(image_dir, filename)
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(256, 256))
    encoding = get_encodings(img)
    encodings_list.append(encoding)

  print(f"Number of images: {len(encodings_list)}")

  with open('encodings.pkl', 'wb') as file:
    import pickle  # Import pickle for clarity
    pickle.dump(encodings_list, file)  # Use pickle.dump for binary data

  with open('enc_names.txt', 'w') as file:  # Use 'w' for text data
    file.writelines(os.listdir(image_dir))
