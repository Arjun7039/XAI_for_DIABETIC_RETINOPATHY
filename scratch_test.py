import tensorflow as tf
import numpy as np

def test_model():
    model_path = "backend/weights/efficientnet_b4_best.keras"
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path, compile=False)
    
    print(f"Model input shape: {model.input_shape}")
    
    # Generate 3 dummy images
    # 1. Zeroes (black image)
    # 2. Random [0, 1]
    # 3. Random [0, 255]
    img_zeros = np.zeros((1, 224, 224, 3), dtype=np.float32)
    img_0_1 = np.random.rand(1, 224, 224, 3).astype(np.float32)
    img_0_255 = (np.random.rand(1, 224, 224, 3) * 255).astype(np.float32)
    
    print("\nTesting Predictions:")
    pred_zeros = model.predict(img_zeros, verbose=0)
    print(f"Zeros input: {pred_zeros} -> class {np.argmax(pred_zeros)}")
    
    pred_0_1 = model.predict(img_0_1, verbose=0)
    print(f"Random [0, 1] input: {pred_0_1} -> class {np.argmax(pred_0_1)}")
    
    pred_0_255 = model.predict(img_0_255, verbose=0)
    print(f"Random [0, 255] input: {pred_0_255} -> class {np.argmax(pred_0_255)}")
    
    print("\nTesting Keras Applications Preprocess Input:")
    from tensorflow.keras.applications.efficientnet import preprocess_input
    
    img_pre_0_1 = preprocess_input(img_0_1.copy())
    pred_pre_0_1 = model.predict(img_pre_0_1, verbose=0)
    print(f"Preprocessed [0, 1]: {pred_pre_0_1} -> class {np.argmax(pred_pre_0_1)}")
    
    img_pre_0_255 = preprocess_input(img_0_255.copy())
    pred_pre_0_255 = model.predict(img_pre_0_255, verbose=0)
    print(f"Preprocessed [0, 255]: {pred_pre_0_255} -> class {np.argmax(pred_pre_0_255)}")

    print("\nModel Layers:")
    for i, layer in enumerate(model.layers[:10]):
        print(f"Layer {i}: {layer.name} ({layer.__class__.__name__})")
        if isinstance(layer, tf.keras.Model):
            print("  Sub-model layers:")
            for j, sub_layer in enumerate(layer.layers[:5]):
                print(f"    Sub-layer {j}: {sub_layer.name} ({sub_layer.__class__.__name__})")
    
if __name__ == "__main__":
    test_model()
