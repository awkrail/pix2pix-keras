from net import create_fcn
from Img2ImgDataset import FlightTestDataset
from PIL import Image
import numpy as np

def main():
    target_size = (224, 224)
    model = create_fcn(target_size)
    model.load_weights("checkpoints/model_weights_20.h5")

    # test_dataset
    test_flight_dataset = FlightTestDataset("data/test/")
    
    for t_path, label in test_flight_dataset.test_dataset:
        predict_image = model.predict(label)
        predict_image = (predict_image + 1) * 128.0
        predict_image = predict_image.reshape(224, 224, 3).astype(np.uint8)
        output_image = Image.fromarray(predict_image)
        output_image.save("data/output/" + t_path)


if __name__ == "__main__":
    main()