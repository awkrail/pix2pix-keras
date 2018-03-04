from Img2ImgDataset import FlightDataset
from net import create_fcn

from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

def main():
    target_size = (224, 224)

    train_flight_dataset = FlightDataset(input_path="data/train_mask/", 
                                         output_path="data/train/", 
                                         data_range=(1, 507))
    valid_flight_dataset = FlightDataset(input_path="data/valid_mask/",
                                         output_path="data/valid/",
                                         data_range=(507, 537))
    X_train, y_train = train_flight_dataset.in_array, train_flight_dataset.out_array
    X_valid, y_valid = valid_flight_dataset.in_array, valid_flight_dataset.out_array

    print("creating model...")
    model = create_fcn(target_size)
    model.summary()

    # set loss function and optimizer
    adam = Adam(lr=1e-5)
    model.compile(optimizer=adam, loss="mean_squared_error")

    # checkpoint
    checkpoint = ModelCheckpoint(filepath="checkpoints/model_weights_{epoch:02d}.h5"+ , save_best_only=False)

    # start training
    model.fit(X_train, y_train, batch_size=5, epochs=20, verbose=1,
              shuffle=True, validation_data=(X_valid, y_valid),
              callbacks=[checkpoint])


if __name__ == "__main__":
    main()