from Img2ImgDataset import FlightDataset

def main():
    train_flight_dataset = FlightDataset(input_path="data/train_mask/", 
                                         output_path="data/train/", 
                                         data_range=(1, 507))
    valid_flight_dataset = FlightDataset(input_path="data/valid_mask/",
                                         output_path="data/valid/",
                                         data_range=(507, 537))
    X_train, y_train = train_flight_dataset.in_array, train_flight_dataset.out_array
    X_valid, y_valid = valid_flight_dataset.in_array, valid_flight_dataset.out_array
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    main()