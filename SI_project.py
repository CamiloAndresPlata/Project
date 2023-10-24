# %% Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tensorflow import keras
from keras.layers import Masking, LSTM, Dense

# %% loading dataset

# Converts a string into a bool for the missing values 

def strtobool(v):
  return v.lower() in ("yes", "true", "t", "1")

# Converts the contents in a .tsf file into a dataframe and returns it along 
# with other meta-data of the dataset: frequency, horizon, whether the dataset 
# contains missing values and whether the series have equal lengths

def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with=np.nan,
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )

loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe("data/nov_data.tsf")
print(loaded_data)
print('frecuency :',frequency)

# %% ploting dataset

plt.figure(figsize=(40,20))
plt.ylim(top=600)

raw_data = np.zeros((46848,6))

for i in range(6):
    t = [loaded_data['start_timestamp'][i] + timedelta(minutes=j*15) for j in range(len(loaded_data['series_value'][i]))]
    initial_date = t.index(datetime(2019, 8, 1, 0, 0))
    x = t[initial_date:] 
    raw_data[:,i] = loaded_data['series_value'][i][initial_date:]
    plt.plot(x, raw_data[:,i], label=loaded_data['series_name'][i])

plt.xlabel('Date')
plt.ylabel('Energy consumption')
plt.legend()


# %% Num of train, validate and test samples

num_train_samples = int(0.75 * len(raw_data))
initial_test = t.index(datetime(2020, 11, 1, 0, 0))
num_test_samples = len(raw_data[initial_test:])
num_val_samples = len(raw_data) - num_train_samples - num_test_samples

print("num_train_samples:", num_train_samples)
print("num_val_samples:", num_val_samples)
print("num_test_samples:", num_test_samples)

# %% Preproccesing data

raw_data = np.ma.masked_invalid(raw_data)

target_data = raw_data.sum(axis=1)

mean = raw_data[:num_train_samples].mean(axis=0)
raw_data -= mean
std = raw_data[:num_train_samples].std(axis=0)
raw_data /= std

raw_data.mask = np.ma.nomask
target_data.mask = np.ma.nomask

raw_data = np.nan_to_num(raw_data)
target_data = np.nan_to_num(target_data)

# %% Instantiating training, validation and testing datasets

sampling_rate = 4
sequence_length = 48
delay = sequence_length
batch_size = 24

train_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=target_data[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=0,
    end_index=num_train_samples)

val_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=target_data[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples,
    end_index=num_train_samples + num_val_samples)

test_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=target_data[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples + num_val_samples)

# %% LSTM model

inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
z = Masking(mask_value=0)(inputs)
z = LSTM(96, recurrent_dropout=0.25)(z)
outputs = Dense(1)(z)

model = keras.Model(inputs, outputs)
callbacks = [keras.callbacks.ModelCheckpoint("lstm_dropout.keras", save_best_only=True)]

model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
history = model.fit(train_dataset, epochs=20, validation_data=val_dataset, callbacks=callbacks)

# %% Training and validation MAE 

loss = history.history["mae"]
val_loss = history.history["val_mae"]
epochs = range(1, len(loss) + 1)

plt.figure(figsize=(8,6))
plt.plot(epochs, loss, "o--k", label="Training MAE")
plt.plot(epochs, val_loss, "o--b", label="Validation MAE")
plt.title("Training and validation MAE")
plt.legend()
plt.show()

# %% Predicted output 

y_pred = model.predict(test_dataset)
test_index = num_train_samples + num_val_samples

plt.figure(figsize=(50,10))
plt.ylim((400,1100))
plt.xlim((-30,2000))
plt.plot(target_data[test_index+2*delay:-3*delay], label='True', marker='o', linestyle='-')
plt.plot(y_pred, label='Predicted', marker='x', linestyle='--')
plt.title('Validation Outputs')
plt.xlabel('Samples')
plt.ylabel('Power (KW)')
plt.legend()
plt.show()
