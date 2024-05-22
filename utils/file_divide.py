import os
import pandas as pd


def load_and_save_dat_files(directory_in, directory_out, line_length):
    files = os.listdir(directory_in)
    files = [os.path.join(directory_in, filename) for filename in files]
    files = [x for x in files if x.endswith(".dat")]
    current_file = 0
    i = 0

    print(files)
    all_data = pd.read_csv(files[current_file], delimiter='\t', header=None)
    current_file += 1

    print("Start here")

    while True:
        if len(all_data) > line_length:
            all_data.loc[0: line_length].to_csv(
                os.path.join(directory_out, f"exp_data_{i * line_length}_{(i + 1) * line_length - 1}.dat"), sep='\t',
                index=False, header=False)
            print("File written", i, len(all_data))
            all_data = all_data.drop(range(line_length))
            all_data = all_data.reset_index(drop=True)
            i += 1

        if len(all_data) <= line_length:
            if current_file == len(files):
                all_data.to_csv(
                    os.path.join(directory_out,
                                 f"exp_data_{i * line_length}_{i * line_length + len(all_data) - 1}.dat"),
                    sep='\t', index=False, header=False)
                print("File written ended", i, len(all_data))
                break
            else:
                data = pd.read_csv(files[current_file], delimiter='\t', header=None)
                current_file += 1
                all_data.append(data, ignore_index=True)

        if len(all_data) == 0:
            break


if __name__ == "__main__":
    load_and_save_dat_files("../dataset/exp", "../dataset/exp/", 2 ** 16)
