import os
import pandas as pd


def divide_files(directory_in: str, directory_out: str, file_name: str, file_ext: str = 'dat',
                 line_length: int = 2 ** 16, set_num: bool = False, sep: str = '\t') -> None:
    """
    Load and save files from a directory to another directory with chosen length.

    Args:
        directory_in (str): The directory where the input dat files are located.
        directory_out (str): The directory where the output files will be saved.
        file_name (str): The base name for the output files.
        file_ext (str, optional): The file extension for the input and output files. Defaults to 'dat'.
        line_length (int, optional): The number of lines to include in each output file. Defaults to 2^16.
        set_num(bool, optional): Whether to add the line number to the output file name. Defaults to False.
        sep(str, optional): The separator for the input and output files. Defaults to '\t'.

    Raises:
        ValueError: If no files are found in the input directory.

    Returns:
        None

    This function loads dat files from the input directory, splits them into chunks of specified line length, and saves each chunk as a separate file in the output directory. The output files are named using the base file name and the range of lines included in each chunk.

    Example:
        load_and_save_dat_files('/path/to/input', '/path/to/output', 'output_file', 'dat', 1000)
    """
    files = sorted([os.path.join(directory_in, filename) for filename in os.listdir(directory_in)
                    if filename.endswith(f".{file_ext}")])
    if not files:
        raise ValueError("No files found in the directory")

    print("Founded files:")
    print(files)

    count = 0

    default_file = os.path.join(directory_in, f"chunks.{file_ext}")
    files.append(default_file)

    for i, file in enumerate(files):
        try:
            data = pd.read_csv(file, delimiter=sep, header=None)
            num_rows = len(data)
            for j in range(0, num_rows, line_length):
                chunk = data.iloc[j:j + line_length]
                if len(chunk) < line_length and file != default_file:
                    if os.path.exists(default_file):
                        chunk.to_csv(default_file, mode='a', sep=sep, index=False, header=False)
                    else:
                        chunk.to_csv(default_file, sep=sep, index=False, header=False)
                else:
                    if set_num:
                        output_file = os.path.join(directory_out,
                                                   f"{file_name}_{count}.{file_ext}")

                    else:
                        output_file = os.path.join(directory_out,
                                                   f"{file_name}_{count * line_length}_{count * line_length + len(chunk) - 1}.{file_ext}")
                    chunk.to_csv(output_file, sep=sep, index=False, header=False)
                    print(f"File saved: {output_file}")
                    count += 1
                del chunk
            del data
            print(f"File written {i + 1}/{len(files)} ({file})")
        except Exception as e:
            print(f"Error processing file {file}: {e}")

    if os.path.exists(default_file):
        os.remove(default_file)


if __name__ == "__main__":
    divide_files("../dataset/data_sim/decor/raw/answer", "../dataset/data_sim/decor/divided_raw/answer", "OTDCR",
                 "txt", 1000, True)

    # divide_files("../dataset/data_exp/sct", "../dataset/data_exp/sct/divided_raw", "exp_data",
    #              "dat", 2 ** 16, False)
