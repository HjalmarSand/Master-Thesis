# Take two image datasets as input and merge them into one

import csv
import shutil
from itertools import chain
from pathlib import Path


def merge_ordered(dataset1: Path, dataset2: Path):
    # Create a new dataset folder with the names of the two datasets with _merged added to the end of them. Expect to be run from root for path
    new_dir = Path(f"images/original_datasets/{dataset1.name}_{dataset2.name}_merged")
    new_dir.mkdir(parents=True, exist_ok=True)

    # make the image dir
    (new_dir / "images").mkdir(exist_ok=True)

    image_extension = ".jpg"
    files1 = sorted(
        (dataset1 / "images").glob(f"*{image_extension}"), key=lambda x: int(x.stem)
    )
    files2 = sorted(
        (dataset2 / "images").glob(f"*{image_extension}"), key=lambda x: int(x.stem)
    )

    file_index = 0
    for file in files1 + files2:
        new_name = f"{file_index:04d}{image_extension}"
        shutil.copy2(file, new_dir / "images" / new_name)
        file_index += 1

    # Open the csv for dataset 1 and dataset 2, keep the heading from the first one, throw for the second, and then refactor the rows equally to how the images were renamed
    csv1 = dataset1 / "gps_interpolated.csv"
    csv2 = dataset2 / "gps_interpolated.csv"
    merged_csv = Path(new_dir / "gps_interpolated.csv")

    # new index
    file_index = 0
    with merged_csv.open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)

        with csv1.open("r", newline="", encoding="utf-8") as in1:
            reader = csv.reader(in1)
            header = next(reader, None)
            writer.writerow(header)
            for row in reader:
                # here we can split row by "," and modify the row to instead use the correct count and then name + ".jpg" for second column
                new_row = [
                    str(file_index),
                    f"{file_index:04d}{image_extension}",
                    row[2],
                    row[3],
                ]
                writer.writerow(new_row)
                file_index += 1

        with csv2.open("r", newline="", encoding="utf-8") as in2:
            reader = csv.reader(in2)
            next(reader, None)  # skip heading
            for row in reader:
                new_row = [
                    str(file_index),
                    f"{file_index:04d}{image_extension}",
                    row[2],
                    row[3],
                ]
                writer.writerow(new_row)
                file_index += 1


if __name__ == "__main__":
    d1 = Path("./images/original_datasets/dataset6")
    d2 = Path("./images/original_datasets/dataset7")

    merge_ordered(d1, d2)
