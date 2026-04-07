import exiftool

exif_path = r"C:\Users\sushu\Downloads\exiftool-13.53_64\exiftool.exe"

with exiftool.ExifTool(executable=exif_path) as et:
    metadata = et.execute_json(
        "-j",
        r"E:\Video_annotator _ v2\factory001_worker002_part09\factory001_worker002_00009.mp4"
    )

print(metadata)