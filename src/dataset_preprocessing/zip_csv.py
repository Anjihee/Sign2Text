import zipfile

with zipfile.ZipFile("merged_labeled_vectors.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
    zipf.write("merged_labeled_vectors.csv")
