import glob
from mh.spectra import model_from_txt

DIRS = [".","JPLUS"]

for DIR in DIRS:
    fnames = glob.iglob(DIR+"/*dat")
    for fname in fnames:
        S = model_from_txt(fname, y_unit="")
        fout = fname[:-4]+".npy"
        S.write(fout, errors=False)
