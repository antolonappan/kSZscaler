import numpy as np
import wget
import shutil
import os
import gzip
import pickle as pl
import pandas as pd


class Magneticum:
    """
    class to download and read Magneticum data
    """

    def __init__(
        self, snap: str, box: str = "", body: str = "cluster", mcut_gal: float = 0
    ):
        self.snap = snap
        self.box = box
        self.body = body
        assert self.body in [
            "cluster",
            "galaxies",
        ], "body should be either 'cluster' or 'galaxies"
        self.fname = self.__datafile__(f"{body}{self.snap}{self.box}.txt")
        self.link = f"http://wwwmpa.mpa-garching.mpg.de/HydroSims/Magneticum/Downloads/Magneticum/Box2{box}_hr/snap_{snap}/{body}.txt.gz"
        self.z = self.redshift_snapshot(snap, box)
        self.mcut_gal = mcut_gal

    def __datafile__(self, fname: str) -> str:
        dir_base = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(dir_base, "Data", fname)

    @staticmethod
    def available_snapshots(box: str = "") -> list:
        assert box in ["", "b"]
        dir_base = os.path.dirname(os.path.realpath(__file__))
        snap = pl.load(open(os.path.join(dir_base, "Data", f"box2{box}.pkl"), "rb"))
        return list(snap.keys())

    @staticmethod
    def redshift_snapshot(snap: str = None, box: str = "") -> float:
        assert box in ["", "b"]
        dir_base = os.path.dirname(os.path.realpath(__file__))
        snapdic = pl.load(open(os.path.join(dir_base, "Data", f"box2{box}.pkl"), "rb"))
        if snap is None:
            return snapdic
        else:
            return snapdic[snap]

    def download_data(self) -> None:
        print(f"Downloading {self.link}")
        wget.download(self.link, out=self.fname + ".gz")

    def unzip_data(self) -> None:
        print(f"Unzipping {self.fname}.gz")
        with gzip.open(self.fname + ".gz", "rb") as s_file, open(
            self.fname, "wb"
        ) as d_file:
            shutil.copyfileobj(s_file, d_file, 65536)

    def get_data(self) -> None:
        self.download_data()
        self.unzip_data()
        os.remove(self.fname + ".gz")
        print(f"Data saved to {self.fname}")

    @property
    def dataframe(self) -> pd.DataFrame:
        if not os.path.isfile(self.fname):
            self.get_data()
        df = pd.read_csv(self.fname, delim_whitespace=True)
        cols = list(df.columns)
        df.columns = cols[1:] + cols[:1]
        df.drop(columns=["#"], inplace=True)
        lenbefore = len(df)
        if (self.mcut_gal > 0) and (self.body == "galaxies"):
            df = df[df["m[Msol/h]"] > self.mcut_gal]
            print(
                f"Galaxies mass cut applied: total-{lenbefore}, current-{len(df)}, removed-{lenbefore-len(df)}"
            )
        return df
