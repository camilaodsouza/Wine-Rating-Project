import click
import pandas as pd
#import numpy as np
#import dask.dataframe as dd
#from distributed import Client
from pathlib import Path
from sklearn.model_selection import train_test_split


def _save_datasets(train, test, outdir: Path):
    """Save data sets into nice directory structure and write SUCCESS flag."""
    out_train = outdir / 'train.csv/'
    out_test = outdir / 'test.csv/'
    flag = outdir / '.SUCCESS'

    train.to_csv(str(out_train))
    test.to_csv(str(out_test))

    flag.touch()


@click.command()
@click.option('--in-csv')
@click.option('--out-dir')
def make_datasets(in_csv, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
 
    df = pd.read_csv(in_csv)

    train, test = train_test_split(df, test_size=0.2, random_state=0)

    # trigger computation
    #n_samples = len(df)

    # TODO: implement proper dataset creation here
    # http://docs.dask.org/en/latest/dataframe-api.html

    # split dataset into train test feel free to adjust test percentage
    #idx = np.arange(n_samples)
    #test_idx = idx[:n_samples // 10]
    #test = ddf.loc[test_idx]

    #train_idx = idx[n_samples // 10:]
    #train = ddf.loc[train_idx]

    #df = pd.read_csv(in_csv)
    
    _save_datasets(train, test, out_dir)


if __name__ == '__main__':
    make_datasets()
