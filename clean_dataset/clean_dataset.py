import re
import click
import pandas as pd
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def _save_clean_dataset(clean_dataset, outdir: Path):
    """Save transformed data set into nice directory structure and write SUCCESS flag."""
    out_clean_dataset = outdir / 'clean.csv/'
    flag = outdir / '.SUCCESS'

    clean_dataset.to_csv(str(out_clean_dataset), index=False)

    flag.touch()


@click.command()
@click.option('--in-csv')
@click.option('--out-dir')
def clean_dataset(in_csv, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv('wine_dataset.csv')
    
    # drop irrelevant columns 
    df = df.drop(columns = ['Unnamed: 0', 'designation', 'region_2', 'taster_twitter_handle', 'winery'])
    # drop duplicates
    df = df.drop_duplicates(ignore_index=True)
    
    # fill null values
    df['price'].fillna(df['price'].median(), inplace=True)
    df['region_1'].fillna(str(df['region_1'].mode()), inplace=True)
    df['taster_name'].fillna(0, inplace=True)
    
    # extract new features
    analyzerObj = SentimentIntensityAnalyzer()
    df['descLen'] = df['description'].map(lambda description: len(description))
    df['compound'] = df['description'].map(lambda description: analyzerObj.polarity_scores(description)['compound'])
    df['year'] = df['title'].map(lambda title: re.search(r"(\d{4})", title).group(1) if re.search(r"(\d{4})", title) else '0')
    df['year'] = pd.to_numeric(df['year'])
    median = df.loc[df['year']>0].median()[4]
    df['year'].replace(0, median, inplace = True)
    
    # encoding categorical features
    X = df.drop(columns=['points', 'description', 'title'])
    y = df['points']
    ce_ord = ce.OrdinalEncoder(cols = ['country', 'province', 'region_1', 'taster_name', 'variety'])
    X_labeled = ce_ord.fit_transform(X, y)
    
    # save transformed dataset to out_dir
    _save_clean_dataset(X_labeled, out_dir)


if __name__ == '__main__':
    clean_dataset()
