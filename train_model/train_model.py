import click
import pickle
import pandas as pd
from xgboost import XGBRegressor
from pathlib import Path

def _save_trained_model(model, outdir: Path):
	"""Save transformed data set into nice directory structure and write SUCCESS flag."""
	trained_model = outdir / 'trained_model.sav/'
	flag = outdir / '.SUCCESS'

	pickle.dump(model, open(trained_model, 'wb'))

	flag.touch()

@click.command()
@click.option('--train-csv')
@click.option('--out-dir')
def train_model(train_csv, out_dir):
	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	# load dataset
	df = pd.read_csv(train_csv)
	# create X and y
	X_train = df.drop(columns=['Unnamed: 0', 'points'])
	y_train = df['points']
	# train model
	model = XGBRegressor()
	model.fit(X_train, y_train, verbose=False)

	_save_trained_model(model, out_dir)

if __name__ == '__main__':
	train_model()
