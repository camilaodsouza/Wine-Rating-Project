import click
import pickle
import logging
import xgboost
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from fpdf import FPDF
#from reportlab.pdfgen import canvas
from sklearn.metrics import mean_squared_error

logging.basicConfig(level=logging.INFO)

@click.command()
@click.option('--in-model')
@click.option('--test-csv')
@click.option('--out-dir')
def evaluate_model(in_model, test_csv, out_dir):
	log = logging.getLogger('evaluate-model')
	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	# load model
	model = pickle.load(open(in_model, 'rb'))
	# load test data
	df = pd.read_csv(test_csv)
	# create X and y
	X_test = df.drop(columns=['Unnamed: 0', 'points'])
	y_test = df['points']
	# make predictions
	y_pred=model.predict(X_test)
	mse = mean_squared_error(y_test, y_pred)
	log.info("Mean squared error was {}".format(mse))

	# calculate feature importance
	xgboost.plot_importance(model)
	plt.title("xgboost.plot_importance(model)")
	#plt.title("xgboost.plot_importance(model)")
	#fi = pd.DataFrame({'feature': list(X_test.columns),
    #               'importance': model.feature_importances_}).\
    #                sort_values('importance', ascending = False)
	
	#importance_plot = sns.barplot(x='importance', y='feature', data=fi)
	#importance_path = (out_dir / 'importance.png/').resolve()
	plt.savefig("importance.png")

	#Code to generate report 
	report_path = out_dir / 'report.pdf/'

	WIDTH = 210
	HEIGHT = 297

	pdf = FPDF()
	pdf.add_page()

	pdf.set_font('Arial', 'B', 16)
	pdf.write(10, "Wine Rating Predictor Report\n")
	pdf.set_font('Arial', 'B', 8)
	pdf.write(10, "Original features used: Country, price, province, region_1, taster_name, variety\n")
	pdf.write(10, "Engineered features:\n")
	pdf.write(5, "    From title: year\n")
	pdf.write(5, "    From description: description length\n")
	pdf.write(5, "    From description: compound.\n")
	pdf.write(5, "Compound is a score from -1 (extremely negative) to +1 (extremely positive) given by sentiment analysis tool VADER.\n")
	pdf.write(5, "\n")
	pdf.write(5, "Baseline: Linear Regression\n")
	pdf.write(5, "Mean Squared Error: 5.00\n")
	pdf.write(5, "\n")
	pdf.write(5, "Baseline: Linear Regression\n")
	pdf.write(5, "Mean Squared Error: {:.2f}\n".format(mse))
	pdf.write(5, "\n")
	pdf.write(5, "Visualizing predictions:\n")
	pdf.write(5, "Actual rating of five first wines: {}\n".format(y_test.values[0:5]))
	pdf.write(5, "Predicted rating of five first wines: {}\n".format(y_pred[0:5]))
	pdf.write(5, "\n")
	pdf.write(5, "Actual x Predicted points distribution:\n")

	plt.style.use('fivethirtyeight')
	plt.rc("figure", figsize=(20,10))
	plt.hist(y_test.values, bins = 20, edgecolor = 'k', alpha = 0.6)
	plt.hist(y_pred, bins = 20, edgecolor = 'k', alpha = 0.5)
	plt.xlabel('Points'); plt.ylabel('Count'); 
	plt.title('Wine points distribution');
	plt.savefig("distribution.png")
	pdf.image("distribution.png", 10, 130, WIDTH*0.75, HEIGHT/3)

	pdf.add_page()
	pdf.write(10, "\n")
	pdf.write(5, "Most relevant features:\n")
	plt.rc("figure", figsize=(20,10))
	xgboost.plot_importance(model)
	plt.title("xgboost.plot_importance(model)")
	plt.savefig("importance.png")
	pdf.image("importance.png", 10, 30, WIDTH*0.75, HEIGHT/3)

	pdf.write(120, "\n")

	pdf.set_font('Arial', 'B', 10)
	pdf.write(5, "Conclusion:\n")
	pdf.write(10, "\n")
	pdf.set_font('Arial', 'B', 8)
	pdf.write(5, "Using a basic linear regression model as our baseline, we achieved a 5.0 MSE in our regression problem.\n")
	pdf.write(5, "We were able to improve this performance by using a XGBoost model with default hyperparameters, which achieved a 3.55 MSE\n")
	pdf.write(5, "We believe that this result can be further improved by tuning both the hyperparameter and the VADER sentiment analysis tool to our specific case. More experiments with different types of encoders can also help us achieve higher result.We recommend implementing a full production solution, not only for the promising predictor metrics, but also for its potential to provide important insight to our costumer about their products and comercial partners.\n")
	pdf.write(5, "More technical details and further graphical visualization can be found on the attached notebook 'WineRatingPredictor.ipynb'.\n")

	pdf.output(report_path, 'F')

if __name__ == '__main__':
	evaluate_model()
