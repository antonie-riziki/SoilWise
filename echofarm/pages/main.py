import pandas as pd 
import plotly.express as px
import streamlit as st 
import matplotlib.pyplot as plt 
import seaborn as sb 
import numpy as np 
import csv
import os
import sys
import warnings
import google.generativeai as genai


from sklearn.impute import SimpleImputer

sys.path.insert(1, './pages')
print(sys.path.insert(1, '../pages/'))

# Get the absolute path of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the pages directory to sys.path
sys.path.insert(0, os.path.join(current_dir, '../pages'))

from ct_model import the_explainer

from dotenv import load_dotenv

load_dotenv()


header = st.container()
home_sidebar = st.container()

with header:
	st.title("Sowing Data, Reaping Insights: A Farmer's Tale")
	st.markdown('Your Digital Protector Against Crop Threats')


	st.write('''
		They say farmers know their soil like the back of their hand, but even the most seasoned farmer might miss a decimal point here and 
		there. That’s where we step in—with data, charts, and maybe a sprinkle of insights to make sense of it all. Let’s dig into this dataset 
		and harvest some valuable insights (no boots required). 🌱📊

		''')
	
	data = r'../src/pdc_data_zenodo.csv'

	current_dir = os.path.dirname(os.path.abspath(__file__))

	# Construct the absolute path to the CSV file
	csv_path = os.path.join(current_dir, '../src/pdc_data_zenodo.csv')

	df = pd.read_csv(csv_path, encoding='latin1')
	st.dataframe(df.head())

	col1, col2 = st.columns(2)

	with col1:
		st.table(df.describe())

		st.write('Missing value')
		st.table(df.isnull().sum()[:8])


	with col2:
		st.write(df.shape)

		st.write('Missing values')
		st.table(df.isnull().sum()[9:])


	with st.expander("See explanation"):
	    st.write('''
	        Our dataset has many missing values posing challenges for analysis and predictive modeling by reducing sample size, 
	        introducing bias, and distorting feature relationships. Missing data often results from collection gaps or user oversight, 
	        which can impact model performance and reliability. Strategies like imputation, feature engineering, and robust model selection 
	        can help mitigate these issues, while also offering opportunities to improve data collection processes and uncover hidden insights..

	        However, we will be implementing the Imputation method to deal with missing values for precision and accuracy
	    ''')

	     # current_dir = os.path.dirname(os.path.abspath(__file__))

	     # # Construct the absolute path to the CSV file
	     # heatmap_path = os.path.join(current_dir, "./assets/img/Heatmap Correlation.jpg")
		
	    st.image("../assets/img/Heatmap Correlation.jpg")

	simple_impute = SimpleImputer()

	def get_categorical_series(df):
	    categories = []
	    simple_impute = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
	    for i in df.select_dtypes(include=['object']):
	        categories.append(i)
	    df[categories] = simple_impute.fit_transform(df[categories])
	    return df.head()


	get_categorical_series(df)


	def get_quantitative_series(df):
	    numericals = []
	    simple_impute = SimpleImputer(missing_values=np.nan, strategy='mean')
	    for i in df.select_dtypes(include=['float64', 'int64']):
	        numericals.append(i)
	    df[numericals] = simple_impute.fit_transform(df[numericals])
	    return df.head()


	get_categorical_series(df)



	col3, col4 = st.columns(2, gap="small")
	
	with col3:

		st.write('For categorical analysis')

		category_choice = st.selectbox(label="select series", options=[i for i in df.select_dtypes(include="object")])

		st.bar_chart(df[category_choice].value_counts())

		plt.title(f"Categorical analysis for the {category_choice}")



	with col4:
		st.write('For Quantitative analysis')

		quantitative_choice = st.selectbox(label="select series", options=[i for i in df.select_dtypes(include=["float64", "int64"])])

		# df = px.data.tips()
		# fig = px.histogram(df[quantitative_choice])
		# fig.show()

		st.bar_chart(df[quantitative_choice])


	st.write('Grouped Results')


	col1, col2 = st.columns(2)

	group_series = ['country', 'region']

	with col1:
		group_column = st.selectbox(label="select series", options=[i for i in df[group_series]])

		country_grp = df.groupby(group_column)


	with col2:

		# total_observations = len(country_grp[group_column].unique())
		# all_observations = [i for i in country_grp[group_column].unique().iloc[0:total_observations]]

		all_observations = country_grp.groups.keys()

		
		if group_column:
			sub_groups = st.selectbox(label="select sub group", options=all_observations)

	extract_list = [i for i in sub_groups]

	get_group = country_grp.get_group(sub_groups)


	if sub_groups:

		grp_series = st.selectbox(label="", options=df.columns)

		if grp_series:
			bar_chart = st.bar_chart(get_group[grp_series].value_counts())

	with st.expander("", expanded=True):
		st.title('''The Explainer''')

		# st.write(f'Group: {group_column} \nSub Group:{sub_groups} \nSeries: {get_group[grp_series].value_counts()}')

		prompt = (
	        f"You have selected the country/region '{group_column}' and are analyzing the '{sub_groups}' series. "
	        f"This graph provides insights into how '{get_group[grp_series].value_counts()}' varies within '{group_column}'.\n\n"
	        f"Key Insights:\n"
	    )

		the_explainer(prompt)



	# st.multiselect()
