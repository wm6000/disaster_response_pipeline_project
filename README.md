# Disaster Response Pipline

This project aims to take data from real world disaster data from tweets and direct messages and build a natural language processing tool that categorizes messages so the information gets to the right response organization or team.

## Table of Contents
- [Getting Started](#getting-started)
  - [Data](#data)
  - [Installing](#installing)
  - [Running the Scripts](#running-the-scripts)
  - [Running the Web App](#running-the-web-app)
- [Files in the Repository](#files-in-the-repository)
- [License](#license)


## Getting Started

### Data

This dataset contains nearly 30,000 messages drawn from events including an earthquake in Haiti in 2010, an earthquake in Chile in 2010, floods in Pakistan in 2010, super-storm Sandy in the U.S.A. in 2012, and news articles spanning a large number of years and 100s of different disasters.

The data has been encoded with 36 different categories related to disaster response and has been stripped of messages with sensitive information in their entirety.

Data includes 2 csv files:
- `disaster_messages.csv`: Messages data.
- `disaster_categories.csv`: Disaster categories of messages.


### Installing

Clone this GitHub repository

Python 3+

Flask==2.3.2

Jinja2==3.1.2

joblib==1.2.0

nltk==3.8.1

numpy==1.24.3

pandas==2.0.1

plotly==5.14.1

regex==2023.5.5

requests==2.30.0

scikit-learn==1.2.2

sklearn==0.0.post5

SQLAlchemy==2.0.13



### Running ETL Script
#### process_data.py | ETL work-flow:

- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

#### Run Command
`python data/process_data.py  data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

### Running NLP Script
#### train_classifier.py | NLP work-flow:

- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

#### Run Command
`python models/train_classifier.py data/DisasterResponse.db data/classifier.pkl`

### Running the Web App
#### Run Command
`python run.py`

Go to http://0.0.0.0:3001/

## Files in the Repository

- `script.py`: [Describe the purpose or functionality of this script.]
- `helper.py`: [Explain the role of this helper script or module.]
- `data/`: [Provide a brief overview of the contents of the 'data' directory.]
- `models/`: [Explain the purpose or types of models stored in the 'models' directory.]
- `templates/`: [Describe the templates used in the web app, if applicable.]
- `app.py`: [Explain the role of this file in the web app, if applicable.]
- `README.md`: [This current file, providing information and instructions.]

## License

[Specify the license under which the project is distributed.]









This project aims to visualize gray and blue whale sightings along the West Coast and how they intersect with global shipping density.

## File Descriptions:
There are three .ipynb files. They each generate a folium map that are displayed in my website blog: https://willswebsite.herokuapp.com/whales

## Installation:
Dependencies:
- [Python Version: 3.11.3]
- [branca==0.6.0]
- [folium==0.14.0]
- [geopandas==0.13.0]
- [numpy==1.24.3]
- [pandas==2.0.1]
- [rasterio==1.3.6]
- [shapely==2.0.1]




## Usage:
Run each .ipynb script to generate the map visualization. You may need to download the file to trust the notebook which will then show the map.

## Data:
The whale sighting data was obtained from the Ocean Biogeographic Information System (OBIS) and the Marine Geospatial Ecology Lab (MGEL) at Duke University: [OBIS](https://seamap.env.duke.edu/)

The Global Shipping Density was obtained from the World Bank: [Global Shipping Traffic Density](https://datacatalog.worldbank.org/search/dataset/0037580/Global-Shipping-Traffic-Density)


## License:
This project is licensed under the MIT License - see the LICENSE file for details.
