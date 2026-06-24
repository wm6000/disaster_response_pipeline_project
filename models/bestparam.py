import joblib
from train_classifier import tokenize, StartingVerbExtractor
# Load the saved GridSearchCV object from the .pkl file
pipeline = joblib.load('models/classifier.pkl')

# Access the best parameters
# Access the best parameters from the underlying GridSearchCV object

step_names = [name for name, _ in pipeline.steps]
print(step_names)
best_params = pipeline.named_steps['clf'].get_params
print(best_params)
