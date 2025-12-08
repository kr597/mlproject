import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        try:
            # DEBUG: show incoming features
            print("DEBUG: incoming features:\n", features)

            # expected feature order (must match training)
            expected_cols = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
                "reading_score",
                "writing_score",
            ]

            # Ensure all expected columns exist (if missing add as None)
            for col in expected_cols:
                if col not in features.columns:
                    features[col] = None

            # Reorder to expected columns
            features = features[expected_cols]

            # Replace None and empty strings with a sentinel for categorical columns
            features = features.replace({None: "missing", "": "missing"})

            # Convert numeric fields to numeric type and fill NaNs (change fill strategy if you want)
            numeric_cols = ["reading_score", "writing_score"]
            for nc in numeric_cols:
                features[nc] = pd.to_numeric(features[nc], errors="coerce")
                features[nc] = features[nc].fillna(0)  # you may prefer median or mean

            print("DEBUG: cleaned features before transform:\n", features)

            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # transform and predict
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,                       # changed to str (form sends strings like "group A")
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
