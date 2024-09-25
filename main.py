from fastapi import FastAPI, HTTPException
from pydantic import BaseModel , conint , Field
import pandas as pd
import joblib
import re
from sklearn.impute import SimpleImputer
from typing import List
import logging

logging.basicConfig(level=logging.DEBUG)
best_dt_model = joblib.load('best_decision_tree_model.pkl')
model_columns = joblib.load('model_columns.pkl')

def fdi_dental_numbering_validator(cls, value: str):
    try:
        numbers=[int(v) for v in value.split()]
    except ValueError:
        raise ValueError("Tous les éléments doivent être des entiers valides.")
    for number in numbers:
        if not ((11<=number<=18)or(21<=number<=28)or(31<=number<=38)or(41<=number<=48)):
            raise ValueError(f"Numéro dentaire invalide '{number}'. Doit être entre 11-18, 21-28, 31-38, ou 41-48.")
    return value

def choice_field_validator(choices: List[str]):
    @classmethod
    def validate(cls, value):
        if value not in choices:
            raise ValueError(f"Invalid choice. Must be one of {choices}")
        return value
    return validate
def range_field_list_validator(min_value: int, max_value: int):
    def validate(cls, value: str):
        
        try:
            numbers = [int(v) for v in value.split()]
        except ValueError:
            raise ValueError("All elements must be valid integers.")

    
        if not all(min_value <= num <= max_value for num in numbers):
            raise ValueError(f"Each value must be between {min_value} and {max_value}.")
        
        return value

    return classmethod(validate)
def choice_field_validator_feature15(choices):
    @classmethod
    def validate(cls, value):
        for item in value:
            if not re.match(r'\d+[<>]?=1', item):
                raise ValueError(f"Invalid format: '{item}'. Expected format is a number followed by <1, =1, or >1.")
        return value
    return validate
def choice_field_validator_feature16(choices):
    @classmethod
    def validate(cls, value):
        pattern = re.compile(rf"^({'|'.join(choices)}) \d+ elements$", re.IGNORECASE)
        if not pattern.match(value):
            raise ValueError(f"Invalid format: '{value}'. Expected format is one of {choices} followed by a number and 'elements'.")
        return value
    return validate
class Entry(BaseModel):
    feature2: str
    feature4: str = Field(description="Feature 4", validator=choice_field_validator(['BEG','HTA','RAA']))
    feature5: str = Field(description="Feature 5", validator=choice_field_validator(['cuspidés','antérieur']))
    feature6: str = Field(description="Feature 6", validator=fdi_dental_numbering_validator)
    feature7: str = Field(description="Feature 7", validator=choice_field_validator(['mobilité','non mobile']))
    feature8: str = Field(description="Feature 8", validator=choice_field_validator(['oui','non']))
    feature9: str = Field(description="Feature 9", validator=choice_field_validator(['oui','non']))
    feature10: str = Field(description="Feature 10", validator=choice_field_validator(['non','oui']))
    feature11: str = Field(description="Feature 11", validator=choice_field_validator(['non','oui']))
    feature12: str = Field(description="Feature 12", validator=choice_field_validator(['conservée','non conservée']))
    feature13: str = Field(description="Feature 13", validator=choice_field_validator(['non traitée','suffisant']))
    feature14: str = Field(description="Feature 14", validator=choice_field_validator(['oui','non']))
    feature15: List[str] = Field(description="Feature 15", validator=choice_field_validator_feature15(["<1", "=1", ">1"]))
    feature16: str = Field(description="Feature 16", validator=choice_field_validator_feature16(["bridge", "couronne", "facette"]))
    feature17: str = Field(description="Feature 17", validator=choice_field_validator(['dsr','non']))
    feature18: str = Field(description="Feature 18", validator=choice_field_validator(['oui','non']))
    feature19: str = Field(description="Feature 19", validator=choice_field_validator(['oui','non']))
    feature20: str = Field(description="Feature 20", validator=choice_field_validator(['non','coulée','foulée']))

app = FastAPI()

@app.post("/predict/")
def predict(entry: Entry):
    
    new_entry_df = pd.DataFrame([entry.dict()])
    new_entry_df["feature2"] = [re.sub(r'\s*ans\s*', '', age) for age in new_entry_df["feature2"]]
    new_entry_df['feature4'] = new_entry_df['feature4'].apply(lambda x: 1 if 'BEG' in x.strip() else 0)
    new_entry_df['feature5'] = new_entry_df['feature5'].apply(lambda x: 1 if x == 'antérieur' else 0)
    def extract_integers(text):
        if pd.isna(text):
           return []
        return [int(num) for num in re.findall(r'\b\d+\b', text)]
    
    extracted_numbers = new_entry_df['feature6'].apply(extract_integers)
    flat_list = [num for sublist in extracted_numbers for num in sublist]
    unique_numbers = list(set(flat_list)) if flat_list else []


    for number in unique_numbers:
        column_name = f'feature6_{number}'
        new_entry_df[column_name] = new_entry_df['feature6'].apply(lambda x: 1 if number in extract_integers(x) else 0)

    bool_columns = ['feature7', 'feature8', 'feature9', 'feature10', 'feature11', 'feature12', 'feature13', 'feature14', 'feature17', 'feature18', 'feature19', 'feature20']
    for col in bool_columns:
        new_entry_df[col] = new_entry_df[col].apply(lambda x: 0 if 'non' in x.strip() else 1)

    def detect_equal_or_implicit_one(x):
        return 1 if "=1" in str(x) or re.search(r'(?<![<>])\b1\b(?![<>])', str(x)) else 0
    new_entry_df['feature15<1'] = new_entry_df['feature15'].apply(lambda x: 1 if '<1' in str(x) else 0)
    new_entry_df['feature15>1'] = new_entry_df['feature15'].apply(lambda x: 1 if '>1' in str(x) else 0)
    new_entry_df['feature15=1'] = new_entry_df['feature15'].apply(detect_equal_or_implicit_one)
    def split_feature16(x):
        match = re.search(r"(\d+)\s*[ééléments|élément|élémrnts|éléments]", x)
        if match:
           index = match.start()
           nombre_element = match.group(1)
           nom = x[:index].strip()
           return pd.Series([nom, nombre_element])
        else:
           return pd.Series([x.strip(), None])
    def split_feature16_1(x):
        match = re.search(r"\b(bridge|couronne|facette|facettes)\b", x, re.IGNORECASE)
        if match:
            terme = match.group(1).capitalize()
            terme = 'Facette' if terme.lower() in ['facette', 'facettes'] else terme
            reste = x.replace(match.group(0), "").strip()
            return pd.Series([terme, reste])
        else:
            return pd.Series(["Couronne", x.strip()])
    new_entry_df[['feature16_1', 'feature16_2']] = new_entry_df['feature16'].apply(split_feature16)
    new_entry_df[['feature16_1_terme', 'feature16_1_reste']] = new_entry_df['feature16_1'].apply(split_feature16_1)

    one_hot_encoded_terme = pd.get_dummies(new_entry_df['feature16_1_terme'], prefix='terme')
    one_hot_encoded_reste = pd.get_dummies(new_entry_df['feature16_1_reste'], prefix='reste')

    new_entry_df = pd.concat([new_entry_df, one_hot_encoded_terme, one_hot_encoded_reste], axis=1)
    imputer_mode = SimpleImputer(strategy='median')
    new_entry_df['feature16_2'] = imputer_mode.fit_transform(new_entry_df[['feature16_2']])
    for col in model_columns:
        if col not in new_entry_df.columns:
            new_entry_df[col] = 0
    logging.debug(new_entry_df.head())
    new_entry_df = new_entry_df[model_columns]
    
    prediction = best_dt_model.predict(new_entry_df)

    return {"prediction": int(prediction[0])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
