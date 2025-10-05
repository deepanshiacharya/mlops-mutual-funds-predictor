from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict

app = FastAPI(title="Mutual Funds Return Predictor", version="1.0.0")

class PredictionInput(BaseModel):
    AMC: str
    Morning_star_rating: int
    Value_Research_rating: int
    month_return: float
    NAV: float
    Year_return: float
    Minimum_investment: float
    AUM: float
    Category: str
    Risk: str

@app.get("/")
def root():
    return {"message": "Mutual Funds 3rd-Year Return Predictor API"}

@app.post("/predict")
def make_prediction(input_data: PredictionInput):
    data_dict = input_data.dict()
    data_dict['Morning star rating'] = data_dict.pop('Morning_star_rating')
    data_dict['Value Research rating'] = data_dict.pop('Value_Research_rating')
    data_dict['1 month return'] = data_dict.pop('month_return')
    data_dict['1 Year return'] = data_dict.pop('Year_return')
    data_dict['Minimum investment'] = data_dict.pop('Minimum_investment')
    
    try:
        result = predict(data_dict)
        return {"predicted_3_year_return": result}
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)