from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend.parser import convert_to_json  # your existing function

app = FastAPI(title="Au-NP Protocol → JSON")

class ParseRequest(BaseModel):
    text: str

@app.post("/parse")
def parse(req: ParseRequest):
    try:
        return convert_to_json(req.text)   # returns a dict → FastAPI auto-serialises to JSON
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
