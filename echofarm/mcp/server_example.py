# echofarm/mcp/server_example.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn

app = FastAPI(title="SoilWise - MCP Server")

# A simple register of tools. In production your MCP lib would expose discovery, schema, etc.
TOOLS = {
    "soil_recommendation": {
        "description": "Given soil metrics, return recommended crop + plan",
        "inputs": {
            "ph": "float",
            "moisture": "float",
            "nitrogen": "float",
            "phosphorus": "float",
            "potassium": "float",
            "temperature": "float"
        }
    },
    "weather": {
        "description": "Get weather forecast for a given location",
        "inputs": {
            "latitude": "float",
            "longitude": "float"
        }
    }
}

@app.get("/mcp/tools")
def list_tools():
    return TOOLS

class InvokeRequest(BaseModel):
    tool: str
    input: Dict[str, Any]

@app.post("/mcp/invoke")
def invoke(req: InvokeRequest):
    tool = req.tool
    data = req.input
    if tool not in TOOLS:
        raise HTTPException(404, detail="tool not found")
    # Minimal example logic — replace with your trained model call
    ph = float(data.get("ph", 7.0))
    nitrogen = float(data.get("nitrogen", 0))
    if ph < 5.5:
        rec = {
            "crop": "Legumes with liming",
            "quality": "Acidic",
            "advice": "Apply agricultural lime; add organic matter. Consider planting groundnuts after liming."
        }
    elif nitrogen < 50:
        rec = {
            "crop": "Maize (with compost)",
            "quality": "Moderate",
            "advice": "Add nitrogen fertilizers or compost. Follow NPK schedule: ... (example)"
        }
    else:
        rec = {
            "crop": "Beans, Kale",
            "quality": "Good",
            "advice": "Normal sowing; follow irrigation schedule for 25°C"
        }
    # return a structured response that your client + UI can render
    return {"tool": tool, "result": rec}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
