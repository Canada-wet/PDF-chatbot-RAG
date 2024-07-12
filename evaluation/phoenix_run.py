import phoenix as px
from fastapi import FastAPI
import uvicorn
session = px.launch_app(host="0.0.0.0", port=6006)

# placeholder for api settings
app = FastAPI()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)