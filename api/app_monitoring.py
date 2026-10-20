from fastapi import FastAPI
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()


@app.get("/")
def home():

    logging.info("Home endpoint accessed")

    return {
        "message": "Monitoring Enabled API"
    }
