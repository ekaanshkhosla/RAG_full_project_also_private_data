import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.api.queries import process_question  # Import the function to handle questions

app = FastAPI()

# Mount static files like CSS and images
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# API endpoint for handling the question from the UI
@app.post("/get-answer", response_class=HTMLResponse)
async def get_answer(request: Request, question: str = Form(...)):
    answer = process_question(question)  # Replace this with your actual logic
    return templates.TemplateResponse("index.html", {"request": request, "answer": answer, "question": question})
