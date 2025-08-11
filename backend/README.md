Using command prompt
Create virtual environment

python -m venv .venv

Activate virtual environment

.venv\Scripts\activate.bat

Install packages 
pip install -r requirements.txt

Run the project
uvicorn app.main:app --reload --port 8000





