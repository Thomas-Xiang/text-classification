FROM python:3.10-slim

ENV APP_HOME /app
WORKDIR $APP_HOME

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API code and the trained model directory
COPY src/ src/
COPY tfidf_logreg_classifier tfidf_logreg_classifier/

EXPOSE 80

# Command to run the application using Uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "80"]