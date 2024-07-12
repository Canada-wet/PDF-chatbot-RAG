FROM python:3.11
WORKDIR /app
COPY . /app/

RUN pip install -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501 


HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Set the entrypoint
ENTRYPOINT ["/bin/bash", "./docker-entrypoint.sh"]

# ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
