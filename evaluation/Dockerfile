FROM python:3.11
WORKDIR /app
COPY . /app/

# RUN pip install hdbscan
RUN pip install -r requirements.txt

# Make port 6006 available to the world outside this container
EXPOSE 6006 


HEALTHCHECK CMD curl --fail http://localhost:6006/_stcore/health

# Set the entrypoint
ENTRYPOINT ["python", "phoenix_run.py"]
