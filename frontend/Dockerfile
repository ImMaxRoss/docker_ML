FROM python:3.11

WORKDIR /app

COPY . .

RUN pip3 install pandas
RUN pip3 install scikit-learn
RUN pip3 install scikit-surprise
RUN pip3 install streamlit

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
