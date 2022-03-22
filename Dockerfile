FROM python:3.8

WORKDIR /mzmarsly

COPY requirements.txt .

RUN pip install -r requirements.txt

# copy python files
COPY ./model/baseline_BERT.py ./model/baseline_BERT.py
COPY ./utils.py ./utils.py
# copy data
COPY ./data/buechel_empathy ./data/buechel_empathy

# run selected file
CMD ["python", "./model/baseline_BERT.py"]