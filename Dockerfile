FROM python:3.8-slim-buster

WORKDIR /mzmarsly

COPY requirements.txt .

RUN pip install -r requirements.txt

# copy python files
COPY ./model/baseline_BERT.py /mzmarsly/model/baseline_BERT.py
COPY ./utils.py /mzmarsly/utils.py
RUN mkdir -p /mzmarsly/output

# copy data
COPY ./data/buechel_empathy /mzmarsly/data/buechel_empathy

# run selected file
#CMD ["python","-u", "./model/baseline_BERT.py", "distress"]