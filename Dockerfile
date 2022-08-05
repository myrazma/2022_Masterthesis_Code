FROM python:3.8-slim-buster
# FROM nvidia/cuda:11.4.2-base-ubuntu20.04

WORKDIR /src

COPY requirements.txt .

RUN pip install torch==1.11.0+cu115 --find-links https://download.pytorch.org/whl/torch_stable.html
RUN pip install -r requirements.txt

# follwing is only needed for deployment
# copy python files 
# use if no volume is being used
#COPY ./model/baseline_BERT.py /mzmarsly/model/baseline_BERT.py
#COPY ./model/multiinput_BERT.py /mzmarsly/model/multiinput_BERT.py
#COPY ./utils.py /mzmarsly/utils.py
#RUN mkdir -p /mzmarsly/output

# copy data
#COPY ./data/buechel_empathy /mzmarsly/data/buechel_empathy
#COPY ./data/lexicon/empathy /mzmarsly/data/lexicon/empathy
#COPY ./data/lexicon/distress /mzmarsly/data/lexicon/distress

# run selected file
# use if no bash is being used
#CMD ["python","-u", "./model/baseline_BERT.py", "distress"]