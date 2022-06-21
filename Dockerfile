FROM python:3.8-slim-buster

WORKDIR /mzmarsly

COPY requirements.txt .

RUN pip install -r requirements.txt
  
CMD ["pip3","install", "torch==1.10.1+cu111", "torchvision==0.11.2+cu111", "torchaudio==0.10.1", "--extra-index-url", "https://download.pytorch.org/whl/cu113"]

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