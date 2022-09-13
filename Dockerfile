FROM python:3.8
RUN wget "https://www.dropbox.com/s/d7loj89txe1sowb/WilhemNet_86.h5?dl=0"  -o WilhemNet_86.h5
RUN mv ./WilhemNet_86.h5 backend/Inference/
ADD . /code
WORKDIR /code
RUN apt update -y && \
  apt-get install python3-opencv -y
RUN pip3 install -r requirements.txt
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]