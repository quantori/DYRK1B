FROM continuumio/miniconda3
RUN conda install -c conda-forge rdkit
RUN apt update && apt install -y wget

ADD requirements.in /app/src/
RUN python3 -m pip install -r /app/src/requirements.in

# copy source code
COPY . /app/src