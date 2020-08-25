FROM python:3.7-slim
COPY . /app
WORKDIR /app
########################################################################################################################################
ENV PYTHONPATH '/app/Web-API:/app/Shared-Libraries/Python:/app/Data-Export:/app/Data-Management:/app/Data-Transform:/app/Analysis:/app/Reports:/app/Tests'
ENV LD_LIBRARY_PATH "/usr/local/lib/:LD_LIBRARY_PATH"
ENV ANSIBLE_CONNECTION local
ENV GIT_SSL_NO_VERIFY true
##################################
# Install EPEL repo
##################################
# Get basic dependencies, update OS
FROM centos:7
RUN yum -y install epel-release \
###################################
# Installing gdal1.11, psycopg2, wget
# gcc gcc-c++ python-devel make
# python-lxml gdal gdal-devel
# libxslt-devel libxml2-devel
###################################
&& yum install -y \
make \
gdal-python.x86_64 \
python-psycopg2.x86_64 \
wget \
gcc \
gcc-c++ \
python-devel \
python-lxml \
gdal \
gdal-devel \
libxslt-devel \
libxml2-devel \
python-reportlab \
postgres*dev* \
openssl* \
nodejs \
npm \
bzip2 \
git \
sudo
##################################
# Adding get-pip.py,
# libspatialindex & pipinstall.sh
# Update Pip
##################################
RUN wget https://bootstrap.pypa.io/get-pip.py \
&& mkdir /tmp/spatialindex \
&& wget --directory-prefix=/tmp/spatialindex/ http://download.osgeo.org/libspatialindex/spatialindex-src-1.7.0.tar.gz \
&& python get-pip.py
##################################
# Build SpatialIndex
# Credit pvillard/agdc-docker
#################################
RUN tar -xvzf /tmp/spatialindex/spatialindex-src-1.7.0.tar.gz -C /tmp/spatialindex \
&& cd /tmp/spatialindex/spatialindex-src-1.7.0/ \
&& ./configure; make; make install \
&& echo export LD_LIBRARY_PATH=/usr/local/lib/ >> /root/.bashrc && source /root/.bashrc \
&& pip install -U Rtree && yum install -y which
########################################################################################################################################
RUN pip install enum-compat==0.0.2\
&& pip install urllib3==1.25.10\
&& pip install chardet==3.0.4\
&& pip install certifi==2020.6.20\
&& pip install idna==2.10\
&& pip install functools32==3.2.3-2\
&& pip install backports.weakref==1.0rc1\
&& pip uninstall tensorflow-tensorboard\
&& pip install tensorflow-tensorboard==1.5.1 --ignore-installed numpy\
&& pip install tensorboard==2.1.0 --ignore-installed numpy\
&& pip install gast==0.2.2\
&& pip install Keras-Preprocessing==1.1.2\
&& pip install termcolor==1.1.0\
&& pip install scipy==1.2.2\
&& pip install google-pasta==0.2.0\
&& pip install opt-einsum==2.3.2\
&& pip install wrapt==1.12.1\
&& pip install tensorflow-estimator==2.1.0\
&& pip install Keras-Applications==1.0.8\
&& pip install mock==3.0.5\
&& pip install geographiclib==1.50\
&& pip install branca==0.4.1\
&& pip install watchdog==0.10.3\
&& pip install validators==0.14.2\
&& pip install tornado==5.1.1\
&& pip install pydeck==0.4.1\
&& pip install astor==0.8.1\
&& pip install future==0.18.2\
&& pip install pandas==0.24.2\
&& pip install altair==3.3.0\
&& pip install python-dateutil==2.8.0\
&& pip install toml==0.10.1\
&& pip install Pillow==6.2.2\
&& pip install base58==1.0.3\
&& pip install tzlocal==2.1\
&& pip install botocore==1.17.45\
&& pip install boto3==1.14.45\
&& pip install click==7.1.2\
&& pip install networkx==2.1\
&& pip install blinker==1.4\
# && pip install scikit-image==0.17.2\
&& pip install --no-deps scikit-image\
&& pip install --upgrade reverse_geocoder

########################################################################################################################################
# INSTALLING OSMNX WITH CONDA
FROM ubuntu:18.04
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version\
&& conda install -c conda-forge osmnx

########################################################################################################################################
 # copy over requirements
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8501
# making directory of app
WORKDIR /app
# copying all files over
COPY . .
# cmd to launch app when container is run
CMD streamlit run trashout.py --server.port $PORT
