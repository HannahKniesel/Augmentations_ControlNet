FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install diffusers==0.21.
RUN pip install numpy==1.26.1
RUN pip install opencv-python==4.8.1.78
RUN pip install pillow==10.1.0
RUN pip install matplotlib==3.8.0
RUN pip install transformers==4.34.1
RUN pip install pathlib
RUN pip install accelerate
RUN pip install regex==2023.10.3                
RUN pip install ftfy


 