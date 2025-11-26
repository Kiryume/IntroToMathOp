FROM sagemath/sagemath:10.7

WORKDIR /home/sage/project

# This gets overrided by the volume
COPY --chown=sage:sage ./lab .

RUN sage -pip install jupyterlab

EXPOSE 8888

CMD ["sage", "-n", "jupyterlab", "--no-browser", "--ip=0.0.0.0", "--port=8888", "--NotebookApp.token=''", "--NotebookApp.password=''"]
