FROM pytorch/pytorch

RUN apt-get update && apt-get install -y tmux python-qt4
RUN pip install tensorflow==1.6.0 tensorboardX pandas graphviz matplotlib 
RUN pip install pyro-ppl jupyter jupytext Pillow
RUN pip install scipy scikit-learn
RUN jupyter notebook --generate-config
RUN echo 'c.NotebookApp.contents_manager_class = "jupytext.TextFileContentsManager"' > /root/.jupyter/jupyter_notebook_config.py

