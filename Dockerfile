FROM pytorch/pytorch

RUN apt-get update
RUN pip install tensorflow tensorboardX pandas graphviz matplotlib pyro-ppl

