FROM pytorch/pytorch

RUN apt-get update && apt-get install -y tmux
RUN pip install tensorflow tensorboardX pandas graphviz matplotlib pyro-ppl

