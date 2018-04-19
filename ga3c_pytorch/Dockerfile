FROM openai/retro-agent:pytorch

# OpenCV dep
RUN apt-get update && \
    apt-get install -y libgtk2.0-dev python3-dev zlib1g-dev libsnappy-dev && \
    rm -rf /var/lib/ap/lists/*

# Retro
RUN . ~/venv/bin/activate && \
    pip install cmake setproctitle opencv-python && \
    git clone --recursive https://github.com/openai/retro.git gym-retro && \
    cd gym-retro && \
    pip install -e . && \
    cd .. && \
    git clone --recursive https://github.com/openai/retro-contest.git && \
    pip install -e "retro-contest/support[docker]"

ADD main.py model.py player_util.py shared_optim.py sonic_config.json sonic_util.py test.py train.py utils.py ./
VOLUME /root/compo/logs /root/compo/trained_models

COPY Sonic.tar.gz .
RUN tar -xzf Sonic.tar.gz -C /root/compo/gym-retro/data/ && \
    rm Sonic.tar.gz

CMD ["python", "-u", "/root/compo/main.py", "--workers", "32", "--gpu-ids", "0 1 2 3"]
