From clearlinux/stacks-dlrs_2-mkl:latest


ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

ENV TFHUB_CACHE_DIR=/workspace/models/
COPY ./. /workspace/

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]

