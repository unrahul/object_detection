From clearlinux/stacks-dlrs_2-mkl:latest


ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

COPY ./. /workspace/

RUN pip install --no-cache-dir -r requirements.txt

RUN echo -e "\e[31mThis server should only be used for debugging purposes."
ENTRYPOINT ["python", "-c ", "hypercorn","-kuvloop", "-b0.0.0.0:5059", "--debug", "-w1"]
CMD ["rest:app"]

