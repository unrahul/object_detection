FROM openfaas/of-watchdog:0.8.2 as watchdog
From clearlinux/stacks-dlrs_2-mkl:latest

COPY --from=watchdog /fwatchdog /usr/bin/fwatchdog
RUN chmod +x /usr/bin/fwatchdog

# Add non root user
RUN addgroup -S app && adduser app -S -G app
RUN chown app /home/app

USER app
ENV PATH=$PATH:/home/app/.local/bin

WORKDIR /home/app/
COPY ./. .

USER root
RUN pip install --no-cache-dir -r requirements.txt
USER app

ENV fprocess="python rest.py"
ENV cgi_headers="true"
ENV mode="http"
ENV upstream_url="http://127.0.0.1:5059"

HEALTHCHECK --interval=3s CMD [ -e /tmp/.lock ] || exit 1

CMD ["fwatchdog"]
