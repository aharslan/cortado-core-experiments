FROM python:3.10

WORKDIR /usr/src/app

COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install cvxopt

ENV PYTHONPATH "${PYTHONPATH}:/usr/src/app"
ENV TIMEOUT "300"
ENV LOG_FILE 'BPI_Challenge_2012.xes'
ENV EVENT_LOG_DIRECTORY "/usr/data/"
ENV RESULT_DIRECTORY "/usr/results/"

COPY . .

CMD [ "python", "./cortado_core/experiments/eventually_follows_pattern_mining/run_all.py" ]