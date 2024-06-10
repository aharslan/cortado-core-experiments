FROM python:3.10

WORKDIR /usr/src/app

COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install cvxopt

ENV PYTHONPATH "${PYTHONPATH}:/usr/src/app"
ENV LOG_FILE 'BPI_Challenge_2012.xes'
ENV EVENT_LOG_DIRECTORY "/usr/data/"
ENV RESULT_DIRECTORY "/usr/results/"
ENV FREQ_COUNT_STRAT "3"
ENV PATTERN_TYPE "1"
ENV SORTING "1"
ENV MIN_SUPPORT "0.1"
ENV REL_SUPP_INIT_MODEL "0.01"
ENV NOISE_THRESHOLD_INIT_MODEL "0"

COPY . .

CMD [ "python", "./cortado_core/experiments/trace_fragments/experiments_artificial.py" ]
