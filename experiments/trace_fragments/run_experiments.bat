:: docker build -t registry.fit.fraunhofer.de/cortado/cortado-core/trace-fragments-experiments:latest -f ./cortado_core/experiments/trace_fragments/Dockerfile .
:: docker push registry.fit.fraunhofer.de/cortado/cortado-core/trace-fragments-experiments:latest
:: .\cortado_core\experiments\trace_fragments\run_experiments.bat BPI_Challenge_2012.xes

docker run -it -e LOG_FILE=%1 -v C:\sources\arbeit\cortado\event-logs:/usr/data -v C:\sources\arbeit\cortado\trace_fragments:/usr/results registry.fit.fraunhofer.de/cortado/cortado-core/trace-fragments-experiments:latest
