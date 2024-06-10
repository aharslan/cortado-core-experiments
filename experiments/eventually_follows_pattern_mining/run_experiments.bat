:: docker build -t registry.fit.fraunhofer.de/cortado/cortado-core/ef-experiments:latest -f ./cortado_core/experiments/eventually_follows_pattern_mining/Dockerfile .
:: docker push registry.fit.fraunhofer.de/cortado/cortado-core/ef-experiments:latest
:: .\cortado_core\experiments\eventually_follows_pattern_mining\run_experiments.bat BPI_Challenge_2012.xes

docker run -it -e LOG_FILE=%1 -v C:\sources\arbeit\cortado\event-logs:/usr/data -v C:\sources\arbeit\cortado\master_thesis:/usr/results registry.fit.fraunhofer.de/cortado/cortado-core/ef-experiments:latest