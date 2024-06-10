# Infix alignment experiments

This file explains how to run the infix alignment experiments.

## Build the docker image

Navigate to the root directory of this repository and run the following command:

```shell
docker build -t registry.fit.fraunhofer.de/cortado/cortado-core/infix-alignment-experiments:latest -f ./cortado_core/experiments/infix_alignments/Dockerfile .
```

## Push the docker image

If you want to update the docker image in the docker registry of the Gitlab-project, run the following commands:

```shell
docker login registry.fit.fraunhofer.de
docker push registry.fit.fraunhofer.de/cortado/cortado-core/infix-alignment-experiments:latest
```

## Run the docker image

Use the following command to run the image:

```shell
docker run -it -v <your dir with event logs>:/usr/data registry.fit.fraunhofer.de/cortado/cortado-core/infix-alignment-experiments:latest
```

This commands mounts `<your dir with event logs>` into the docker container and creates a csv-file containing the results in the same directory. The input file has to be named `input.xes` in the default case, but it is possible to use different filenames by overwriting an environment variable (see below).

The following environment variables can be overwritten in the docker run command:

| Variable           | Default Value   | Description                                                                                                                                                                                                                                                                      |
|--------------------|-----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| DATA_PATH          | /usr/data/      | Name of directory in the container that contains the data.
| DATA_FILENAME      | input           | Name of the event log. Note that the extension is NOT part of the name. If there is a process tree with the same name in the directory (a ptml-file), the experiment script uses the process tree instead of generating a process tree. |
| MODE               | file            | 'file' runs the experiments for the given DATA_FILENAME. 'dir' runs the experiments for all event logs in 'DATA_PATH'. |
| NOISE_THRESHOLD    | 0.9             | Noise threshold used for the generating of a process tree using the inductive miner algorithm.                                                                                                                                                                                   |
| RANDOM_SAMPLE_SIZE | 10000           | Threshold for generating a random sample of all generated infixes.                                                                                                                                                                                                               |
| TIMEOUT            | 5               | Timeout per alignment computation.  
| INFIX_TYPE         | infix           | 'infix' for infix-alignments and 'postfix' for postfix-alignments  

The following example illustrates the usage of environment variables for the log `ccc19.xes` in the directory `C:\event-logs\` using a timeout of 10 seconds per infix.

```shell
docker run -it -v C:/event-logs:/usr/data -e DATA_FILENAME="ccc19" -e TIMEOUT="10" registry.fit.fraunhofer.de/cortado/cortado-core/infix-alignment-experiments:latest
```