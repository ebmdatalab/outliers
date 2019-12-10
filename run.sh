#!/bin/bash

docker build -t outliers -f config/outliers.Dockerfile .
docker run -ti -v ${PWD}:/usr/local/bin/outliers -p 8888:8888 outliers