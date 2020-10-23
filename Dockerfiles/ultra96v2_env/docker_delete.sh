#!/bin/sh
docker ps -aq | xargs docker rm
docker images -aq | xargs docker rmi
