#!/bin/bash

# Get the container ID of the running minio container
container_id=$(docker ps -qf "name=minio")

# Use docker inspect to get the IP address
ip_address=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$container_id")

echo "$ip_address"