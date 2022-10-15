#! /bin/bash
touch /workspace/run.log
nohup jupyter notebook --allow-root > /workspace/run.log 2>&1 