#!/usr/bin/env bash

spark2-submit --conf spark.ui.port=5793 --driver-memory 12g ex.py
