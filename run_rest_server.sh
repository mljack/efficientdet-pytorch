#!/bin/bash

export FLASK_APP=server.py
flask run --host=0.0.0.0 --without-threads
