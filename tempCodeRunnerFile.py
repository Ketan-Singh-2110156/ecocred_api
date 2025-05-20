from flask import Flask, request, jsonify
from web3 import Web3
from joblib import load
import numpy as np
from dotenv import load_dotenv
from flask_cors import CORS
import os