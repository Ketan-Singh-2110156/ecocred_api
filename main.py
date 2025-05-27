from flask import Flask, request, jsonify
from web3 import Web3
from joblib import load
import numpy as np
from dotenv import load_dotenv
from flask_cors import CORS
import os

load_dotenv()
app = Flask(__name__)
CORS(app)

rpc_url = os.getenv("RPC_URL")
web3 = Web3(Web3.HTTPProvider(rpc_url))

if not web3.is_connected():
    raise Exception("Failed to connect to blockchain")

contract_address = os.getenv("CONTRACT_ADDRESS")
contract_abi = [
    {
        "inputs": [{"internalType": "uint256", "name": "id", "type": "uint256"}],
        "name": "getData",
        "outputs": [{"internalType": "uint256[]", "name": "", "type": "uint256[]"}],
        "stateMutability": "view",
        "type": "function"
    }
]

contract = web3.eth.contract(address=contract_address, abi=contract_abi)
id = 1

@app.route('/predict', methods=['GET'])
def predict_from_chain():
    try:
        data = contract.functions.getData(id).call()
        if len(data) < 2:
            return jsonify({"error": "Not enough data on-chain"}), 400
        
        val1, val2 = data[-2], data[-1]

        model = load('model.pkl')
        res = []
        res.append(model.predict(np.array([[val1, val2]])))
        res.append(model.predict(np.array([[res[-1][0], val2]])))
        res.append(model.predict(np.array([[res[-2][0], res[-1][0]]])))
        predictions = [300 + (abs(float(r)) % 200) for r in res]
        return jsonify({"input": [val1, val2], "prediction": predictions}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return jsonify({"message": "Ready to fetch from chain and predict"}), 200

if __name__ == '__main__':
    app.run(debug=True)
