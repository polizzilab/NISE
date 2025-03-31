import sys
import traceback
from flask import Flask, request, jsonify

from batch_inference import load_model_and_modules, predict_structures, create_pdb_strings
app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello():
    return "Hello, World!"


@app.route('/boltz', methods=['POST'])
def get_data():
    json_output = jsonify({})
    try:
        data = request.get_json()
        predicted_structure_data, collated_batch, all_structures = predict_structures(data['sequences'], LIGAND_SMILES, DEVICE, model, tokenizer, featurizer, ccd, predict_args)
        pdb_strs = create_pdb_strings(predicted_structure_data, collated_batch, all_structures)
        json_output = jsonify({'pdb_strs': pdb_strs})
    except Exception as e:
        with open(f'worker_{PORT}_error.txt', 'w') as f:
            traceback.print_exc(file=f)
            f.write(str(e))
    return json_output


if __name__ == '__main__':
    PORT = int(sys.argv[1])
    NUM_RECYCLES = int(sys.argv[2])
    NUM_DIFFUSION_STEPS = int(sys.argv[3])
    LIGAND_SMILES = sys.argv[4]
    DEVICE = sys.argv[5]

    predict_args = {
        'recycling_steps': NUM_RECYCLES,
        'sampling_steps': NUM_DIFFUSION_STEPS,
        'diffusion_samples': 1,
    }

    model, ccd, tokenizer, featurizer = load_model_and_modules(DEVICE, predict_args)
    app.run(debug=True, port=PORT)
