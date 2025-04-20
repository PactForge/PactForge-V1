# backend.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from app_logic import (
    strip_type,
    perform_analysis,
    obtain_information_holes,
    get_data,
    generate_embeddings,
    initialize_chromadb,
    extract_samples,
    generate_agreement,
)
import os

app = Flask(__name__)
CORS(app)

# Initialize ChromaDB and client
clientdb, all_dbs_dict = initialize_chromadb()

@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.get_json()
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({'response': 'Please enter a message.'}), 400

    global current_step
    if not hasattr(send_message, 'current_step'):
        send_message.current_step = 0
        send_message.agreement_type = None
        send_message.important_info = ""
        send_message.extra_info = ""

    if send_message.current_step == 0:
        send_message.agreement_type = strip_type(user_message)
        list_agreements = ["rent", "contractor", "employment", "nda", "franchise"]
        if send_message.agreement_type in list_agreements:
            send_message.current_step = 1
            return jsonify({'response': f"Okay, let's create a {send_message.agreement_type} agreement. What are the important details?"})
        else:
            return jsonify({'response': "That agreement type is not supported. Please choose from: rent, nda, franchise, contractor, employment."})

    elif send_message.current_step == 1:
        send_message.important_info = user_message
        send_message.current_step = 2
        return jsonify({'response': "Got it. Are there any specific clauses or extra details you'd like to add?"})

    elif send_message.current_step == 2:
        send_message.extra_info = user_message
        analysis_text, analysis_positive = perform_analysis(send_message.agreement_type, send_message.important_info)
        if analysis_positive:
            dbname = f"{send_message.agreement_type}_agreements"
            querydb = all_dbs_dict[send_message.agreement_type]
            user_query = send_message.important_info + send_message.extra_info
            query_embed = generate_embeddings([user_query], False)
            results = querydb.query(query_embeddings=query_embed, n_results=querydb.count())
            relevant_documents = results['documents']

            sample_agreements = extract_samples(send_message.agreement_type)

            holes = obtain_information_holes(send_message.important_info, send_message.extra_info, send_message.agreement_type)
            obtained_info = get_data(holes, send_message.agreement_type)

            generated_text = generate_agreement(send_message.important_info, send_message.extra_info, send_message.agreement_type, relevant_documents, sample_agreements, obtained_info)

            send_message.current_step = 0 # Reset for the next agreement
            return jsonify({'response': f"Here is the generated agreement:\n\n{generated_text}"})
        else:
            send_message.current_step = 1 # Go back to asking for important info
            return jsonify({'response': f"{analysis_text} Please provide more details."})

    return jsonify({'response': 'Something went wrong.'}), 500

if __name__ == '__main__':
    app.run(debug=True)