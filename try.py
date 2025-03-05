from flask import Flask, render_template, request, jsonify, session
import nest_asyncio
import os
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

# Apply nest_asyncio for running async code in sync environment
nest_asyncio.apply()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for using session

# Set your API keys
os.environ['OPENAI_API_KEY'] = 'OPENAI_API_KEY'
llama_api_key = 'LLAMA_API_KEY'

# Initialize LlamaParse
parser = LlamaParse(api_key=llama_api_key, result_type="markdown")

# Load the documents from a directory
documents = SimpleDirectoryReader(input_files=["data/attention.pdf"]).load_data()

# Create index for querying
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

@app.route('/')
def index():
    # Initialize chat history in session if not present
    if 'chat_history' not in session:
        session['chat_history'] = []
    return render_template('index.html', chat_history=session['chat_history'])

@app.route('/query', methods=['POST'])
def query():
    user_query = request.form.get('query')
    response = query_engine.query(user_query)

    # Format the response
    bot_response = response.response

    # Save user message and bot response to chat history
    session['chat_history'].append({"user": user_query, "bot": bot_response})

    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)
