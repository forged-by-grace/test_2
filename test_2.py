# Import dependencies
from flask import Flask, request, jsonify
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI
import pandas as pd
from config import OPENAI_API_KEY


# Initialize flask app
app = Flask(__name__)

# Task 1:  Import sales_data_sample.csv into panda and create a dataframe
def create_dataframe(filename: str):   

    # Read data from csv file
    df = pd.read_csv(f'./{filename}.csv', encoding = "ISO-8859-1")
    return df

# Task 2: Make a flask api that we can ask natural language question and we can give back a human answer based on the data in your dataframe. 
@app.route('/query-dataset', methods=['POST'])
def query_dataset(): 
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        
        # Get request body
        body = request.json

        # Process the request
        response = create_prompt(query=body.get('query'), dataset='sales_data_sample')
        
        return jsonify({'data': response})  
    else:
        return 'Content-Type not supported!' 


def create_prompt(query, dataset):
    # Get dataframe
    df = create_dataframe(dataset)    

    # Init pandas_ai 
    llm = OpenAI(api_token=OPENAI_API_KEY)    
    
    # Convert df to smart df
    sdf = SmartDataframe(df, config={"llm": llm})

    # Return response
    response = sdf.chat(query=query)

    return response


# Task 3: Make a flask api that we can ask natural language question and we can give back a human answer based on the data in your dataframe. 
@app.route('/filter/get-customer-gender', methods=['POST'])
def get_customer_gender(): 
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):        
        # Get body
        body = request.json

        # Get orderid from body
        order_id = body.get('orderid')

        # Create query
        query = f"What is the gender of the customer with orderid '{order_id}'"

        # Get dataset
        dataset = 'orders'

        # Process the request
        response = create_prompt(query=query, dataset=dataset)
        
        return jsonify({'data': response})  
    else:
        return 'Content-Type not supported!' 


# Task 4: make a simple recommendation engine where user can pass in a month to the flask api and get back a recommendation for a trend happening or anomaly in that month.
@app.route('/month-trends', methods=['POST'])
def get_trends(): 
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        # Get body
        body = request.json
        
        # Get orderid from body
        month = body.get('month')

        # Create query
        query = f"What are the trends or anomaly in the month '{month}'"

        # Get dataset
        dataset = 'sales_data_sample'

        # Process the request
        response = create_prompt(query=query, dataset=dataset)
        
        return jsonify({'data': response})  
    else:
        return 'Content-Type not supported!' 
    

# Configure the server
if __name__ == '__main__':
    app.run(debug=True)

