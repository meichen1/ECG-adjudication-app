
from flask import Flask
# from flask_caching import Cache
from itsdangerous import URLSafeSerializer, BadSignature
import dash
# import asyncio
from dash import html, dcc
from dash.dependencies import Input, Output, State
import urllib
import os
import pandas as pd
# from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import datetime
from applayout import applayout_aftervalidation, SPREADSHEET_ID


#########################################################################################################
## get the environment variables batch and dash name
batch_num = int(os.getenv('BATCH')) if os.getenv('BATCH') else ValueError('BATCH number not provided')
dash_app_name = os.getenv('NAME', 'DashApp')+' Batch: ' + str(batch_num)  # Get the Dash app name


############################################################################################################
# Initialize Flask server and Dash app
server = Flask(__name__)
app = dash.Dash(__name__, server=server)
app.title = dash_app_name


IMAGE_DIR = '/workspace/ecgimgs/UnCertain/'# Directory containing images
masterlistcsvFile = '/workspace/ECGXML/uncertain_files_fullpath.csv'

# Get list of image files
processed_files = pd.read_csv(f'{IMAGE_DIR}processed_files_batch{str(batch_num)}.csv').values.tolist()
image_files = [img for xml, img in processed_files]


###########################################################################################################


# Configure Google Sheets API
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SERVICE_ACCOUNT_FILE = os.getenv('SERVICE_ACCOUNT_FILE') 
if not SERVICE_ACCOUNT_FILE:
    raise ValueError('SERVICE_ACCOUNT_FILE environment variable is not set')
# SPREADSHEET_ID = os.getenv('SPREADSHEET_ID')
if not SPREADSHEET_ID:
    raise ValueError('SPREADSHEET_ID environment variable is not set')
RANGE_NAME = 'Uncertain!A1:J3000' 

# Secret key for token generation and validation
secret_key = os.getenv('SECRET_KEY_FOR_TOKEN') 
if not secret_key:
    raise ValueError('SECRET_KEY environment variable is not set')
serializer = URLSafeSerializer(secret_key)
# Load credentials
creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)





layout_initial = html.Div([
        dcc.Location(id='url', refresh=False),
        html.Div(id='page-content', children=[html.H2('Verifying...')]),
        html.Div(id='hidden-div', style={'display': 'none'}, 
        children=[
        html.Button('Submit-fake', id='submit-button', n_clicks=0),
        *[dcc.Input(id=f'adjudication-{i}', type='text', value='') for i in range(len(image_files))] +
        [dcc.Input(id=f'other-findings-{i}', type='text', value='') for i in range(len(image_files))] +
        [dcc.Input(id=f'comments-{i}', type='text', value='') for i in range(len(image_files))]+
        [dcc.Input(id=f'confidencelevel-{i}', type='text', value='') for i in range(len(image_files))],

        dcc.Input(id='physicians', type='text', value=''),
        dcc.DatePickerSingle(id='date-picker', date=datetime.datetime.now().date())
        ]),
        html.Div(id='output-message')])

app.layout = layout_initial
    
    
# Callback to validate token and render content
@app.callback(
        Output('page-content', 'children'),
        [Input('url', 'search')],
        prevent_initial_call=True, 
        suppress_callback_exceptions=True
    )
def display_page(search):
    # Extract token from URL parameters
    query_params = urllib.parse.parse_qs(search.lstrip('?'))
    token = query_params.get('token', [None])[0]
    # Debugging statement to check if token is extracted
    print(f'Token: {token}')
    if not token:
        return html.Div([
            html.H2('Error: Token not found'),
        ])
    
    # Validate token
    try:
        data = serializer.loads(token)
        user = data['user']
        return applayout_aftervalidation(image_files, IMAGE_DIR, user=user)
    except BadSignature:
        return html.Div([
            html.H2('Error: Invalid token.')
        ])


@app.callback(
    Output('output-message', 'children'),
    [Input('submit-button', 'n_clicks')],
    [State(f'adjudication-{i}', 'value') for i in range(len(image_files))] +
    [State(f'other-findings-{i}', 'value') for i in range(len(image_files))] +
    [State(f'comments-{i}', 'value') for i in range(len(image_files))] + 
    [State(f'confidencelevel-{i}', 'value') for i in range(len(image_files))],
    [State('physicians', 'value'), State('date-picker', 'date')],
    prevent_initial_call=True,
    suppress_callback_exceptions=True
)
def submit_ratings(n_clicks, *ratings):
    print(ratings)

    if n_clicks > 0:
        num_images = len(image_files)
        adjudications = [ratings[i] for i in range(0, num_images)]
        other_findings = [ratings[i] for i in range(num_images, 2*num_images)]
        comments = [ratings[i] for i in range(2*num_images, 3*num_images)]
        confidencelevels = [ratings[i] for i in range(3*num_images, 4*num_images)]
        physician = ratings[-2]
        date = ratings[-1]
        try:    
            # Check if any adjudication, physician, or date is None
            if any(adjudication is None for adjudication in adjudications) or any(confidencelevel is None for confidencelevel in confidencelevels) or physician is None or physician.strip() == '' or date is None:
                return dcc.Markdown('''
                * Please make sure to enter your **name, date**.                   
                * And select the **adjudication** and **certainty level** for each image.
                ''')
            
            # Prepare data for Google Sheets
            values = []
            for i in range(num_images):
                row = [processed_files[i][0], image_files[i], adjudications[i], ','.join(other_findings[i] or []), comments[i], confidencelevels[i], physician, date, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), batch_num]
                values.append(row)
                
            ## save the data to processed_files as a backup
            processed_files_df = pd.DataFrame(values, columns=['xml_file', 'img_file', 'adjudication', 'other_findings', 'comments', 'confidencelevel', 'physician', 'date', 'timestamp', 'batch'])
            processed_files_df.to_csv(f'{IMAGE_DIR}adjudicated_batch{str(batch_num)}.csv', index=False)
            
            
            service = build('sheets', 'v4', credentials=creds)
            body = {'values': values}
            # Find the smallest blank row and update without rewriting the whole sheet
            sheet = service.spreadsheets()
            result = sheet.values().get(spreadsheetId=SPREADSHEET_ID, range=RANGE_NAME).execute()
            values_existing = result.get('values', [])
            
            # Find the first empty row
            first_empty_row = len(values_existing) + 1
            
            # Update Google Sheet starting from the first empty row
            update_range = f'Uncertain!A{first_empty_row}:J{first_empty_row + len(values) - 1}'
            result = sheet.values().update(
                spreadsheetId=SPREADSHEET_ID,
                range=update_range,
                valueInputOption='RAW',
                body=body
            ).execute()
            
            # If successful, perform additional steps
            if result.get("updatedCells") > 0:
                # Additional steps here
                print("Successfully updated Gsheet.")
            
            google_sheet_url = f'https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}'
            
            return dcc.Markdown('''
            Adjudications submitted successfully! Thank you.
            
            View the Google Sheet:
            '''+
            google_sheet_url
            )
            
        except HttpError as error:
            return f'An error occurred: {error}'
    
    return ''




if __name__ == '__main__':
    
    port_num = 8050 + batch_num
    app.run_server(debug=False, host='0.0.0.0', port=port_num)
    # asyncio.run(main())

    