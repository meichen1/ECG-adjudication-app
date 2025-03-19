from dash import html, dcc
import base64
import os

SPREADSHEET_ID = os.getenv('SPREADSHEET_ID')
batch_num = os.getenv('BATCH')
google_sheet_url = f'https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}'

def encode_image(image_path):
    with open(image_path, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')
    return f'data:image/jpeg;base64,{encoded}'



def applayout_aftervalidation(image_files, IMAGE_DIR, user):
    return html.Div(
        [
            html.H1("WCT ECG Adjudication", style={'text-align': 'center', 'margin-top': '20px', 'margin-bottom': '20px', 'font-size': '2em'}),
            html.H6(f"Batch {batch_num} - 50 files", style={'text-align': 'center', 'margin-top': '20px', 'margin-bottom': '20px', 'font-size': '1.5em'}),
            dcc.Markdown('''
                 * Thank you for adjudicating ECGs for the AI-WCT project. Please review the wide complex tachycardia (WCT) below and select the most appropriate diagnosis. 
                 * Disagreements will be presented at rounds for consensus (adjudicator selections will be confidential).
                 * Your adjudications will be uploaded to this Google Sheet for further analysis:
                 ''' +
                 google_sheet_url +
                 '''
                 * Thank you for your expertise and time!
                 ''', style={'margin-top': '20px', 'margin-bottom': '20px'}),
            html.Div(
                "Please click the submit button below to save your results!",
                style={
                    'backgroundColor': 'rgba(255, 99, 71, 0.7)',
                    'color': 'black',
                    'padding': '10px',
                    'fontWeight': 'bold',
                    'textAlign': 'center',
                    'fontSize': '1.2em',
                    'margin': '20px 0'
                }
            ),
            html.Div([
                html.Div([
                    html.Label('Physician Name: (required for submission)'),
                    dcc.Textarea(
                        id='physicians',
                        placeholder=f'Enter {user} Name here...',
                        style={'width': '100%', 'height': '40px'}
                    ),
                ], style={'width': '47%', 'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '2%'}),
                html.Div([
                    html.Label('Date: (required for submission)'),
                    dcc.DatePickerSingle(
                        id='date-picker',
                        placeholder='Select a date',
                        style={'width': '100%', 'height': '40px'}
                    ),
                ], style={'width': '49%', 'display': 'inline-block', 'vertical-align': 'top'}),
            ], style={'width': '100%', 'display': 'flex', 'justify-content': 'space-between'}),
            html.Br(),
            html.Div([
                html.Div([
                    dcc.Markdown(
                        f"{str(i+1)}. Patient PIN: {img.split('_')[0]}; Date: {' '.join(img.split('_')[1:3])}", style={'fontSize': '0.8em','color': 'LightSlateGray'}
                    ),
                    html.Img(src=encode_image(os.path.join(IMAGE_DIR, img)),
                             style={'width': '100%', 'height': 'auto', 'display': 'inline-block', 'margin': '10px'}),
                    html.Div([
                        html.Div([
                            html.Label('Adjudication: (required for submission)'),
                            html.Div(style={'height': '10px'}),
                            dcc.RadioItems(
                                id=f'adjudication-{i}',
                                options=[
                                    {'label': 'Ventricular Tachycardia (VT)', 'value': 'vt'},
                                    {'label': 'Supraventricular Tachycardia (SVT)', 'value': 'svt'},
                                    {'label': 'Pacing', 'value': 'pacing'},
                                    {'label': 'Uncertain', 'value': 'uncertain'},
                                ],
                                inline=False,
                            ),
                        ], style={'width': '48%', 'display': 'inline-block', 'margin-left': '1%'}),
                        html.Div([
                            dcc.Dropdown(
                                id=f'other-findings-{i}',
                                options=[
                                    {'label': 'Atrial Tachycardia (AT)', 'value': 'AT'},
                                    {'label': 'AV reentrant tachycardia (AVRT)', 'value': 'AVRT'},
                                    {'label': 'AV nodal reentrant tachycardia (AVNRT)', 'value': 'AVNRT'},
                                    {'label': 'Atrial Flutter (AFL)', 'value': 'AFL'},
                                    {'label': 'Atrial Fibrillation (AF)', 'value': 'AF'},
                                    {'label': 'Sinus Rhythm/Tachycardia (SR)', 'value': 'SR'},
                                    {'label': 'Other (OT)', 'value': 'OT'},
                                    {'label': 'RBBB', 'value': 'RBBB'},
                                    {'label': 'LBBB', 'value': 'LBBB'},
                                    {'label': 'LAFB', 'value': 'LAFB'},
                                    {'label': 'LPFB', 'value': 'LPFB'},
                                    {'label': 'LVH', 'value': 'LVH'},
                                    {'label': 'IVCD', 'value': 'IVCD'},
                                    {'label': 'Pre-Excited', 'value': 'Pre-Excited'},
                                    {'label': 'Frequent PACs', 'value': 'Frequent PACs'},
                                    {'label': 'Frequent PVCs', 'value': 'Frequent PVCs'},
                                ],
                                multi=True,
                                placeholder="Other findings (multi-select)",
                            ),
                            html.Label('Comments:'),
                            dcc.Textarea(
                                id=f'comments-{i}',
                                placeholder='Enter comments here...',
                                style={'width': '100%', 'height': '40px'}
                            ),
                        ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '1%'}),
                        html.Div([
                            html.Br(),
                            html.Label('How certain are you of the above diagnosis?'),
                            html.Div(style={'height': '10px'}),
                            dcc.RadioItems(
                                id=f'confidencelevel-{i}',
                                options=[
                                    {'label': 'Certain (>90% likely)', 'value': 'confidence90'},
                                    {'label': 'Somewhat Certain (70% ~ 90% likely)', 'value': 'confidence70'},
                                    {'label': 'Uncertain (50% equal likelihood of any dx)', 'value': 'confidence50'},
                                ],
                                inline=False,
                            ),
                        ], style={'width': '48%', 'display': 'inline-block', 'margin-left': '1%'}),
                        html.Br(),
                        html.Br(),
                        html.Br(),
                    ], style={'display': 'inline-block', 'vertical-align': 'top', 'width': '100%'})
                ]) for i, img in enumerate(image_files)
            ]),
            html.Div(
                "You have to click the submit button below to save your results!",
                style={
                    'backgroundColor': 'rgba(255, 99, 71, 0.7)',
                    'color': 'black',
                    'padding': '10px',
                    'fontWeight': 'bold',
                    'textAlign': 'center',
                    'fontSize': '1em',
                    'margin': '20px 0'
                }
            ),
            dcc.Markdown('''
                * Please make sure to enter your **name, date**, **adjudication** and **certainty level** for each ECG.
                * If successful, you will see a confirmation message below.
                '''),
            html.Button('Submit Ratings', id='submit-button', n_clicks=0, style={
                'fontWeight': 'bold',
                'margin': 'auto',
                'display': 'block',
                'padding': '10px 10px',
                'backgroundColor': 'floralwhite',
            })
        ]
    )
