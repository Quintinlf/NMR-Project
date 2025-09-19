from __future__ import print_function
import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# If modifying Docs/Drive, you need these scopes
SCOPES = ['https://www.googleapis.com/auth/documents',
          'https://www.googleapis.com/auth/drive.file']

def authenticate_google():
    creds = None
    # token.json stores your access/refresh tokens
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials, log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

def create_google_doc(title="Lab Report"):
    creds = authenticate_google()
    service = build('docs', 'v1', credentials=creds)
    doc = service.documents().create(body={'title': title}).execute()
    print(f"Created document: {doc['title']} (ID: {doc['documentId']})")
    return doc['documentId']

if __name__ == "__main__":
    doc_id = create_google_doc("Test Chemical Report")
    print("Open this Doc ID in your browser to see your new document.")
