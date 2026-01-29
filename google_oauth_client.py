import os
from pathlib import Path
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.credentials import Credentials


class GoogleOAuthClient:
    """Handles Google OAuth2 authentication."""
    
    def __init__(self, secrets_file: str, token_file: str):
        if not os.path.exists(secrets_file):
            raise FileNotFoundError(f"OAuth secrets file not found: {secrets_file}")
        
        if (token_file == "" or token_file is None):
            raise ValueError("Token file path must be provided.")
        
        self.secrets_file = secrets_file
        self.token_file = token_file

    def get_credentials(self, scopes: list[str]) -> Credentials:
        """
        Gets an OAuth2 credentials (access token) from Google, if required.
        Creates a new token or refreshes an existing one.
        """
        if not os.path.exists(self.token_file):
            oauth_flow = InstalledAppFlow.from_client_secrets_file(
                self.secrets_file, scopes
            )
            
            credentials = oauth_flow.run_local_server(
                port=9090,
                prompt="consent",
                access_type="offline",
                include_granted_scopes="true",
            )

            Path(self.token_file).write_text(credentials.to_json())
        else:
            credentials = Credentials.from_authorized_user_file(self.token_file)
            if credentials.expired:
                credentials.refresh(GoogleAuthRequest())
                Path(self.token_file).write_text(credentials.to_json())

        return credentials
