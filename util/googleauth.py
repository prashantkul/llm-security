import google.auth
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account


class GoogleAuthManager:
    def __init__(self, scopes=None):
        self.scopes = scopes or []
        self.credentials = None
        self._initialize_credentials()

    def _initialize_credentials(self):
        try:
            self.credentials, _ = google.auth.default(scopes=self.scopes)
        except Exception as e:
            print(f"Error initializing default credentials: {e}")
            self.credentials = None

    def get_credentials(self):
        if (
            self.credentials
            and self.credentials.expired
            and self.credentials.refresh_token
        ):
            self.credentials.refresh(Request())
        return self.credentials

    def get_auth_headers(self):
        credentials = self.get_credentials()
        if not credentials:
            return {}
        return {"Authorization": f"Bearer {credentials.token}"}

    def set_service_account_key_file(self, key_file_path):
        try:
            self.credentials = service_account.Credentials.from_service_account_file(
                key_file_path, scopes=self.scopes
            )
        except Exception as e:
            print(f"Error setting service account credentials: {e}")
            self.credentials = None

    def is_authenticated(self):
        return self.credentials is not None and not self.credentials.expired

    def refresh_credentials(self):
        if self.credentials and hasattr(self.credentials, "refresh"):
            self.credentials.refresh(Request())
        else:
            self._initialize_credentials()
