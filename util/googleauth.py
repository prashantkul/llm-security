from google.oauth2 import service_account
from google.auth.transport.requests import Request
import time


class GoogleAuthManager:
    def __init__(self, key_file_path, scopes=None):
        self.key_file_path = key_file_path
        self.scopes = scopes or ["https://www.googleapis.com/auth/cloud-platform"]
        self.credentials = None
        self._initialize_credentials()

    def _initialize_credentials(self):
        try:
            self.credentials = service_account.Credentials.from_service_account_file(
                self.key_file_path, scopes=self.scopes
            )
            print("Service account credentials initialized successfully.")
        except Exception as e:
            print(f"Error initializing service account credentials: {e}")
            self.credentials = None

    def get_pure_credentials(self):
        return self.credentials

    def _refresh_token_if_expired(self):
        if not self.credentials:
            raise ValueError(
                "Credentials not initialized. Please check your service account key file."
            )

        if not self.credentials.valid:
            try:
                self.credentials.refresh(Request())
                print("Service account token refreshed successfully.")
            except Exception as e:
                print(f"Error refreshing token: {e}")
                self.credentials = None

    def get_token(self):
        self._refresh_token_if_expired()
        if self.credentials and self.credentials.valid:
            return self.credentials.token
        return None

    def get_auth_headers(self):
        token = self.get_token()
        if not token:
            raise ValueError(
                "Unable to obtain a valid token. Please check your service account credentials."
            )
        return {"Authorization": f"Bearer {token}"}

    def is_authenticated(self):
        return self.credentials is not None and self.credentials.valid

    def refresh_credentials(self):
        self._initialize_credentials()
        self._refresh_token_if_expired()


# # Example usage
# if __name__ == "__main__":
#     auth_manager = GoogleAuthManager("path/to/your/service-account-key.json")
#     headers = auth_manager.get_auth_headers()
#     print(f"Auth headers: {headers}")
