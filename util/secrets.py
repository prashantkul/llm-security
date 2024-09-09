from google.cloud import secretmanager
from google.api_core import exceptions
import os
from typing import List, Dict, Optional
from util.googleauth import GoogleAuthManager

class GoogleSecretManagerAPIKey:
    SUPPORTED_SERVICES = ["google", "anthropic", "openai", "huggingface"]

    def __init__(self, project_id: str):
        sa_key_path = os.path.join(
            os.environ["SA_KEY_PATH"], os.environ["SA_KEY_NAME"]
        )
        self.project_id = project_id
        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        self.auth_manager = GoogleAuthManager(scopes=scopes)

        if sa_key_path:
            self.auth_manager.set_service_account_key_file(sa_key_path)

        self.client = secretmanager.SecretManagerServiceClient(
            credentials=self.auth_manager.get_credentials()
        )

    def _get_secret_name(self, service: str) -> str:
        """Generate the full secret name."""
        return f"projects/82097005135/secrets/{service}_api_key/versions/latest"

    def _verify_service(self, service: str) -> bool:
        """Verify if the service is supported."""
        if service not in self.SUPPORTED_SERVICES:
            print(
                f"Unsupported service: {service}. Supported services are: {', '.join(self.SUPPORTED_SERVICES)}"
            )
            return False
        return True

    def get_key(self, service: str) -> Optional[str]:
        """Retrieve a verified API key from Google Secret Manager."""
        if not self._verify_service(service):
            return None

        try:
            name = self._get_secret_name(service)
            response = self.client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
        except exceptions.NotFound:
            print(f"API key for {service} not found.")
            return None
        except Exception as e:
            print(f"Error retrieving API key for {service}: {str(e)}")
            return None

    def set_key(self, service: str, key: str):
        """Set or update a verified API key in Google Secret Manager."""
        if not self._verify_service(service):
            return

        try:
            parent = f"projects/{self.project_id}"
            secret_id = f"{service}_api_key"

            # Check if the secret already exists
            try:
                self.client.get_secret(
                    request={"name": f"{parent}/secrets/{secret_id}"}
                )
            except exceptions.NotFound:
                # Create a new secret
                self.client.create_secret(
                    request={
                        "parent": parent,
                        "secret_id": secret_id,
                        "secret": {"replication": {"automatic": {}}},
                    }
                )

            # Add a new version of the secret
            self.client.add_secret_version(
                request={
                    "parent": f"{parent}/secrets/{secret_id}",
                    "payload": {"data": key.encode("UTF-8")},
                }
            )
            print(f"Successfully set/updated API key for {service}")
        except Exception as e:
            print(f"Error setting API key for {service}: {str(e)}")

    def remove_key(self, service: str):
        """Remove a verified API key from Google Secret Manager."""
        if not self._verify_service(service):
            return

        try:
            name = f"projects/{self.project_id}/secrets/{service}_api_key"
            self.client.delete_secret(request={"name": name})
            print(f"Successfully removed API key for {service}")
        except exceptions.NotFound:
            print(f"API key for {service} not found.")
        except Exception as e:
            print(f"Error removing API key for {service}: {str(e)}")

    def get_all_keys(self) -> Dict[str, Optional[str]]:
        """Retrieve all supported API keys."""
        return {service: self.get_key(service) for service in self.SUPPORTED_SERVICES}

    def list_secrets(self):
        parent = f"projects/{self.project_id}"
        secrets = self.client.list_secrets(request={"parent": parent})
        return [secret.name for secret in secrets]


# # Example usage:
# if __name__ == "__main__":
#     # Make sure to set your Google Cloud project ID
#     project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
#     if not project_id:
#         raise ValueError("Please set the GOOGLE_CLOUD_PROJECT environment variable")

#     key_manager = VerifiedAPIKeyManager(project_id)

#     # Set API keys (in a real scenario, you'd get these securely)
#     key_manager.set_key("google", "google_api_key_123")
#     key_manager.set_key("anthropic", "anthropic_api_key_456")
#     key_manager.set_key("openai", "openai_api_key_789")

#     # Retrieve individual keys
#     print(f"Google API Key: {key_manager.get_key('google')}")
#     print(f"Anthropic API Key: {key_manager.get_key('anthropic')}")
#     print(f"OpenAI API Key: {key_manager.get_key('openai')}")

#     # Retrieve all keys
#     all_keys = key_manager.get_all_keys()
#     print("All API Keys:")
#     for service, key in all_keys.items():
#         print(f"{service.capitalize()}: {'*****' if key else 'Not set'}")

#     # Try to set an unsupported key
#     key_manager.set_key("unsupported", "some_key")

#     # Remove a key
#     key_manager.remove_key("openai")

#     # Try to access a removed key
#     print(f"Trying to access removed OpenAI key: {key_manager.get_key('openai')}")

#     # Final state of all keys
#     print("\nFinal state of API Keys:")
#     all_keys = key_manager.get_all_keys()
#     for service, key in all_keys.items():
#         print(f"{service.capitalize()}: {'*****' if key else 'Not set'}")

# simple test to check IAM permissions
# if __name__ == "__main__":
#     # Make sure to set your Google Cloud project ID
#     project_id = "pk-arg-prj1"

#     key_manager = GoogleSecretManagerAPIKey(project_id=project_id)
#     print(key_manager.list_secrets())
#     print(key_manager.get_key(service="google"))
