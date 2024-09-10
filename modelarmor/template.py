import json
import requests
from dotenv import load_dotenv
import os
from util.googleauth import GoogleAuthManager


class ModelArmorTemplateManager:
    def __init__(self, project_id, location, sa_key_path):
        self.project_id = project_id
        self.location = location
        self.template_id = None
        self.base_url = f"https://modelarmor.{location}.rep.googleapis.com/v1alpha/projects/{project_id}/locations/{location}"
        self.filter_config = self._get_default_filter_config()
        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        self.auth_manager = GoogleAuthManager(key_file_path=sa_key_path, scopes=scopes)

    def set_template_id(self, template_id):
        self.template_id = template_id

    def _get_default_filter_config(self):
        return {
            "filter_config": {
                "rai_settings": {
                    "rai_filters": [
                        {
                            "filter_type": "HATE_SPEECH",
                            "confidence_level": "LOW_AND_ABOVE",
                        },
                        {"filter_type": "TOXIC", "confidence_level": "LOW_AND_ABOVE"},
                        {"filter_type": "VIOLENT", "confidence_level": "LOW_AND_ABOVE"},
                        {
                            "filter_type": "SEXUALLY_EXPLICIT",
                            "confidence_level": "LOW_AND_ABOVE",
                        },
                        {
                            "filter_type": "HARASSMENT",
                            "confidence_level": "LOW_AND_ABOVE",
                        },
                        {
                            "filter_type": "PROFANITY",
                            "confidence_level": "LOW_AND_ABOVE",
                        },
                        {
                            "filter_type": "DEATH_HARM_TRAGEDY",
                            "confidence_level": "LOW_AND_ABOVE",
                        },
                        {
                            "filter_type": "FIREARMS_WEAPONS",
                            "confidence_level": "LOW_AND_ABOVE",
                        },
                        {
                            "filter_type": "PUBLIC_SAFETY",
                            "confidence_level": "LOW_AND_ABOVE",
                        },
                        {"filter_type": "HEALTH", "confidence_level": "LOW_AND_ABOVE"},
                        {
                            "filter_type": "RELIGIOUS_BELIEF",
                            "confidence_level": "LOW_AND_ABOVE",
                        },
                        {
                            "filter_type": "ILLICIT_DRUGS",
                            "confidence_level": "LOW_AND_ABOVE",
                        },
                        {
                            "filter_type": "WAR_CONFLICT",
                            "confidence_level": "LOW_AND_ABOVE",
                        },
                        {
                            "filter_type": "POLITICS",
                            "confidence_level": "LOW_AND_ABOVE",
                        },
                        {"filter_type": "FINANCE", "confidence_level": "LOW_AND_ABOVE"},
                        {"filter_type": "LEGAL", "confidence_level": "LOW_AND_ABOVE"},
                    ]
                },
                "piAndJailbreakFilterSettings": {"filterEnforcement": "ENABLED"},
                "maliciousUriFilterSettings": {"filterEnforcement": "ENABLED"},
                "sdpSettings": {"basicConfig": {"filterEnforcement": "ENABLED"}},
            }
        }

    def get_access_token(self):
        return self.auth_manager.get_token()

    def _make_request(self, method, url, data=None):
        access_token = self.get_access_token()
        if not access_token:
            return False, "Failed to obtain access token"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}",
        }

        try:
            if method == "GET":
                response = requests.get(url, headers=headers)
            elif method == "POST":
                response = requests.post(url, headers=headers, data=json.dumps(data))
            elif method == "PATCH":
                response = requests.patch(url, headers=headers, data=json.dumps(data))
            elif method == "DELETE":
                response = requests.delete(url, headers=headers)
            else:
                return False, f"Unsupported HTTP method: {method}"

            response.raise_for_status()
            return True, response.json()
        except requests.RequestException as e:
            return False, f"Request failed: {str(e)}"

    def create(self):
        url = f"{self.base_url}/templates?template_id={self.template_id}"
        success, result = self._make_request("POST", url, self.filter_config)
        return success, "Template created successfully" if success else result

    def view(self):
        if not self.template_id:
            return False, "Template ID is not set"
        url = f"{self.base_url}/templates/{self.template_id}"
        success, result = self._make_request("GET", url)
        if success:
            # Pretty print the JSON result
            pretty_result = json.dumps(result, indent=2, sort_keys=True)
            return success, pretty_result
        else:
            return success, "Failed to view template"

    def update(self):
        if not self.template_id:
            return False, "Template ID is not set"
        url = f"{self.base_url}/templates/{self.template_id}?update_mask=filter_config"
        success, result = self._make_request("PATCH", url, self.filter_config)
        return success, "Template updated successfully" if success else result

    def list(self):
        url = f"{self.base_url}/templates"
        success, result = self._make_request("GET", url)
        return success, result if success else "Failed to list templates"

    def delete(self):
        if not self.template_id:
            return False, "Template ID is not set"
        url = f"{self.base_url}/templates/{self.template_id}"
        success, result = self._make_request("DELETE", url)
        return success, "Template deleted successfully" if success else result

    def update_rai_filter(self, filter_type, confidence_level):
        for filter in self.filter_config["filter_config"]["rai_settings"][
            "rai_filters"
        ]:
            if filter["filter_type"] == filter_type:
                filter["confidence_level"] = confidence_level
                return True
        return False

    def set_pi_and_jailbreak_filter(self, enforcement):
        self.filter_config["filter_config"]["piAndJailbreakFilterSettings"][
            "filterEnforcement"
        ] = enforcement

    def set_malicious_uri_filter(self, enforcement):
        self.filter_config["filter_config"]["maliciousUriFilterSettings"][
            "filterEnforcement"
        ] = enforcement

    def set_sdp_settings(self, enforcement):
        self.filter_config["filter_config"]["sdpSettings"]["basicConfig"][
            "filterEnforcement"
        ] = enforcement

    def get_current_config(self):
        return self.filter_config

    def set_full_config(self, new_config):
        self.filter_config = new_config

    def print_available_operations(self):
        print("Available operations:")
        print("1. Create template")
        print("2. View template")
        print("3. Update template")
        print("4. List templates")
        print("5. Delete template")


# Usage example
if __name__ == "__main__":
    print("Loading .env file")
    _ = load_dotenv("local.env")

    project_id = os.environ["GOOGLE_CLOUD_PROJECT"]
    if not project_id:
        raise ValueError("Please set the GOOGLE_CLOUD_PROJECT environment variable")
    ma_region = os.environ["REGION"]
    sa_key_path = os.path.join(os.environ["SA_KEY_PATH"], os.environ["SA_KEY_NAME"])

    manager = ModelArmorTemplateManager(
        project_id, ma_region, sa_key_path=sa_key_path
    )

    # Print available operations and ask for user input
    manager.print_available_operations()
    # Ask for template name
    operation = input("Enter the number of the operation you want to perform: ")

    # Perform the selected operation
    if operation == "1":
        template_name = input("Enter the template name to create: ")
        manager.set_template_id(template_name)
        print("\nCreating template:")
        success, result = manager.create()
    elif operation == "2":
        template_name = input("Enter the template name to view: ")
        manager.set_template_id(template_name)
        print("\nViewing template:")
        success, result = manager.view()
    elif operation == "3":
        template_name = input("Enter the template name to update: ")
        manager.set_template_id(template_name)
        print("\nUpdating template:")
        
        while True:
            print("\nWhat would you like to update?")
            print("1. RAI filter")
            print("2. Prompt Injection and Jailbreak filter")
            print("3. Malicious URI filter")
            print("4. SDP settings")
            print("5. Finish updates and submit")
            
            update_choice = input("Enter your choice (1-5): ")
            
            if update_choice == "1":
                filter_types = [
                    "HATE_SPEECH", "TOXIC", "VIOLENT", "SEXUALLY_EXPLICIT",
                    "HARASSMENT", "PROFANITY", "DEATH_HARM_TRAGEDY",
                    "FIREARMS_WEAPONS", "PUBLIC_SAFETY", "HEALTH",
                    "RELIGIOUS_BELIEF", "ILLICIT_DRUGS", "WAR_CONFLICT",
                    "POLITICS", "FINANCE", "LEGAL"
                ]
                print("\nAvailable RAI filters:")
                for i, filter_type in enumerate(filter_types, 1):
                    print(f"{i}. {filter_type}")
                
                filter_choice = int(input("Enter the number of the filter to update: ")) - 1
                if 0 <= filter_choice < len(filter_types):
                    confidence_levels = ["LOW_AND_ABOVE", "MEDIUM_AND_ABOVE", "HIGH"]
                    print("\nAvailable confidence levels:")
                    for i, level in enumerate(confidence_levels, 1):
                        print(f"{i}. {level}")
                    
                    level_choice = int(input("Enter the number of the confidence level: ")) - 1
                    if 0 <= level_choice < len(confidence_levels):
                        manager.update_rai_filter(filter_types[filter_choice], confidence_levels[level_choice])
                        print(f"Updated {filter_types[filter_choice]} to {confidence_levels[level_choice]}")
                    else:
                        print("Invalid confidence level choice.")
                else:
                    print("Invalid filter choice.")
            
            elif update_choice == "2":
                enforcement = input("Enter enforcement for Prompt Injection and Jailbreak filter (ENABLED/DISABLED): ").upper()
                if enforcement in ["ENABLED", "DISABLED"]:
                    manager.set_pi_and_jailbreak_filter(enforcement)
                    print(f"Updated Prompt Injection and Jailbreak filter to {enforcement}")
                else:
                    print("Invalid enforcement value. Please enter ENABLED or DISABLED.")
            
            elif update_choice == "3":
                enforcement = input("Enter enforcement for Malicious URI filter (ENABLED/DISABLED): ").upper()
                if enforcement in ["ENABLED", "DISABLED"]:
                    manager.set_malicious_uri_filter(enforcement)
                    print(f"Updated Malicious URI filter to {enforcement}")
                else:
                    print("Invalid enforcement value. Please enter ENABLED or DISABLED.")
            
            elif update_choice == "4":
                enforcement = input("Enter enforcement for SDP settings (ENABLED/DISABLED): ").upper()
                if enforcement in ["ENABLED", "DISABLED"]:
                    manager.set_sdp_settings(enforcement)
                    print(f"Updated SDP settings to {enforcement}")
                else:
                    print("Invalid enforcement value. Please enter ENABLED or DISABLED.")
            
            elif update_choice == "5":
                break
            
            else:
                print("Invalid choice. Please enter a number between 1 and 5.")
        
        print("\nSubmitting updates...")
        success, result = manager.update()
    elif operation == "4":
        print("\nListing templates:")
        success, result = manager.list()
    elif operation == "5":
        template_name = input("Enter the template name to delete: ")
        manager.set_template_id(template_name)
        confirm = input("Are you sure you want to delete this template? (y/n): ")
        if confirm.lower() != "y":
            print("Template deletion canceled.")
            exit(0)
        print("\nDeleting template:")
        success, result = manager.delete()
    else:
        print("Invalid operation selected.")
        exit(1)

    print(f"Operation result: {'Success' if success else 'Failed'}")
    print(result)

    # Additional interaction for viewing current config
    if success and operation in ["1", "2", "3"]:
        view_config = input("Do you want to view the current configuration? (y/n): ")
        if view_config.lower() == "y":
            print(json.dumps(manager.get_current_config(), indent=2))
