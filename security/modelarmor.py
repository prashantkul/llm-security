import requests
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from typing import List, Tuple, Dict
import json
from util.googleauth import GoogleAuthManager


class ModelArmorDetector:
    def __init__(
        self, project_id: str, location: str, template_name: str, sa_key_path: str, auth: GoogleAuthManager = None
    ):
        self.api_url = f"https://modelarmor.us-central1.rep.googleapis.com/v1alpha/projects/{project_id}/locations/{location}/templates/{template_name}:sanitizeUserPrompt"
        if auth is None:
            self.auth_manager = GoogleAuthManager(sa_key_path)
        self.auth_manager = auth
        self.credentials = self.auth_manager.get_token()
        print("Initialized ModelArmorDetector....")

    def detect_injection(self, user_prompt: str) -> Dict:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.credentials}",
        }
        data = {"user_prompt_data": {"text": user_prompt}}

        response = requests.post(self.api_url, headers=headers, json=data)
        response.raise_for_status()
        json_response = response.json()
        print("*"*100)
        print("Prompt: \n")
        print(user_prompt)
        print("*"*50)
        print("Raw ModelArmor Response: \n")
        print(json.dumps(json_response, indent=4))
        print("*"*100)
        print("\n")
        return json_response

    def parse_response(self, response_data: Dict) -> Dict[str, any]:
        result = {
            "injection_detected": False,
            "injection_confidence": None,
            "other_detections": [],
        }

        sanitization_result = response_data.get("sanitizationResult", {})
        filter_results = sanitization_result.get("filterResults", [])

        for filter_result in filter_results:
            if "piAndJailbreakFilterResult" in filter_result:
                pi_result = filter_result["piAndJailbreakFilterResult"]
                if pi_result["matchState"] == "MATCH_FOUND":
                    result["injection_detected"] = True
                    result["injection_confidence"] = pi_result["confidenceLevel"]
            elif "raiFilterResult" in filter_result:
                rai_result = filter_result["raiFilterResult"]
                for type_result in rai_result.get("raiFilterTypeResults", []):
                    if type_result["matchState"] == "MATCH_FOUND":
                        result["other_detections"].append(
                            {
                                "type": type_result["filterType"],
                                "confidence": type_result.get("confidenceLevel", "N/A"),
                            }
                        )

        return result

    def is_injection_attempt(self, user_prompt: str) -> Tuple[bool, Dict[str, any]]:
        response_data = self.detect_injection(user_prompt)
        parsed_result = self.parse_response(response_data)
        return parsed_result["injection_detected"], parsed_result
