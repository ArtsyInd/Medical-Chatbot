import reactpy as rp
import requests

class Chatbot(rp.Component):
    def __init__(self):
        self.state = {
            "messages": [],
            "inputs": {
                "symptoms": "",
                "severity": "",
                "pre_existing_condition": "",
                "temporal_info": "",
                "past_treatments": "",
                "surgeries": "",
                "medications": "",
                "treatments_administered": ""
            },
            "totalBilling": 0
        }

    def handleInputChange(self, field, event):
        inputs = self.state["inputs"]
        inputs[field] = event.target.value
        self.set_state({"inputs": inputs})

    def handleSend(self, event):
        inputs = self.state["inputs"]
        input_text = " ".join(inputs.values())
        
        if any(value.strip() for value in inputs.values()):
            self.set_state({
                "messages": self.state["messages"] + [{"sender": "user", "text": input_text}],
                "inputs": {key: "" for key in inputs}
            })

            response = requests.post("/predict", json=inputs)
            if response.status_code == 200:
                data = response.json()
                icd_code = data.get("icd_code")
                billing = data.get("billing")

                if icd_code:
                    self.set_state({
                        "messages": self.state["messages"] + [
                            {"sender": "bot", "text": f"Predicted ICD code: {icd_code}"},
                            {"sender": "bot", "text": f"Billing Information: ${billing}"}
                        ],
                        "totalBilling": self.state["totalBilling"] + billing
                    })
                else:
                    self.set_state({
                        "messages": self.state["messages"] + [
                            {"sender": "bot", "text": "Sorry, I couldn't predict the ICD code for your symptoms."}
                        ]
                    })

    def renderMessages(self):
        return [
            rp.div(
                {"key": index, "className": f"message {message['sender']}"},
                message["text"]
            ) for index, message in enumerate(self.state["messages"])
        ]

    def renderInput(self, label, field):
        return rp.div(
            {"className": "input-group"},
            rp.label({}, label),
            rp.input(
                {
                    "type": "text",
                    "value": self.state["inputs"][field],
                    "onChange": lambda event: self.handleInputChange(field, event)
                }
            )
        )

    def render(self):
        return rp.div(
            {"className": "chatbot-container"},
            rp.div({"className": "messages"}, self.renderMessages()),
            rp.div(
                {"className": "input-area"},
                self.renderInput("Symptoms", "symptoms"),
                self.renderInput("Severity", "severity"),
                self.renderInput("Pre-existing Condition", "pre_existing_condition"),
                self.renderInput("Temporal Info", "temporal_info"),
                self.renderInput("Past Treatments", "past_treatments"),
                self.renderInput("Surgeries", "surgeries"),
                self.renderInput("Medications", "medications"),
                self.renderInput("Treatments Administered", "treatments_administered"),
                rp.button({"onClick": self.handleSend}, "Send")
            ),
            rp.div({"className": "total-billing"}, f"Total Billing: ${self.state['totalBilling']}")
        )

rp.render(rp.create_element(Chatbot), "#root")
