// This code assumes you're using ReactPy

import { Component, render, createElement } from "https://unpkg.com/reactpy@latest/web.js";
import { useState } from "https://unpkg.com/reactpy@latest/hooks.js";
import { useEffect } from "https://unpkg.com/reactpy@latest/effects.js";

class Chatbot extends Component {
    constructor(props) {
        super(props);
        this.state = {
            messages: [],
            inputs: {
                symptoms: "",
                severity: "",
                pre_existing_condition: "",
                temporal_info: "",
                past_treatments: "",
                surgeries: "",
                medications: "",
                treatments_administered: ""
            },
            totalBilling: 0
        };
    }

    handleInputChange(field, event) {
        const { inputs } = this.state;
        inputs[field] = event.target.value;
        this.setState({ inputs });
    }

    handleSend() {
        const { inputs } = this.state;
        const inputText = Object.values(inputs).join(" ");
        
        if (Object.values(inputs).some(value => value.trim())) {
            this.setState({
                messages: [...this.state.messages, { sender: "user", text: inputText }],
                inputs: Object.fromEntries(Object.keys(inputs).map(key => [key, ""]))
            });

            fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(inputs)
            })
            .then(response => response.json())
            .then(data => {
                if (data.icd_code) {
                    const { icd_code, billing } = data;
                    this.setState({
                        messages: [
                            ...this.state.messages,
                            { sender: "bot", text: `Predicted ICD code: ${icd_code}` },
                            { sender: "bot", text: `Billing Information: $${billing}` }
                        ],
                        totalBilling: this.state.totalBilling + billing
                    });
                } else {
                    this.setState({
                        messages: [...this.state.messages, { sender: "bot", text: "Sorry, I couldn't predict the ICD code for your symptoms." }]
                    });
                }
            })
            .catch(error => {
                console.error("Error:", error);
            });
        }
    }

    renderMessages() {
        return this.state.messages.map((message, index) => (
            <div key={index} className={`message ${message.sender}`}>
                {message.text}
            </div>
        ));
    }

    renderInput(label, field) {
        return (
            <div className="input-group">
                <label>{label}</label>
                <input
                    type="text"
                    value={this.state.inputs[field]}
                    onChange={event => this.handleInputChange(field, event)}
                />
            </div>
        );
    }

    render() {
        return (
            <div className="chatbot-container">
                <div className="messages">{this.renderMessages()}</div>
                <div className="input-area">
                    {this.renderInput("Symptoms", "symptoms")}
                    {this.renderInput("Severity", "severity")}
                    {this.renderInput("Pre-existing Condition", "pre_existing_condition")}
                    {this.renderInput("Temporal Info", "temporal_info")}
                    {this.renderInput("Past Treatments", "past_treatments")}
                    {this.renderInput("Surgeries", "surgeries")}
                    {this.renderInput("Medications", "medications")}
                    {this.renderInput("Treatments Administered", "treatments_administered")}
                    <button onClick={() => this.handleSend()}>Send</button>
                </div>
                <div className="total-billing">Total Billing: ${this.state.totalBilling}</div>
            </div>
        );
    }
}

render(<Chatbot />, document.getElementById("root"));
