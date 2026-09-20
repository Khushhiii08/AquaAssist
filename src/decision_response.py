def get_decision_response(decision):

    if decision == "CLARIFY":
        return {
            "type": "clarify",
            "message": (
                "I need more information before giving a recommendation. "
                "Please provide the relevant water-quality measurements "
                "if available."
            )
        }

    if decision == "ABSTAIN":
        return {
            "type": "abstain",
            "message": (
                "Warning: The available evidence is not sufficient "
                "to provide a safe recommendation."
            )
        }

    return None