class ApiResponse:

    def __init__(self, category="", predicted="", message=""):
        self.category = category
        self.predicted = predicted
        self.message = message

    def json(self):
        return {"category": self.category, "predicted": self.predicted, "message": self.message}
