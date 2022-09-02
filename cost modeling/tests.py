

class Test():
    def __init__(self, name: str, data_dictionary: dict):
        self.name = name
        self.cost = data_dictionary["cost"]
        self.sample_stability = data_dictionary["sample_stability"]
        self.blood = data_dictionary["blood"]
        self.result_waiting_time = data_dictionary["result_waiting_time"]
        self.result_vars = data_dictionary["result_vars"]
        self.other_tests_from_sample = data_dictionary["other_tests_from_sample"]

    def test_possible_from_sample(self, new_test: str):
        """Determines whether a certain test could be done with the same blood sample or not.

        Args:
            new_test (str): The test that could possibly share the same blood sample

        Returns:
            boolean: True for possible, False for not possible.
        """
        return new_test in self.other_tests_from_sample


    def __str__(self) -> str:
        return "Lab test: " + self.name

    def __repr__(self) -> str:
        return self.__str__()
        