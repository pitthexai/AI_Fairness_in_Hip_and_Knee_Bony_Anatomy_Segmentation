from dataclasses import dataclass

@dataclass
class KneeAnatomy:
    classes = {
        0: "Background",
        1: "R Patella",
        2: "R Femur",
        3: "R Tibia",
        4: "R Fibula",
        5: "L Patella",
        6: "L Femur",
        7: "L Tibia",
        8: "L Fibula"
    }

    def get_class(self, class_name):
        self.classes[class_name]

    def get_num_classes(self):
        return len(self.classes)

@dataclass
class HipAnatomy:
    classes = {
        0: "Background",
        1: "R Acetabulum",
        2: "L Acetabulum",
        3: "R Ilium, Ischium, and Pubis",
        4: "L Ilium, Ischium, and Pubis",
        5: "R Femur",
        6: "L Femur"
    }

    def get_class(self, class_name):
        self.classes[class_name]

    def get_num_classes(self):
        return len(self.classes)
