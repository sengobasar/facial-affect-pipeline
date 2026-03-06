from models.mental_state_classifier import MentalStateClassifier
import numpy as np

classifier = MentalStateClassifier()

sample = np.array([
    -0.2, 0.6, 0.5,
    0.3, 0.2,
    -0.1, 0.5, 0.4,
    120, 0.02, 0.03, 500
])

p = classifier.predict_probability(sample)
label = classifier.predict_label(sample)

print("Probability:", p)
print("Label:", label)