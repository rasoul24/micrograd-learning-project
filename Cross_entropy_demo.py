from sklearn.datasets import load_digits
from BackPropagation.back_propagation_engine import Value
from BackPropagation.neural_network import MLP


import matplotlib.pyplot as plt
import numpy as np

# 1. Charger les images (1797 images au total, format 8x8)
digits = load_digits()
X = digits.data  # Forme : (1797, 64)
y = digits.target  # Forme : (1797,) contenant des chiffres de 0 à 9


# 3. Normaliser les pixels entre 0 et 1 (très important pour les réseaux de neurones)
X_normalized = X / 16.0

# Maintenant, vous pouvez injecter X_normalized et y_binary dans votre boucle !
# Par exemple, pour un petit lot de 10 images :
Xb = X_normalized[:100]
yb = y[:100]

# Vos inputs compatibles avec votre moteur de Value :
inputs = [[Value(x) for x in xrow] for xrow in Xb]

image_vector = Xb[5]

image = image_vector.reshape(8,8)

plt.imshow(image)
plt.title(f"Label / Chiffre : {yb[5]}")
plt.show()




model = MLP(64, [16, 10])
print(model)
print("number of parameters", len(model.parameters()))

# outputs = list(map(model, inputs))
# exps = [o.exp() for o in outputs]
# sum_exps = sum(exps)

# probs = [e / sum_exps for e in exps]

# correct_prob = probs[yb]

# loss = -correct_prob.log()



# Préparation du lot
inputs_batch = [[Value(x) for x in xrow] for xrow in Xb]

for k in range(1000):

    # Forward : un vecteur de 10 scores par image
    outputs = [model(xrow) for xrow in inputs_batch]

    losses = []

    # Softmax et cross-entropy pour chaque image
    for logits, target in zip(outputs, yb):

        # Stabilisation numérique du softmax
        max_logit = max(o.data for o in logits)

        exps = [(o - max_logit).exp() for o in logits]
        sum_exps = sum(exps)

        probs = [e / sum_exps for e in exps]

        # Probabilité de la classe correcte
        correct_prob = probs[int(target)]

        # Cross-entropy
        losses.append(-correct_prob.log())

    # Perte moyenne du batch
    total_loss = sum(losses) / len(losses)

    # Backward
    model.zero_grad()
    total_loss.backward()

    # Descente de gradient
    #learning_rate = 1.0 - 0.9 * k / 100
    learning_rate = 0.01

    for p in model.parameters():
        p.data -= learning_rate * p.grad

    if k%100==0:
        print(f"step {k} loss {total_loss.data}")
