#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))  # 4 fruits x 3 people

people = ['Farrah', 'Fred', 'Felicia']
fruits = ['Apples', 'Bananas', 'Oranges', 'Peaches']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']  # colors per fruit

# Start with zeros for stacking
bottom = np.zeros(fruit.shape[1])

plt.figure(figsize=(8,6))

# Stack each fruit row
for i in range(fruit.shape[0]):
    plt.bar(people, fruit[i], bottom=bottom, color=colors[i], width=0.5, label=fruits[i])
    bottom += fruit[i]  # update bottom for next fruit

plt.ylabel('Quantity of Fruit')
plt.ylim(0, 80)
plt.yticks(np.arange(0, 81, 10))
plt.title('Number of Fruit per Person')
plt.legend()
plt.show()
