import matplotlib.pyplot as plt

# Your accuracy data
accuracy = [
    (0, 0.7989584803581238), (1, 0.8965784907341003), (2, 0.9359097480773926),
    (3, 0.9377826452255249), (4, 0.947238564491272), (5, 0.9254944920539856),
    (6, 0.937371551990509), (7, 0.9456854462623596), (8, 0.9508016705513),
    (9, 0.9526746273040771), (10, 0.9450916051864624)
]

# Split into x and y values
rounds, acc_values = zip(*accuracy)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(rounds, acc_values, marker='o', linestyle='-', color='purple')
plt.title("Accuracy Over Training Rounds")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.xticks(rounds)
plt.ylim(0, 1)
plt.grid(True)
plt.tight_layout()
plt.show()
