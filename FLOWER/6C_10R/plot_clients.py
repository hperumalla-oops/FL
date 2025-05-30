import matplotlib.pyplot as plt

# Eval accuracy data per client (copy-pasted from your logs)
client_1_acc = [
    0.9023638367652893, 0.9403905272483826, 0.9434738159179688,
    0.9530661106109619, 0.9606029391288757, 0.954093873500824,
    0.9558067917823792, 0.9609455466270447, 0.9664268493652344,
    0.9671120047569275
]

client_2_acc = [
    0.9061322212219238, 0.9431312084197998, 0.9479273557662964,
    0.9534087181091309, 0.9585474729537964, 0.9588900208473206,
    0.9588900208473206, 0.9626584649085999, 0.963001012802124,
    0.9681397676467896
]

client_3_acc = [
    0.9068173766136169, 0.9441589713096619, 0.9441589713096619,
    0.9534087181091309, 0.9588900208473206, 0.9612880945205688,
    0.9616307020187378, 0.964371383190155, 0.9650565385818481,
    0.9708804488182068
]

# Rounds
rounds = list(range(1, 11))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(rounds, client_1_acc, marker='o', label='Client 1', color='blue')
plt.plot(rounds, client_2_acc, marker='s', label='Client 2', color='green')
plt.plot(rounds, client_3_acc, marker='^', label='Client 3', color='red')

plt.title('Client Evaluation Accuracy over Rounds')
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.xticks(rounds)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
