import matplotlib.pyplot as plt

# The slices will be ordered and plotted counter-clockwise.
labels = 'Positive', 'Neutral', 'Negative'
sizes = [15, 30, 20]
colors = ['lightgreen', 'cyan', 'magenta']

plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)

# Set aspect ratio to be equal so that pie is drawn as a circle.
plt.axis('equal')

plt.show()