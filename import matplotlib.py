import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset and rename the columns to match the expected format
points = pd.read_csv('linear_regression_data.csv')
points = points.rename(columns={'x': 'studytime', 'y': 'score'})

# Loss function for the current slope and intercept
def loss_function(m, b, points):
    total_error = 0.0
    for i in range(len(points)):
        x = points.iloc[i].studytime
        y = points.iloc[i].score
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(points))

# Gradient descent step
def gradient_descent(m_current, b_current, points, learning_rate):
    m_gradient = 0.0
    b_gradient = 0.0
    n = float(len(points))

    for i in range(len(points)):
        x = points.iloc[i].studytime
        y = points.iloc[i].score
        error = y - (m_current * x + b_current)
        m_gradient += -(2 / n) * x * error
        b_gradient += -(2 / n) * error

    new_m = m_current - learning_rate * m_gradient
    new_b = b_current - learning_rate * b_gradient
    return new_m, new_b

# Initialize hyperparameters
m = 0.0
b = 0.0
learning_rate = 0.001
num_iterations = 1000

# Run gradient descent
for i in range(num_iterations):
    m, b = gradient_descent(m, b, points, learning_rate)

# Print results
print(f"Final slope: {m:.4f}")
print(f"Final intercept: {b:.4f}")
print(f"Final loss: {loss_function(m, b, points):.4f}")

# Plot the raw data and the final regression line
x_values = points.studytime
y_values = points.score
x_line = [x_values.min(), x_values.max()]
y_line = [m * x + b for x in x_line]

plt.figure(figsize=(8, 6))
plt.scatter(x_values, y_values, alpha=0.7, label='Data points')
plt.plot(x_line, y_line, color='red', linewidth=2, label=f'Best fit: y = {m:.3f}x + {b:.3f}')
plt.xlabel('Study Time')
plt.ylabel('Score')
plt.title('Linear Regression with Gradient Descent Fit')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('optimal_regression_line.png', dpi=100, bbox_inches='tight')
plt.close()
print('Saved plot to optimal_regression_line.png')
