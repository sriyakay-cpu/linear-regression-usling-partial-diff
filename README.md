
 Linear Regression

 
Linear regression is a method to model the relationship between two variables by fitting a straight line to observed data.
Real-world examples:

Predicting the stress on a beam given its load
Estimating current from voltage in a resistor
Forecasting temperature based on altitude

2. The Mathematical Foundation
The Equation of a Line
The core equation is simply the equation of a straight line:
ŷ = mx + c
Or in standard ML notation:
ŷ = β₀ + β₁x


The Cost Function (Mean Squared Error)
We want to find the line that minimizes the total error. We use the Mean Squared Error (MSE):
        1   n
MSE =  ─── Σ (yᵢ - ŷᵢ)²
        n  i=1

Squaring the errors ensures negatives don't cancel positives, and penalizes large errors more heavily.


The Normal Equations (Exact Solution)
For simple linear regression, we can solve directly for β₀ and β₁:
     Σ(xᵢ - x̄)(yᵢ - ȳ)

        Σ(xᵢ - x̄)²
β₀ = ȳ - β₁ · x̄



3. How the Algorithm Works
Step 1: Collect data points (x₁,y₁), (x₂,y₂), ..., (xₙ,yₙ)
Step 2: Calculate means x̄ and ȳ
Step 3: Compute slope β₁ using the Normal Equation
Step 4: Compute intercept β₀
Step 5: Use ŷ = β₀ + β₁x to make predictions

4. Python Implementation
Method A — From Scratch (to understand the math)
pythonimport numpy as np
import matplotlib.pyplot as plt

# Sample data: Load (kN) vs Deflection (mm) of a beam
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])   # Load in kN
y = np.array([2.1, 3.9, 6.2, 7.8, 10.1, 12.0, 14.2, 15.8])  # Deflection in mm

# Step 1: Calculate means
x_mean = np.mean(x)
y_mean = np.mean(y)

# Step 2: Calculate slope (β₁)
numerator   = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sum((x - x_mean) ** 2)
beta_1 = numerator / denominator

# Step 3: Calculate intercept (β₀)
beta_0 = y_mean - beta_1 * x_mean

print(f"Slope     β₁ = {beta_1:.4f}")
print(f"Intercept β₀ = {beta_0:.4f}")
print(f"Equation: ŷ = {beta_0:.4f} + {beta_1:.4f}x")

# Step 4: Make predictions
y_pred = beta_0 + beta_1 * x

# Step 5: Plot
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='blue', label='Actual Data', zorder=5)
plt.plot(x, y_pred, color='red', label=f'Fit: ŷ = {beta_0:.2f} + {beta_1:.2f}x')
plt.xlabel('Load (kN)')
plt.ylabel('Deflection (mm)')
plt.title('Linear Regression — Beam Load vs Deflection')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
