# -----------------------------
# Math Wizard AI — DL Edition
# -----------------------------

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean, median, mode, StatisticsError
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

st.set_page_config(page_title="Math Wizard AI", layout="wide")
st.title("""🧙‍♂️ Math Wizard AI — Predictive & Calculative Engine ('Combination of Machine and Deep Learning')""")
st.markdown("**Made by Ali Kahoot**")

# -----------------------------
# Sidebar: Choose Feature
# -----------------------------
option = st.sidebar.selectbox(
    "Choose a feature",
    ["Linear Equation", "Quadratic Equation", "Statistics (Mean, Median, Mode)",
     "Volume & Area Calculator", "Algebra Expression Solver"]
)

# -----------------------------
# Helper: Train DL Model for Linear Equations
# -----------------------------
@st.cache_data
def train_linear_model():
    samples = 10000
    a = np.random.randint(1, 10, size=samples)
    b = np.random.randint(-10, 10, size=samples)
    c = np.random.randint(-20, 20, size=samples)
    X = np.stack([a, b, c], axis=1)
    y = (c - b) / a
    model = Sequential([
        Dense(64, activation='relu', input_shape=(3,)),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.001), loss='mse')
    model.fit(X, y, epochs=20, batch_size=32, validation_split=0.1, verbose=0)
    return model

@st.cache_data
def train_quadratic_model():
    samples = 10000
    a = np.random.randint(1, 10, size=samples)
    b = np.random.randint(-20, 20, size=samples)
    c = np.random.randint(-20, 20, size=samples)
    X = np.stack([a, b, c], axis=1)
    D = b**2 - 4*a*c
    y = np.where(D >= 0, (-b + np.sqrt(D)) / (2*a), 0)
    model = Sequential([
        Dense(128, activation='relu', input_shape=(3,)),
        Dense(128, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.001), loss='mse')
    model.fit(X, y, epochs=30, batch_size=32, validation_split=0.1, verbose=0)
    return model

# -----------------------------
# Linear Equation Solver
# -----------------------------
if option == "Linear Equation":
    st.header("Solve Linear Equation ax + b = c")
    a_val = st.number_input("a", value=1.0)
    b_val = st.number_input("b", value=0.0)
    c_val = st.number_input("c", value=1.0)
    if st.button("Predict Solution"):
        linear_model = train_linear_model()
        pred = linear_model.predict(np.array([[a_val, b_val, c_val]]))
        st.success(f"✅ Predicted x ≈ {pred[0][0]:.3f}")

# -----------------------------
# Quadratic Equation Solver
# -----------------------------
elif option == "Quadratic Equation":
    st.header("Solve Quadratic Equation ax² + bx + c = 0")
    a_val = st.number_input("a", value=1.0)
    b_val = st.number_input("b", value=0.0)
    c_val = st.number_input("c", value=1.0)
    if st.button("Predict x1"):
        quadratic_model = train_quadratic_model()
        pred = quadratic_model.predict(np.array([[a_val, b_val, c_val]]))
        st.success(f"✅ Predicted x1 ≈ {pred[0][0]:.3f} (DL Approximation)")

# -----------------------------
# Statistics
# -----------------------------
elif option == "Statistics (Mean, Median, Mode)":
    st.header("Calculate Mean, Median, Mode")
    numbers_input = st.text_area("Enter numbers separated by commas (e.g., 1,2,3,4)")
    if st.button("Calculate Statistics"):
        try:
            numbers = [float(n.strip()) for n in numbers_input.split(",")]
            avg = mean(numbers)
            med = median(numbers)
            try:
                mod = mode(numbers)
            except StatisticsError:
                mod = "No unique mode"
            st.success(f"✅ Mean: {avg}, Median: {med}, Mode: {mod}")
            
            plt.figure(figsize=(6,3))
            plt.plot(range(1, len(numbers)+1), numbers, marker='o', linestyle='-')
            plt.xlabel("Index")
            plt.ylabel("Value")
            plt.title("Line Graph of Numbers")
            plt.grid(True)
            st.pyplot(plt)
        except:
            st.error("Invalid input. Please enter numbers separated by commas.")

# -----------------------------
# Volume & Area Calculator
# -----------------------------
elif option == "Volume & Area Calculator":
    st.header("Calculate Volume & Area of Shapes")
    shape = st.selectbox("Select Shape", ["Cube", "Cuboid", "Sphere", "Cylinder", "Cone", "Rectangle", "Square"])
    
    if st.button("Calculate"):
        if shape == "Cube":
            side = st.number_input("Side Length", value=1.0)
            volume = side**3
            area = 6*side**2
        elif shape == "Cuboid":
            l = st.number_input("Length", value=1.0)
            w = st.number_input("Width", value=1.0)
            h = st.number_input("Height", value=1.0)
            volume = l*w*h
            area = 2*(l*w + l*h + w*h)
        elif shape == "Sphere":
            r = st.number_input("Radius", value=1.0)
            volume = (4/3)*np.pi*r**3
            area = 4*np.pi*r**2
        elif shape == "Cylinder":
            r = st.number_input("Radius", value=1.0)
            h = st.number_input("Height", value=1.0)
            volume = np.pi*r**2*h
            area = 2*np.pi*r*(r+h)
        elif shape == "Cone":
            r = st.number_input("Radius", value=1.0)
            h = st.number_input("Height", value=1.0)
            volume = (1/3)*np.pi*r**2*h
            area = np.pi*r*(r + np.sqrt(h**2 + r**2))
        elif shape == "Rectangle":
            l = st.number_input("Length", value=1.0)
            w = st.number_input("Width", value=1.0)
            area = l*w
            volume = "N/A"
        elif shape == "Square":
            s = st.number_input("Side", value=1.0)
            area = s**2
            volume = "N/A"
        
        st.success(f"✅ Area: {area}, Volume: {volume}")

# -----------------------------
# Algebra Expression Solver
# -----------------------------
elif option == "Algebra Expression Solver":
    st.header("Solve Simple Algebraic Expressions")
    expr = st.text_input("Enter expression using x (e.g., 2*x + 3 = 7)")
    x_value = st.number_input("Value of x (optional for evaluation)", value=0.0)
    if st.button("Evaluate Expression"):
        try:
            result = eval(expr.replace("x", f"({x_value})"))
            st.success(f"✅ Result: {result}")
        except:
            st.error("Invalid expression. Use x properly.")






