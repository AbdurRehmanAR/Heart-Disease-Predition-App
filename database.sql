-- Heart Disease Application Database Schema
-- Database: heart_disease_db

CREATE DATABASE IF NOT EXISTS heart_disease_db;
USE heart_disease_db;

-- 1. Users Table
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    fullname VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    username VARCHAR(50) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Heart Predictions Table
CREATE TABLE IF NOT EXISTS heart_predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    patient_id VARCHAR(50) NOT NULL,
    patient_name VARCHAR(100) NOT NULL,
    age INT NOT NULL,
    gender INT NOT NULL, -- 0 = Female, 1 = Male
    cp INT NOT NULL,     -- Chest pain type (0-3)
    blood_pressure FLOAT NOT NULL, -- Resting blood pressure (trestbps)
    cholesterol FLOAT NOT NULL,    -- Serum cholesterol (chol)
    fbs INT NOT NULL,              -- Fasting blood sugar > 120 mg/dl
    restecg INT NOT NULL,          -- Resting ECG results
    heart_rate FLOAT NOT NULL,     -- Max heart rate (thalach)
    exang INT NOT NULL,            -- Exercise induced angina
    oldpeak FLOAT NOT NULL,        -- ST depression
    slope INT NOT NULL,            -- Slope of peak exercise ST segment
    ca INT NOT NULL,               -- Number of major vessels (0-3)
    thal INT NOT NULL,             -- Thalassemia (0-3)
    prediction_result INT NOT NULL, -- Mapped class (0, 1, 2, 3, 4)
    lr_prediction INT NOT NULL,     -- Logistic Regression (0=Low, 1=Medium, 2=High)
    rf_prediction INT NOT NULL,     -- Random Forest (0=Low, 1=Medium, 2=High)
    ann_prediction INT NOT NULL,    -- Neural Network (0=Low, 1=Medium, 2=High)
    risk_level VARCHAR(50) NOT NULL, -- Final prediction risk label
    risk_color VARCHAR(50) NOT NULL, -- Color gradient/code for CSS
    recommendation TEXT NOT NULL,    -- Doctor recommendation text
    graph_base64 LONGTEXT NOT NULL,  -- Base64 encoded comparison bar chart
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- 3. User Profile Table
CREATE TABLE IF NOT EXISTS user_profile (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT UNIQUE NOT NULL,
    height FLOAT DEFAULT NULL,
    weight FLOAT DEFAULT NULL,
    blood_group VARCHAR(5) DEFAULT NULL,
    heart_dimension VARCHAR(50) DEFAULT NULL,
    emergency_contact VARCHAR(50) DEFAULT NULL,
    blood_pressure VARCHAR(20) DEFAULT NULL, -- Static blood pressure reading for user profile
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Seed default user (admin / admin123) for authentication
-- Note: password hash for 'admin123' using scrypt/pbkdf2 will be generated and validated by Flask.
-- However, we seed a fallback plaintext user check or simple pbkdf2 hash.
-- The backend will automatically seed this admin user if not present, so we don't strictly need a hardcoded hash here,
-- but having the schema setup is primary.
