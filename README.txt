# Heart Disease Prediction Application – User Manual



## Step 1: Launch the Application



1. Open **Command Prompt (CMD)**.



   * Press **Windows + R**

   * Type **cmd**

   * Press **Enter**



2. Navigate to the project directory:



```bash

cd C:\Users\hp\Downloads\HearDisease_Application

```



3. Run the application:



```bash

python app.py

```



---



## Step 2: Open the Web Application



Open your browser and visit:



```text

http://localhost:5000

```



### Login Credentials



* Username: **admin**

* Password: **admin123**



> Note: The SQLite database is automatically created when the application runs for the first time.



---



## Step 3: Dashboard



After successful login, the Dashboard appears.



The dashboard provides:



* Total Patients

* Low Risk Patients

* Medium Risk Patients

* High Risk Patients

* Quick access to patient assessment functions



---



## Step 4: Patient Risk Assessment



1. Enter patient information in the assessment form.

2. Click **Analyze Patient Risk**.

3. The system processes the data using the trained machine learning model.

4. The prediction result is displayed.

5. Download the generated patient report if required.



---



## Step 5: Patient History



Navigate to **Patient History**.



Features:



* View all previous patient assessments.

* Search patient records.

* Check assessment dates.

* View predicted risk levels.

* Download assessment reports.

* Review complete patient details.



Example Patient History Table:



| Patient ID | Patient Name | Assessment Date     | Risk Level |

| ---------- | ------------ | ------------------- | ---------- |

| 1002       | Ali          | 2026-06-20 05:35 PM | Low Risk   |

| poo1       | nsns         | 2026-06-19 12:52 PM | Low Risk   |

| poo1       | nsns         | 2026-06-19 11:43 AM | Low Risk   |



---



## Step 6: My Profile



Open **My Profile** to view personal and health-related information.



### Profile Information



* Profile Photo

* User ID

* Email Address

* Account Status

* Membership Date



### Medical Report Section



Displays:



* Blood Pressure

* Heart Dimension

* Cholesterol Level

* Heart Rate

* Weight

* Height

* BMI

* Blood Group



### Health Risk Assessment



The profile page also shows:



* Risk Score

* Risk Category

* Health Recommendations

* Prediction Statistics



### Recent Prediction Activity



Users can view:



* Patient Name

* Patient ID

* Date and Time

* Blood Pressure

* Cholesterol

* Heart Rate

* Prediction Result



---



## Step 7: Logout



1. Click the **Logout** button.

2. The session ends securely.

3. The application redirects back to the **Login Page**.

4. Users can log in again using their credentials.



---



## Application Features Summary



* Secure Login System

* Heart Disease Risk Prediction

* Patient Data Management

* Prediction Report Download

* Patient History Tracking

* User Profile Management

* Health Metrics Monitoring

* SQLite Database Integration

* Machine Learning-Based Risk Assessment
