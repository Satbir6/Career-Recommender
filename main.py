import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = "career_recommender.csv"
career_data = pd.read_csv(file_path)

# Handle potential leading/trailing spaces in column names
career_data.columns = career_data.columns.str.strip()

# Identify the 'First Job Title' column
job_title_column = None
for col in career_data.columns:
    if "Job title" in col:
        job_title_column = col
        break

# Initialize LabelEncoders
course_encoder = LabelEncoder()
specialization_encoder = LabelEncoder()
job_title_encoder = LabelEncoder()

# Encode categorical variables for UG Course and UG Specialization
career_data["UG Course"] = course_encoder.fit_transform(
    career_data["What was your course in UG?"]
)
career_data["UG Specialization"] = specialization_encoder.fit_transform(
    career_data["What is your UG specialization? Major Subject (Eg; Mathematics)"]
)

# Encode the target variable 'First Job Title'
career_data["First Job Title"] = job_title_encoder.fit_transform(
    career_data[job_title_column]
)

# Multi-label binarization for Interests and Skills
mlb_interests = MultiLabelBinarizer()
mlb_skills = MultiLabelBinarizer()

career_data["Interests"] = (
    career_data["What are your interests?"].fillna("").apply(lambda x: x.split(";"))
)
career_data["Skills"] = (
    career_data["What are your skills ? (Select multiple if necessary)"]
    .fillna("")
    .apply(lambda x: x.split(";"))
)

# Binarize Interests and Skills
interests_encoded = mlb_interests.fit_transform(career_data["Interests"])
skills_encoded = mlb_skills.fit_transform(career_data["Skills"])

# Add the encoded multi-label features to the main DataFrame
career_data = pd.concat(
    [
        career_data,
        pd.DataFrame(interests_encoded, columns=mlb_interests.classes_),
        pd.DataFrame(skills_encoded, columns=mlb_skills.classes_),
    ],
    axis=1,
)

# Drop original Interests and Skills columns
career_data = career_data.drop(columns=["Interests", "Skills"])

# Drop columns that are no longer needed
columns_to_drop = [
    "What is your name?",
    "What is your gender?",
    "Did you do any certification courses additionally?",
    "If yes, please specify your certificate course title.",
    "Are you working?",
    job_title_column,
    "Have you done masters after undergraduation? If yes, mention your field of masters.(Eg; Masters in Mathematics)",
]

career_data = career_data.drop(columns=columns_to_drop)

# Encode remaining categorical columns if any
for column in career_data.columns:
    if pd.api.types.is_object_dtype(career_data[column]):
        career_data[column] = LabelEncoder().fit_transform(
            career_data[column].astype(str)
        )

# Ensure only numeric data before scaling
non_numeric_cols = career_data.select_dtypes(exclude=["number"]).columns
if len(non_numeric_cols) > 0:
    print(f"Non-numeric columns found and will be dropped: {non_numeric_cols.tolist()}")
    career_data = career_data.drop(columns=non_numeric_cols)

# Save the final processed feature names used during training
final_feature_names = career_data.drop(columns=["First Job Title"]).columns.tolist()

# Scale the numeric features
scaler = StandardScaler()
X = scaler.fit_transform(career_data.drop(columns=["First Job Title"]))

# Define the feature set and target variable
y = career_data["First Job Title"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")


# Function to recommend a career based on user input
def recommend_career(input_data):
    input_df = pd.DataFrame([input_data])

    # Handle unseen labels in UG Course and UG Specialization
    input_df["UG Course"] = input_df["UG Course"].apply(
        lambda x: (
            course_encoder.transform([x])[0] if x in course_encoder.classes_ else -1
        )
    )
    input_df["UG Specialization"] = input_df["UG Specialization"].apply(
        lambda x: (
            specialization_encoder.transform([x])[0]
            if x in specialization_encoder.classes_
            else -1
        )
    )

    input_df = pd.concat(
        [
            input_df,
            pd.DataFrame(
                mlb_interests.transform(input_df["Interests"]),
                columns=mlb_interests.classes_,
            ),
            pd.DataFrame(
                mlb_skills.transform(input_df["Skills"]), columns=mlb_skills.classes_
            ),
        ],
        axis=1,
    )

    input_df = input_df.drop(columns=["Interests", "Skills"])

    for column in input_df.columns:
        if pd.api.types.is_object_dtype(input_df[column]):
            input_df[column] = LabelEncoder().fit_transform(
                input_df[column].astype(str)
            )

    # Ensure the input DataFrame has the same feature names in the same order as the training data
    missing_cols = set(final_feature_names) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0  # Add missing columns with default value 0

    input_df = input_df[final_feature_names]

    # Reorder columns to match training order
    input_df = input_df.reindex(columns=final_feature_names)

    input_scaled = scaler.transform(input_df)
    career = model.predict(input_scaled)
    return job_title_encoder.inverse_transform(career)


# Example usage
example_input = {
    "UG Course": "B.Sc",
    "UG Specialization": "Computer Applications",
    "Interests": ["Cloud computing"],
    "Skills": ["Linux", "Git"],
    "CGPA": 8.5,
    "Certification": "Yes",
}

predicted_career = recommend_career(example_input)
print(f"Suggested Career: {predicted_career}")
