import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Load the trained model
model = joblib.load('c:/Users/Priyansh Das/Downloads/dt_model.pkl')  # Replace '/content/dt_model.pkl' with the actual path to your saved model file

#age vs target
def create_age_vs_heart_disease_plot(df, user_age=None):
    age_groups = df.groupby(pd.cut(df['age'], bins=[20, 30, 40, 50, 60, 70, 80, 90]))['target'].mean()
    fig = go.Figure(data=go.Bar(x=age_groups.index.astype(str), y=age_groups.values * 100, marker=dict(color='red')))
    
    # Highlight user input age with a different color
    if user_age is not None:
        age_group_labels = ['(20, 30]', '(30, 40]', '(40, 50]', '(50, 60]', '(60, 70]', '(70, 80]', '(80, 90]']
        user_age_index = age_group_labels.index(user_age)
        fig.update_traces(marker=dict(color=['deeppink'] * user_age_index + ['skyblue'] + ['deeppink'] * (len(age_group_labels) - user_age_index - 1)))
    
    fig.update_layout(title_text='Percentage of Heart Disease Presence in Different Age Groups',
                      title_font=dict(size=24),
                      xaxis_title='Age Group',
                      yaxis_title='Percentage (%)')
    return fig


def create_age_vs_heart_disease_data(df, selected_age=None):
    age_groups = df.groupby(pd.cut(df['age'], bins=[20, 30, 40, 50, 60, 70, 80, 90]))['target'].mean()

    # Create a highlight_color list with the same length as age_groups
    highlight_color = ['blue' if age_interval.left == selected_age else 'red' for age_interval in age_groups.index]

    return age_groups.index.astype(str), age_groups.values * 100, highlight_color


df = pd.read_csv("heart.csv")
# Function to make prediction
def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    # Create a new numpy array with the input data
    new_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]

    # Make predictions
    predictions = model.predict(new_data)

    # Return the predicted class (convert 0 to "No" and 1 to "Yes")
    return "Yes" if predictions[0] == 1 else "No"

# Streamlit app
def main():
    st.title("Heart Disease Prediction")

    # Input widgets (same as before)
    age = st.slider("Age", 20, 100, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.slider("Resting Blood Pressure", 90, 200, 120)
    chol = st.slider("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar", [0, 1])
    restecg = st.selectbox("Resting ECG", [0, 1, 2])
    thalach = st.slider("Maximum Heart Rate", 50, 220, 150)
    exang = st.selectbox("Exercise-Induced Angina", [0, 1])
    oldpeak = st.slider("ST Depression", 0.0, 6.2, 1.0)
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.slider("Number of Major Vessels", 0, 3, 1)
    thal = st.slider("Thalassemia Type", 0, 3, 2)

    # "Submit" button
    if st.button("Submit"):
        # Make a prediction
        prediction = predict_heart_disease(age, 1 if sex == "Male" else 0, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)

        # Display the result
        st.subheader("Prediction Result:")
        if prediction == 'Yes':
            # If prediction is "Yes", display in red
            result_text = f"<h3 style='text-align: center; color: red;'>Heart Disease: {prediction}</h3>"
        else:
            # If prediction is "No", display in green
            result_text = f"<h3 style='text-align: center; color: lightgreen;'>Heart Disease: {prediction}</h3>"

        # Use Markdown to display the formatted text
        st.markdown(result_text, unsafe_allow_html=True)
        
        # If age is not None, show the highlighted bar plot
        fig_age_vs_heart_disease = create_age_vs_heart_disease_plot(df, user_age=f'({(age//10)*10}, {(age//10)*10+10}]')
        st.plotly_chart(fig_age_vs_heart_disease, use_container_width=True, key='age_vs_heart_disease')

        # Show the selected age
        st.write(f"Selected Age: {age}")

        # Create a scatter plot for correlation between 'age' and 'chol'
        fig = go.Figure(data=go.Scatter(x=df['age'], y=df['chol'], mode='markers', marker=dict(color='yellow', size=8), showlegend=False))
        fig.update_layout(title_text='Correlation Between Age and Cholesterol',
                          title_font=dict(size=24),
                          xaxis_title='Age',
                          yaxis_title='Cholesterol')

        st.plotly_chart(fig)

        # Create a histogram for resting blood pressure distribution by heart disease
        fig_resting_bp = px.histogram(df, x='trestbps', color='target', marginal='rug', nbins=10, title='Resting Blood Pressure Distribution by Heart Disease')
        fig_resting_bp.update_layout(title_font=dict(size=28))  # Increase the title font size
        st.plotly_chart(fig_resting_bp, use_container_width=True)
        
        # Create the Pair Plot
        fig_pair_plot = px.scatter_matrix(df, dimensions=['age', 'trestbps', 'chol', 'thalach', 'oldpeak'], color='target', title='Pair Plot')
        fig_pair_plot.update_layout(title_font=dict(size=28))
        st.plotly_chart(fig_pair_plot, use_container_width=True)

        # Create the Box Plot
        fig_box_plot = px.box(df, x='slope', y='oldpeak', color='target', title='Slope vs. Oldpeak by Heart Disease')
        fig_box_plot.update_layout(title_font=dict(size=24))  # Increase the title font size
        st.plotly_chart(fig_box_plot, use_container_width=True)

if __name__ == "__main__":
    main()
